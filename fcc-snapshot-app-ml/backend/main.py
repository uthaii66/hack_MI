import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # stabilize OpenMP on macOS

# --- stdlib
from functools import lru_cache
from hashlib import sha1
from datetime import datetime, timedelta
import uuid, io, base64, json, random

# --- typing & pydantic/fastapi
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query, Body, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- ML / math / viz
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import requests
from jose import jwt, jwk

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(title="FCC Snapshot API (ML, FHIR, Batch)", version="0.4.0")

# CORS: use env ALLOWED_ORIGINS, fallback to local dev URLs; no wildcard
origins_env = os.getenv("ALLOWED_ORIGINS", "").split(",")
allowed_origins = [o.strip() for o in origins_env if o.strip()] or [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["authorization", "content-type"],
)

# ------------------------------------------------------------------------------
# Auth (OIDC/JWT via JWKS) + Structured logging
# ------------------------------------------------------------------------------
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "true").lower() == "true"
OIDC_ISS = os.getenv("OIDC_ISS")
OIDC_AUD = os.getenv("OIDC_AUD")
OIDC_JWKS_URL = os.getenv("OIDC_JWKS_URL")

# Structured logger (PII/PHI scrubbed)
logger = logging.getLogger("fcc.ai")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(message)s'))
if not logger.handlers:
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

def log_event(event: str, **kv):
    redacted_keys = {"profile", "events", "payload", "shap_plot"}
    safe = {k: ("[redacted]" if k in redacted_keys else v) for k, v in kv.items()}
    logger.info(json.dumps({"event": event, **safe}))

# JWKS cache
_JWKS = {"keys": []}
_JWKS_TS = 0

def _load_jwks(force: bool = False):
    global _JWKS, _JWKS_TS
    if not OIDC_JWKS_URL:
        return
    now = int(datetime.utcnow().timestamp())
    if force or (now - _JWKS_TS > 300):
        resp = requests.get(OIDC_JWKS_URL, timeout=5)
        resp.raise_for_status()
        _JWKS = resp.json()
        _JWKS_TS = now

def _pem_for_kid(kid: str) -> bytes:
    _load_jwks()
    for k in _JWKS.get("keys", []):
        if k.get("kid") == kid:
            return jwk.construct(k).to_pem()
    _load_jwks(force=True)
    for k in _JWKS.get("keys", []):
        if k.get("kid") == kid:
            return jwk.construct(k).to_pem()
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Signing key not found")

@app.middleware("http")
async def authn(request: Request, call_next):
    # Allow health without auth or disable globally via env
    # Also bypass auth automatically in non-prod or when OIDC is not configured
    if (
        not AUTH_REQUIRED
        or request.url.path == "/healthz"
        or os.getenv("APP_ENV", "dev").lower() != "prod"
        or not (OIDC_ISS and OIDC_AUD and OIDC_JWKS_URL)
    ):
        return await call_next(request)

    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]

    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if not kid:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing kid")
        pub_pem = _pem_for_kid(kid)
        claims = jwt.decode(
            token,
            pub_pem,
            algorithms=["RS256", "RS384", "RS512", "ES256", "ES384"],
            audience=OIDC_AUD,
            issuer=OIDC_ISS,
        )
        request.state.user = (
            claims.get("preferred_username")
            or claims.get("upn")
            or claims.get("email")
            or claims.get("sub")
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[AUTH] JWT validation failed: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return await call_next(request)

# ------------------------------------------------------------------------------
# Synthetic dictionaries / constants
# ------------------------------------------------------------------------------
EVENT_TYPES = [
    "ADT_ADMIT", "ADT_DISCHARGE", "LAB_RESULT", "MED_REFILL", "CLAIM",
    "VITALS", "APPOINTMENT"
]
LABS = [
    ("Hemoglobin","g/dL",(11.5,16.0)),
    ("Ferritin","ng/mL",(20,250)),
    ("A1C","%",(4.0,5.6)),
    ("Creatinine","mg/dL",(0.6,1.3)),
    ("Sodium","mmol/L",(135,145)),
    ("Potassium","mmol/L",(3.5,5.1)),
]
ICD10_CODES = ["E11.9","I10","E78.5","J06.9","M54.5","R53.83","R73.03","D50.9","E03.9"]
CPT_CODES = ["99213","99214","83036","85025","80050"]
APPT_TYPES = ["Primary Care", "Endocrinology", "Cardiology", "Telehealth Follow-up", "Lab Draw"]
DISPOSITIONS = ["Home", "SNF", "Rehab", "Home w/ Home Health"]
ADMIT_REASONS = ["Chest pain", "Hyperglycemia", "Syncope", "Shortness of breath", "Fever"]
PHARMACIES = ["GoodHealth Pharmacy", "CareFirst Rx", "MediPlus Pharmacy"]
FACILITIES = [
    {"id": "FAC1001", "name": "Oak Valley Hospital"},
    {"id": "FAC1002", "name": "Riverbend Medical"},
    {"id": "FAC1003", "name": "Sunrise Clinic"},
]

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def random_date_within(days: int) -> str:
    d = datetime.utcnow() - timedelta(days=random.randint(0, days))
    return d.strftime("%Y-%m-%d")

def random_iso_within(days: int) -> str:
    delta_days = random.randint(0, days)
    delta_secs = random.randint(0, 24*3600 - 1)
    d = datetime.utcnow() - timedelta(days=delta_days, seconds=delta_secs)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")

def _seed_from_member(member_id: str) -> int:
    return int(sha1(member_id.encode("utf-8")).hexdigest(), 16) % (2**31-1)

def generate_member_profile(member_id: str) -> Dict[str, Any]:
    rnd = random.Random(_seed_from_member(member_id))
    sex = rnd.choice(["F","M"])
    age = rnd.randint(22, 87)
    has_t2d = rnd.random() < 0.35
    has_htn = rnd.random() < 0.55
    has_hld = rnd.random() < 0.45
    conditions = []
    if has_t2d: conditions.append("T2D")
    if has_htn: conditions.append("HTN")
    if has_hld: conditions.append("HLD")
    tendencies = {
        "a1c_mean": 7.2 if has_t2d else 5.4,
        "a1c_sd": 0.8 if has_t2d else 0.3,
        "hgb_mean": 12.8 if sex=="F" else 14.2,
        "hgb_sd": 1.1,
        "sbp_mean": 142 if has_htn else 122,
        "sbp_sd": 12,
        "dbp_mean": 88 if has_htn else 76,
        "dbp_sd": 8,
    }
    return {
        "member_id": member_id,
        "name": f"Member {member_id[-4:]}",
        "age": age,
        "sex": sex,
        "conditions": conditions,
        "tendencies": tendencies,
    }

def sample_lab_value(name: str, lo: float, hi: float, profile: Dict[str, Any]):
    rnd = random.Random()
    rnd.seed(random.getrandbits(32))
    if name == "A1C":
        mean = profile["tendencies"]["a1c_mean"]
        sd = profile["tendencies"]["a1c_sd"]
        val = max(3.5, rnd.gauss(mean, sd))
    elif name == "Hemoglobin":
        mean = profile["tendencies"]["hgb_mean"]
        sd = profile["tendencies"]["hgb_sd"]
        val = max(7.5, rnd.gauss(mean, sd))
    else:
        if rnd.random() < 0.2:
            tail = rnd.choice(["low","high"])
            if tail == "low":
                val = rnd.uniform(max(lo*0.5, lo-5*(hi-lo)), lo*0.95)
            else:
                val = rnd.uniform(hi*1.05, hi*1.5)
        else:
            val = rnd.uniform(lo, hi)
    flag = "L" if val < lo else ("H" if val > hi else "N")
    return round(val, 2), flag

def gen_events(n: int = 12, days: int = 120, profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if profile is None:
        profile = generate_member_profile("ANON")

    evs: List[Dict[str, Any]] = []
    for _ in range(n):
        t = random.choice(EVENT_TYPES)
        payload: Dict[str, Any] = {}

        if t == "LAB_RESULT":
            name, unit, (lo, hi) = random.choice(LABS)
            val, flag = sample_lab_value(name, lo, hi, profile)
            payload = {
                "test_name": name, "value": val, "unit": unit,
                "ref_range": f"{lo}-{hi} {unit}", "flag": flag,
                "performing_lab": random.choice(["Pinecrest Diagnostics","MetroLab"]),
                "lo": lo, "hi": hi, "observation_id": f"OBS{random.randint(100000,999999)}"
            }

        elif t in ["ADT_ADMIT","ADT_DISCHARGE"]:
            fac = random.choice(FACILITIES)
            payload = {
                "facility_id": fac["id"], "facility": fac["name"],
                "reason": random.choice(ADMIT_REASONS) if t == "ADT_ADMIT" else None,
                "disposition": random.choice(DISPOSITIONS) if t == "ADT_DISCHARGE" else None
            }

        elif t == "MED_REFILL":
            drug = random.choice(["Metformin 500 mg","Atorvastatin 20 mg","Lisinopril 10 mg"])
            if "T2D" in profile.get("conditions", []) and random.random() < 0.6:
                drug = "Metformin 500 mg"
            payload = {
                "drug": drug, "days_supply": random.choice([30,60,90]),
                "pharmacy": random.choice(PHARMACIES),
                "refill_date": random_iso_within(days),
                "adherence_estimate": round(random.uniform(0.6, 1.0), 2)
            }

        elif t == "CLAIM":
            payload = {
                "cpt": random.choice(CPT_CODES),
                "icd10": random.sample(ICD10_CODES, k=random.randint(1,2)),
                "amount": round(random.uniform(45,350), 2),
                "place_of_service": random.choice(["11-Office","23-ED","22-OP Hospital"])
            }

        elif t == "VITALS":
            sbp = max(90, int(random.gauss(profile["tendencies"]["sbp_mean"], profile["tendencies"]["sbp_sd"])))
            dbp = max(50, int(random.gauss(profile["tendencies"]["dbp_mean"], profile["tendencies"]["dbp_sd"])))
            payload = {
                "bp_systolic": sbp, "bp_diastolic": dbp,
                "heart_rate": random.randint(55,110),
                "resp_rate": random.randint(12,22),
                "spo2": random.randint(92,100),
                "temp_c": round(random.uniform(36.0,38.5),1),
                "bmi": round(random.uniform(18.5,36.0),1),
                "source": random.choice(["Home device","Clinic"])
            }

        elif t == "APPOINTMENT":
            start_iso = random_iso_within(days)
            appt_type = random.choice(APPT_TYPES)
            if "T2D" in profile.get("conditions", []) and random.random() < 0.5:
                appt_type = random.choice(["Endocrinology","Telehealth Follow-up"])
            payload = {
                "appt_type": appt_type,
                "provider": random.choice(["Dr. M. Patel","Dr. L. Chen","NP R. Gomez"]),
                "location": random.choice(["Oak Valley Medical Office","Telehealth"]),
                "start_iso": start_iso,
                "status": random.choice(["Scheduled","Completed","No-Show"]) if random.random() < 0.9 else "Rescheduled"
            }

        evs.append({
            "ts": random_date_within(days),
            "ts_iso": random_iso_within(days),
            "type": t,
            "payload": payload
        })

    # Correlated follow-ups: abnormal A1C → Metformin + Endo claim + Endo appt
    recent_a1c_abnormal = next((e for e in evs if e["type"]=="LAB_RESULT"
                                and e["payload"]["test_name"]=="A1C"
                                and e["payload"]["flag"]=="H"), None)
    if recent_a1c_abnormal:
        if not any(e["type"]=="MED_REFILL" and "Metformin" in e["payload"].get("drug","") for e in evs):
            evs.append({
                "ts": random_date_within(days), "ts_iso": random_iso_within(days),
                "type": "MED_REFILL",
                "payload": {
                    "drug": "Metformin 500 mg",
                    "days_supply": random.choice([30,60,90]),
                    "pharmacy": random.choice(PHARMACIES),
                    "refill_date": random_iso_within(days),
                    "adherence_estimate": round(random.uniform(0.6,1.0),2)
                }
            })
        evs.append({
            "ts": random_date_within(days), "ts_iso": random_iso_within(days),
            "type": "CLAIM",
            "payload": {"cpt":"99214","icd10":["E11.9"],"amount":round(random.uniform(85,250),2),"place_of_service":"11-Office"}
        })
        evs.append({
            "ts": random_date_within(days), "ts_iso": random_iso_within(days),
            "type": "APPOINTMENT",
            "payload": {
                "appt_type":"Endocrinology",
                "provider": random.choice(["Dr. M. Patel","Dr. L. Chen"]),
                "location": random.choice(["Oak Valley Medical Office","Telehealth"]),
                "start_iso": random_iso_within(days),
                "status": random.choice(["Scheduled","Completed"])
            }
        })

    evs.sort(key=lambda e: e["ts_iso"], reverse=True)
    return evs

# ------------------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------------------
FEATURE_NAMES = [
    "recent_discharge","abnormal_lab_count","claims_last_60d","med_refill_count",
    "creatinine_trend","low_spo2_count","high_bp_count"
]

def _last_creatinines(events: List[Dict[str, Any]], k: int = 3) -> List[float]:
    labs = [e for e in events if e["type"]=="LAB_RESULT" and e["payload"].get("test_name")=="Creatinine"]
    labs.sort(key=lambda e: e["ts_iso"])
    vals = [e["payload"]["value"] for e in labs[-k:]]
    return vals

def _slope(values: List[float]) -> float:
    if len(values) < 2: return 0.0
    xs = np.arange(len(values))
    m, _ = np.polyfit(xs, np.array(values, dtype=float), 1)
    return float(m)

def extract_features(events: List[Dict[str, Any]]) -> List[float]:
    recent_discharge = int(any(
        e["type"]=="ADT_DISCHARGE" and
        (datetime.utcnow() - datetime.strptime((e["ts_iso"] or f"{e['ts']}T00:00:00Z")[:10], "%Y-%m-%d")).days <= 30
        for e in events
    ))
    abn_count = sum(1 for e in events if e["type"]=="LAB_RESULT" and e["payload"].get("flag") in ["H","L"])
    claims_60 = sum(1 for e in events if e["type"]=="CLAIM" and
                    (datetime.utcnow() - datetime.strptime((e["ts_iso"] or f"{e['ts']}T00:00:00Z")[:10], "%Y-%m-%d")).days <= 60)
    refill_count = sum(1 for e in events if e["type"]=="MED_REFILL")

    # Trend + vitals features
    cr_vals = _last_creatinines(events, 3)
    creatinine_trend = _slope(cr_vals)  # rising positive = worse
    low_spo2_count = sum(1 for e in events if e["type"]=="VITALS" and isinstance(e["payload"].get("spo2"), int) and e["payload"]["spo2"] < 94)
    high_bp_count = sum(1 for e in events if e["type"]=="VITALS" and (
        (isinstance(e["payload"].get("bp_systolic"), int) and e["payload"]["bp_systolic"] > 140) or
        (isinstance(e["payload"].get("bp_diastolic"), int) and e["payload"]["bp_diastolic"] > 90)
    ))

    return [recent_discharge, abn_count, claims_60, refill_count, creatinine_trend, low_spo2_count, high_bp_count]

# ------------------------------------------------------------------------------
# Model training / caching
# ------------------------------------------------------------------------------
def train_model(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    X, y = [], []
    for _ in range(1200):
        profile = generate_member_profile(f"MEM{random.randint(10000,99999)}")
        ev = gen_events(14, days=120, profile=profile)
        f = extract_features(ev)
        # Probabilistic label from weighted sum + noise
        w = np.array([0.9, 0.7, 0.6, -0.3, 1.2, 0.8, 0.8], dtype=float)
        z = float(np.dot(w, np.array(f, dtype=float)) + np.random.normal(0, 0.8))
        # thresholds tuned for synthetic distribution
        label = 2 if z > 3.0 else (1 if z > 1.2 else 0)  # 0 low, 1 medium, 2 high
        X.append(f); y.append(label)
    X = np.array(X, dtype=float); y = np.array(y, dtype=int)
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    params = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss", "max_depth": 3, "eta": 0.3}
    model = xgb.train(params, dtrain, num_boost_round=50)
    # explainer = shap.TreeExplainer(model)
    explainer = shap.Explainer(model)
    return model, explainer

@lru_cache(maxsize=1)
def get_model_and_explainer():
    return train_model()

# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------
class SnapshotResponse(BaseModel):
    member_id: str
    generated_at: str
    profile: Dict[str, Any]
    risk: Dict[str, Any]
    summary: Dict[str, Any]

class TimelineResponse(BaseModel):
    member_id: str
    events: List[Dict[str, Any]]

class BatchRequest(BaseModel):
    member_ids: List[str]

# ------------------------------------------------------------------------------
# Helpers: SHAP image + narratives
# ------------------------------------------------------------------------------
def shap_image_base64(feature_values: List[float], shap_vector: np.ndarray) -> str:
    # Simple bar chart of absolute SHAP impacts (top 7)
    idx = np.argsort(np.abs(shap_vector))[::-1][:7]
    names = [FEATURE_NAMES[int(i)] for i in idx]
    vals = shap_vector[idx]
    fig, ax = plt.subplots(figsize=(4, 2.6), dpi=150)
    colors = ["#ef4444" if v >= 0 else "#10b981" for v in vals]
    ax.barh(range(len(idx))[::-1], np.abs(vals)[::-1], color=colors[::-1])
    ax.set_yticks(range(len(idx))[::-1])
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Impact (|SHAP|)", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def build_narratives(features: List[float], shap_vec: np.ndarray) -> List[str]:
    narratives = []
    fmap = dict(zip(FEATURE_NAMES, features))
    s = dict(zip(FEATURE_NAMES, shap_vec))

    def add(text):
        if text not in narratives:
            narratives.append(text)

    if s.get("recent_discharge", 0) > 0.05 and fmap.get("recent_discharge") == 1:
        add("Recent discharge (≤30d)")
    if s.get("abnormal_lab_count", 0) > 0.05 and fmap.get("abnormal_lab_count", 0) > 0:
        add(f"{int(fmap['abnormal_lab_count'])} abnormal lab(s)")
    if s.get("claims_last_60d", 0) > 0.05 and fmap.get("claims_last_60d", 0) > 0:
        add(f"{int(fmap['claims_last_60d'])} claims in last 60d")
    if s.get("creatinine_trend", 0) > 0.05 and fmap.get("creatinine_trend", 0) > 0:
        add("Rising creatinine levels")
    if s.get("low_spo2_count", 0) > 0.05 and fmap.get("low_spo2_count", 0) > 0:
        add(f"{int(fmap['low_spo2_count'])} low SpO₂ episode(s)")
    if s.get("high_bp_count", 0) > 0.05 and fmap.get("high_bp_count", 0) > 0:
        add(f"{int(fmap['high_bp_count'])} high BP reading(s)")
    if s.get("med_refill_count", 0) < -0.05 and fmap.get("med_refill_count", 0) > 0:
        add("Medication refills (adherent)")

    if not narratives:
        add("Maintain routine care; review at next visit")
    return narratives[:5]

# ------------------------------------------------------------------------------
# Dynamic recommendations helper
# ------------------------------------------------------------------------------
def build_next_steps(profile: Dict[str, Any], events: List[Dict[str, Any]], features: List[float]) -> List[str]:
    """Return prioritized, de-duplicated next steps from features + events."""
    recs: List[str] = []
    fmap = dict(zip(FEATURE_NAMES, features))

    def add(txt: str):
        if txt and txt not in recs:
            recs.append(txt)

    # 1) Transitions of care
    if fmap.get("recent_discharge") == 1:
        add("Schedule 7‑day post‑discharge follow‑up (TCM)")
        add("Care coordinator outreach within 48h to reconcile meds")

    # 2) Vitals-based prompts
    if fmap.get("high_bp_count", 0) >= 2:
        add("Optimize hypertension regimen; add home BP log for 2 weeks")
    elif fmap.get("high_bp_count", 0) == 1:
        add("Recheck BP next visit; consider ambulatory monitoring")

    if fmap.get("low_spo2_count", 0) >= 1:
        add("Assess hypoxia cause; check pulse oximetry and respiratory symptoms")

    # 3) Lab trends and abnormalities
    if fmap.get("creatinine_trend", 0) > 0.05:
        add("Repeat BMP/creatinine in 1–2 weeks; review nephrotoxic meds")

    abnormal_labs = [e for e in events if e.get("type") == "LAB_RESULT" and e.get("payload", {}).get("flag") in ("H","L")]
    for e in abnormal_labs[:5]:
        name = e["payload"].get("test_name")
        flag = e["payload"].get("flag")
        if name == "A1C" and flag == "H":
            add("Endocrinology follow‑up; reinforce diabetes self‑management and Metformin adherence")
        if name == "Sodium" and flag in ("H","L"):
            add("Evaluate dysnatremia; review fluids and medications")
        if name == "Potassium" and flag in ("H","L"):
            add("Address potassium abnormality; check meds (ACEi/diuretics)")
        if name == "Hemoglobin" and flag == "L":
            add("Assess anemia; consider iron studies and stool occult blood")

    # 4) Utilization signal
    if fmap.get("claims_last_60d", 0) >= 3:
        add("Care coordination review of utilization in last 60 days")

    # 5) Medication adherence/context
    if fmap.get("med_refill_count", 0) == 0 and any("T2D" in profile.get("conditions", [])):
        add("Medication review for diabetes; confirm refills and access")

    # Always cap and provide fallback
    if not recs:
        recs = ["Follow‑up per care plan"]
    return recs[:6]

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.post("/api/test_snapshot", response_model=SnapshotResponse)
def test_snapshot(
    payload: dict = Body(...)
):
    model, explainer = get_model_and_explainer()
    member_id = payload.get("member_id", "ANON")
    profile = payload.get("profile", {})
    events = payload.get("events", [])

    # Save full synthetic data in non-prod only
    if os.getenv("APP_ENV", "dev").lower() != "prod":
        try:
            synth_path = os.path.join(os.path.dirname(__file__), f"synthetic_data_{member_id}_test.json")
            with open(synth_path, "w") as f:
                json.dump({
                    "member_id": member_id,
                    "profile": profile,
                    "events": events
                }, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save synthetic data for {member_id}: {e}")

    # Features + predict
    fvec_list = extract_features(events)
    # Save extracted features only in non-prod
    if os.getenv("APP_ENV", "dev").lower() != "prod":
        try:
            save_path = os.path.join(os.path.dirname(__file__), f"features_{member_id}_test.json")
            with open(save_path, "w") as f:
                json.dump({
                    "member_id": member_id,
                    "features": dict(zip(FEATURE_NAMES, fvec_list)),
                    "events_count": len(events)
                }, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save features for {member_id}: {e}")

    fvec = np.array(fvec_list, dtype=float).reshape(1, -1)
    dtest = xgb.DMatrix(fvec, feature_names=FEATURE_NAMES)
    probs = model.predict(dtest)[0]  # [LOW, MEDIUM, HIGH]
    pred_class = int(np.argmax(probs))
    label_map = {0:"LOW",1:"MEDIUM",2:"HIGH"}
    label = label_map[pred_class]
    # SHAP (robust to SHAP/XGBoost versions)
    try:
        res = explainer(fvec)  # new SHAP API
        vals = getattr(res, "values", None)
        if vals is None:
            raise RuntimeError("SHAP returned no values")
        if vals.ndim == 3:
            sv = np.asarray(vals[0, :, pred_class], dtype=float)
        elif vals.ndim == 2:
            sv = np.asarray(vals[0, :], dtype=float)
        else:
            raise RuntimeError(f"Unexpected SHAP values shape: {vals.shape}")
        order = np.argsort(np.abs(sv))[::-1]
        top_idx = [int(i) for i in order[:3]]
        factors = [f"{FEATURE_NAMES[i]}: impact {sv[i]:.2f}" for i in top_idx]
        shap_plot = shap_image_base64(fvec_list, sv)
        narratives = build_narratives(fvec_list, sv)
    except Exception as e:
        print(f"[WARN] SHAP explanation failed: {e}")
        feats = fvec_list
        order = np.argsort(np.abs(np.asarray(feats)))[::-1]
        top_idx = [int(i) for i in order[:3]]
        labels = FEATURE_NAMES
        factors = [f"{labels[i]}: value {feats[i]}" for i in top_idx]
        shap_plot = ""  # degrade gracefully
        narratives = ["Model explanation unavailable — using heuristic factors"]

    # Dynamic recommendations based on features + events
    next_steps = build_next_steps(profile, events, fvec_list)

    # Summary block (unchanged logic; now on richer events)
    abnormal_labs = [{
        "name": e["payload"]["test_name"],
        "value": e["payload"]["value"],
        "unit": e["payload"]["unit"],
        "flag": e["payload"]["flag"]
    } for e in events if e["type"]=="LAB_RESULT" and e["payload"].get("flag") in ["H","L"]][:5]

    summary = {
        "recent_events": events[:5],
        "key_findings": [
            f"{len(abnormal_labs)} abnormal lab(s) in last 120 days",
            f"{sum(1 for e in events if e['type']=='ADT_ADMIT')} admissions; {sum(1 for e in events if e['type']=='ADT_DISCHARGE')} discharges"
        ],
        "abnormal_labs": abnormal_labs,
        "recommended_next_steps": next_steps
    }

    log_event("test_snapshot", member_id=member_id, risk=label)
    return {
        "member_id": member_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "profile": profile,
        "risk": {
            "score": float(max(probs)),
            "label": label,
            "factors": factors,
            "narratives": narratives,
            "shap_plot": shap_plot,
        },
        "summary": summary
    }

@app.get("/api/members/{member_id}/timeline", response_model=TimelineResponse)
def get_timeline(member_id: str, days: int = Query(120, ge=1, le=365), n: int = Query(12, ge=1, le=200)):
    profile = generate_member_profile(member_id)
    return {"member_id": member_id, "events": gen_events(n=n, days=days, profile=profile)}

@app.get("/api/members/{member_id}/tick")
def tick(member_id: str, days: int = Query(3, ge=1, le=120)):
    profile = generate_member_profile(member_id)
    return {"member_id": member_id, "event": gen_events(n=1, days=days, profile=profile)[0]}

@app.get("/api/members/{member_id}/snapshot", response_model=SnapshotResponse)
def get_snapshot(member_id: str):
    model, explainer = get_model_and_explainer()
    profile = generate_member_profile(member_id)
    events = gen_events(14, days=120, profile=profile)

    # Save full synthetic data only in non-prod
    if os.getenv("APP_ENV", "dev").lower() != "prod":
        try:
            synth_path = os.path.join(os.path.dirname(__file__), f"synthetic_data_{member_id}.json")
            with open(synth_path, "w") as f:
                json.dump({
                    "member_id": member_id,
                    "profile": profile,
                    "events": events
                }, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save synthetic data for {member_id}: {e}")

    # Features + predict
    fvec_list = extract_features(events)
    # Save extracted features only in non-prod
    if os.getenv("APP_ENV", "dev").lower() != "prod":
        try:
            save_path = os.path.join(os.path.dirname(__file__), f"features_{member_id}.json")
            with open(save_path, "w") as f:
                json.dump({
                    "member_id": member_id,
                    "features": dict(zip(FEATURE_NAMES, fvec_list)),
                    "events_count": len(events)
                }, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save features for {member_id}: {e}")

    fvec = np.array(fvec_list, dtype=float).reshape(1, -1)
    dtest = xgb.DMatrix(fvec, feature_names=FEATURE_NAMES)
    probs = model.predict(dtest)[0]  # [LOW, MEDIUM, HIGH]
    pred_class = int(np.argmax(probs))
    label_map = {0:"LOW",1:"MEDIUM",2:"HIGH"}
    label = label_map[pred_class]
# SHAP (robust to SHAP/XGBoost versions)
    try:
        res = explainer(fvec)  # new SHAP API
        vals = getattr(res, "values", None)
        # vals shapes:
        #  - multiclass: (n_samples, n_features, n_classes)
        #  - older/binary/regression: (n_samples, n_features)
        if vals is None:
            raise RuntimeError("SHAP returned no values")

        if vals.ndim == 3:
            sv = np.asarray(vals[0, :, pred_class], dtype=float)
        elif vals.ndim == 2:
            sv = np.asarray(vals[0, :], dtype=float)
        else:
            raise RuntimeError(f"Unexpected SHAP values shape: {vals.shape}")

        order = np.argsort(np.abs(sv))[::-1]
        top_idx = [int(i) for i in order[:3]]
        factors = [f"{FEATURE_NAMES[i]}: impact {sv[i]:.2f}" for i in top_idx]
        shap_plot = shap_image_base64(fvec_list, sv)
        narratives = build_narratives(fvec_list, sv)

    except Exception as e:
        print(f"[WARN] SHAP explanation failed: {e}")  # minimal log to console
        feats = fvec_list
        order = np.argsort(np.abs(np.asarray(feats)))[::-1]
        top_idx = [int(i) for i in order[:3]]
        labels = FEATURE_NAMES
        factors = [f"{labels[i]}: value {feats[i]}" for i in top_idx]
        shap_plot = ""  # degrade gracefully
        narratives = ["Model explanation unavailable — using heuristic factors"]

    # Dynamic recommendations based on features + events
    next_steps = build_next_steps(profile, events, fvec_list)

    # Summary block (unchanged logic; now on richer events)
    abnormal_labs = [{
        "name": e["payload"]["test_name"],
        "value": e["payload"]["value"],
        "unit": e["payload"]["unit"],
        "flag": e["payload"]["flag"]
    } for e in events if e["type"]=="LAB_RESULT" and e["payload"].get("flag") in ["H","L"]][:5]

    summary = {
        "recent_events": events[:5],
        "key_findings": [
            f"{len(abnormal_labs)} abnormal lab(s) in last 120 days",
            f"{sum(1 for e in events if e['type']=='ADT_ADMIT')} admissions; {sum(1 for e in events if e['type']=='ADT_DISCHARGE')} discharges"
        ],
        "abnormal_labs": abnormal_labs,
        "recommended_next_steps": next_steps
    }

    log_event("snapshot", member_id=member_id, risk=label)
    return {
        "member_id": member_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "profile": profile,
        "risk": {
            "score": float(max(probs)),
            "label": label,
            "factors": factors,
            "narratives": narratives,
            "shap_plot": shap_plot,
        },
        "summary": summary
    }

# ---------------- FHIR export ----------------
def to_fhir_bundle(member_id: str, profile: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    bundle_id = f"bundle-{uuid.uuid4()}"
    entries = []
    # Patient
    patient_id = f"pat-{member_id}"
    patient = {
        "resourceType": "Patient",
        "id": patient_id,
        "gender": "female" if profile.get("sex") == "F" else "male",
        "birthDate": str(max(1930, 2025 - int(profile.get("age", 40)))),
        "name": [{"use": "official", "text": profile.get("name", f"Member {member_id[-4:]}")}],
        "extension": [{"url": "https://example.org/condition-tags", "valueString": ",".join(profile.get("conditions", []))}],
        "identifier": [{"system": "https://fcc.example/members", "value": member_id}],
    }
    entries.append({"fullUrl": f"urn:uuid:{patient_id}", "resource": patient})
    # Map events
    for e in events:
        t = e.get("type")
        ts = e.get("ts_iso") or f"{e.get('ts')}T00:00:00Z"
        p = e.get("payload", {})
        rid = str(uuid.uuid4())
        if t == "LAB_RESULT":
            entries.append({"fullUrl": f"urn:uuid:{rid}", "resource": {
                "resourceType": "Observation", "id": rid, "status": "final",
                "category": [{"coding": [{"system":"http://terminology.hl7.org/CodeSystem/observation-category","code":"laboratory"}]}],
                "code": {"text": p.get("test_name")},
                "subject": {"reference": f"Patient/{patient_id}"},
                "effectiveDateTime": ts,
                "valueQuantity": {"value": p.get("value"), "unit": p.get("unit")},
                "interpretation": [{"text": {"H":"high","L":"low","N":"normal"}.get(p.get("flag"),"")}],
                "note": [{"text": f"Ref range {p.get('ref_range')}"}],
            }})
        elif t in ("ADT_ADMIT","ADT_DISCHARGE"):
            entries.append({"fullUrl": f"urn:uuid:{rid}", "resource": {
                "resourceType":"Encounter","id":rid,
                "status":"finished" if t=="ADT_DISCHARGE" else "in-progress",
                "class":{"system":"http://terminology.hl7.org/CodeSystem/v3-ActCode","code":"IMP"},
                "subject":{"reference": f"Patient/{patient_id}"},
                "period":{"start": ts},
                "serviceType":{"text": p.get("reason") or p.get("disposition")},
                "location":[{"location":{"display": p.get("facility")}}],
            }})
        elif t == "APPOINTMENT":
            entries.append({"fullUrl": f"urn:uuid:{rid}", "resource": {
                "resourceType":"Appointment","id":rid,
                "status": (p.get("status","scheduled") or "scheduled").lower(),
                "serviceType":[{"text": p.get("appt_type")}],
                "start": p.get("start_iso", ts),
                "participant":[{"actor":{"display": p.get("provider")}, "status":"accepted"}],
                "reasonCode":[{"text":"Follow-up"}],
                "supportingInformation":[{"display": p.get("location")}],
            }})
        elif t == "MED_REFILL":
            entries.append({"fullUrl": f"urn:uuid:{rid}", "resource": {
                "resourceType":"MedicationRequest","id":rid,
                "status":"active","intent":"order",
                "subject":{"reference": f"Patient/{patient_id}"},
                "authoredOn": p.get("refill_date", ts),
                "medicationCodeableConcept":{"text": p.get("drug")},
                "dosageInstruction":[{"text": f"{p.get('days_supply',30)}-day supply"}],
                "performer":{"display": p.get("pharmacy")},
            }})
        elif t == "CLAIM":
            entries.append({"fullUrl": f"urn:uuid:{rid}", "resource": {
                "resourceType":"ExplanationOfBenefit","id":rid,
                "status":"active","type":{"text":"professional"},
                "patient":{"reference": f"Patient/{patient_id}"},"created": ts,"outcome":"complete",
                "item":[{"sequence":1,"productOrService":{"text": p.get("cpt")},"diagnosisSequence":[1]}],
                "diagnosis":[{"sequence":1,"diagnosisCodeableConcept":{"text": ", ".join(p.get("icd10", []))}}],
                "payment":{"amount":{"value": p.get("amount"), "currency":"USD"}},
            }})
        elif t == "VITALS":
            entries.append({"fullUrl": f"urn:uuid:{rid}", "resource": {
                "resourceType":"Observation","id":rid,"status":"final",
                "category":[{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/observation-category","code":"vital-signs"}]}],
                "code":{"text":"Vital signs panel"},"subject":{"reference": f"Patient/{patient_id}"},"effectiveDateTime": ts,
                "component":[
                    {"code":{"text":"Systolic BP"},"valueQuantity":{"value": p.get("bp_systolic"),"unit":"mmHg"}},
                    {"code":{"text":"Diastolic BP"},"valueQuantity":{"value": p.get("bp_diastolic"),"unit":"mmHg"}},
                    {"code":{"text":"SpO2"},"valueQuantity":{"value": p.get("spo2"),"unit":"%"}},
                    {"code":{"text":"Heart rate"},"valueQuantity":{"value": p.get("heart_rate"),"unit":"bpm"}},
                    {"code":{"text":"Temp C"},"valueQuantity":{"value": p.get("temp_c"),"unit":"°C"}},
                ],
            }})
    return {"resourceType":"Bundle","type":"collection","id": bundle_id,"entry": entries}

@app.get("/api/members/{member_id}/fhir")
def fhir_bundle(member_id: str):
    profile = generate_member_profile(member_id)
    events = gen_events(30, days=120, profile=profile)
    return to_fhir_bundle(member_id, profile, events)

@app.post("/api/members/snapshot/batch")
def snapshot_batch(req: BatchRequest):
    model, _ = get_model_and_explainer()
    out = []
    for mid in req.member_ids:
        profile = generate_member_profile(mid)
        events = gen_events(14, days=120, profile=profile)
        fvec = np.array(extract_features(events), dtype=float).reshape(1, -1)
        dtest = xgb.DMatrix(fvec, feature_names=FEATURE_NAMES)
        probs = model.predict(dtest)[0]
        label = ["LOW","MEDIUM","HIGH"][int(np.argmax(probs))]
        out.append({"member_id": mid, "risk": {"score": float(max(probs)), "label": label}})
    return {"results": out}

@app.get("/healthz")
def healthz():
    return {"ok": True}