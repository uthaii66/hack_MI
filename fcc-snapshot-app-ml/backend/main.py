# --- Import required libraries ---
from fastapi import FastAPI, Query  # FastAPI for building the API and Query for params
from fastapi.middleware.cors import CORSMiddleware  # CORS for frontend-backend communication
from pydantic import BaseModel  # For data validation and serialization
from typing import List, Dict, Any  # Type hints
from datetime import datetime, timedelta  # Date/time utilities
import os
# Set OpenMP thread count to 1 for stability on macOS (fixes some XGBoost/SHAP issues)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from functools import lru_cache  # For caching model/explainer
import random  # For generating synthetic data

import numpy as np  # Numerical operations
import xgboost as xgb  # ML model
import shap  # Model explainability

from hashlib import sha1

# --- Initialize FastAPI app ---
app = FastAPI(title="FCC Snapshot API with XGBoost + SHAP", version="0.2.0")


# --- Allow frontend apps to access the API (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Synthetic event and lab definitions for demo (enriched) ---
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
    {"id": "FAC1003", "name": "Sunrise Clinic"}
]

# --- Demographics & condition-aware distributions ---
def _seed_from_member(member_id: str) -> int:
    """Deterministic seed per member to keep data stable across requests."""
    return int(sha1(member_id.encode("utf-8")).hexdigest(), 16) % (2**31-1)

def generate_member_profile(member_id: str) -> Dict[str, Any]:
    """Create a deterministic synthetic profile with age/sex/conditions."""
    rnd = random.Random(_seed_from_member(member_id))
    sex = rnd.choice(["F","M"])
    age = rnd.randint(22, 87)
    # Chronic conditions sampled with simple priors
    has_t2d = rnd.random() < 0.35
    has_htn = rnd.random() < 0.55
    has_hld = rnd.random() < 0.45  # hyperlipidemia
    conditions = []
    if has_t2d: conditions.append("T2D")
    if has_htn: conditions.append("HTN")
    if has_hld: conditions.append("HLD")
    # Baseline tendencies (shift lab/vital means)
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

def sample_lab_value(name: str, lo: float, hi: float, profile: Dict[str, Any]) -> (float, str):
    """Draw a lab value using condition-aware distributions and return (value, flag)."""
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
        # general case: biased toward normal with 20% abnormal tails
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


# --- Utility: Generate a random date within N days ago ---
def random_date_within(days: int) -> str:
    """
    Generate a random date string within the last `days` days (YYYY-MM-DD).
    """
    d = datetime.utcnow() - timedelta(days=random.randint(0, days))
    return d.strftime("%Y-%m-%d")

def random_iso_within(days: int) -> str:
    """ISO‑8601 timestamp within last `days` days with seconds precision."""
    delta_days = random.randint(0, days)
    delta_secs = random.randint(0, 24*3600 - 1)
    d = datetime.utcnow() - timedelta(days=delta_days, seconds=delta_secs)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Generate a list of synthetic events for a member ---
def gen_events(n: int = 12, days: int = 120, profile: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Generate a list of synthetic clinical events for a member with enriched payloads,
    condition-aware distributions, and *correlated* follow-on events.
    """
    if profile is None:
        profile = generate_member_profile("ANON")

    evs: List[Dict[str, Any]] = []
    # Primary pass: generate base events with condition-aware values
    for _ in range(n):
        t = random.choice(EVENT_TYPES)
        payload: Dict[str, Any] = {}

        if t == "LAB_RESULT":
            name, unit, (lo, hi) = random.choice(LABS)
            val, flag = sample_lab_value(name, lo, hi, profile)
            payload = {
                "test_name": name,
                "value": val,
                "unit": unit,
                "ref_range": f"{lo}-{hi} {unit}",
                "flag": flag,
                "performing_lab": random.choice(["Pinecrest Diagnostics", "MetroLab" ]),
                "lo": lo,
                "hi": hi,
                "observation_id": f"OBS{random.randint(100000,999999)}"
            }

        elif t in ["ADT_ADMIT", "ADT_DISCHARGE"]:
            fac = random.choice(FACILITIES)
            payload = {
                "facility_id": fac["id"],
                "facility": fac["name"],
                "reason": random.choice(ADMIT_REASONS) if t == "ADT_ADMIT" else None,
                "disposition": random.choice(DISPOSITIONS) if t == "ADT_DISCHARGE" else None
            }

        elif t == "MED_REFILL":
            drug = random.choice(["Metformin 500 mg","Atorvastatin 20 mg","Lisinopril 10 mg"])
            # If T2D present, bias toward Metformin refills
            if "T2D" in profile.get("conditions", []) and random.random() < 0.6:
                drug = "Metformin 500 mg"
            payload = {
                "drug": drug,
                "days_supply": random.choice([30,60,90]),
                "pharmacy": random.choice(PHARMACIES),
                "refill_date": random_iso_within(days),
                "adherence_estimate": round(random.uniform(0.6, 1.0), 2)
            }

        elif t == "CLAIM":
            pos = random.choice(["11-Office","23-ED","22-OP Hospital"])
            # If HTN and vitals high → more office visits; if recent ADT → ED
            cpt = random.choice(CPT_CODES)
            payload = {
                "cpt": cpt,
                "icd10": random.sample(ICD10_CODES, k=random.randint(1,2)),
                "amount": round(random.uniform(45, 350),2),
                "place_of_service": pos
            }

        elif t == "VITALS":
            sbp = max(90, int(random.gauss(profile["tendencies"]["sbp_mean"], profile["tendencies"]["sbp_sd"])) )
            dbp = max(50, int(random.gauss(profile["tendencies"]["dbp_mean"], profile["tendencies"]["dbp_sd"])) )
            payload = {
                "bp_systolic": sbp,
                "bp_diastolic": dbp,
                "heart_rate": random.randint(55, 110),
                "resp_rate": random.randint(12, 22),
                "spo2": random.randint(92, 100),
                "temp_c": round(random.uniform(36.0, 38.5), 1),
                "bmi": round(random.uniform(18.5, 36.0), 1),
                "source": random.choice(["Home device", "Clinic"])
            }

        elif t == "APPOINTMENT":
            start_iso = random_iso_within(days)
            appt_type = random.choice(APPT_TYPES)
            # If T2D and recent abnormal A1C → bias to Endocrinology or Telehealth follow-up
            if "T2D" in profile.get("conditions", []) and random.random() < 0.5:
                appt_type = random.choice(["Endocrinology", "Telehealth Follow-up"])    
            payload = {
                "appt_type": appt_type,
                "provider": random.choice(["Dr. M. Patel","Dr. L. Chen","NP R. Gomez"]),
                "location": random.choice(["Oak Valley Medical Office", "Telehealth"]),
                "start_iso": start_iso,
                "status": random.choice(["Scheduled","Completed","No-Show"]) if random.random() < 0.9 else "Rescheduled"
            }

        evs.append({
            "ts": random_date_within(days),           # legacy YYYY-MM-DD date
            "ts_iso": random_iso_within(days),        # ISO-8601
            "type": t,
            "payload": payload
        })

    # Secondary pass: create *correlated* follow-up events (e.g., high A1C → refill + claim + appt)
    # Look for the most recent abnormal A1C
    recent_a1c_abnormal = next((e for e in evs if e["type"]=="LAB_RESULT" and e["payload"]["test_name"]=="A1C" and e["payload"]["flag"]=="H"), None)
    if recent_a1c_abnormal:
        # Add Metformin refill if not already present
        if not any(e["type"]=="MED_REFILL" and "Metformin" in e["payload"].get("drug","") for e in evs):
            evs.append({
                "ts": random_date_within(days),
                "ts_iso": random_iso_within(days),
                "type": "MED_REFILL",
                "payload": {
                    "drug": "Metformin 500 mg",
                    "days_supply": random.choice([30,60,90]),
                    "pharmacy": random.choice(PHARMACIES),
                    "refill_date": random_iso_within(days),
                    "adherence_estimate": round(random.uniform(0.6, 1.0), 2)
                }
            })
        # Add endocrinology claim
        evs.append({
            "ts": random_date_within(days),
            "ts_iso": random_iso_within(days),
            "type": "CLAIM",
            "payload": {
                "cpt": "99214",
                "icd10": ["E11.9"],
                "amount": round(random.uniform(85, 250),2),
                "place_of_service": "11-Office"
            }
        })
        # Add a specialist appointment
        evs.append({
            "ts": random_date_within(days),
            "ts_iso": random_iso_within(days),
            "type": "APPOINTMENT",
            "payload": {
                "appt_type": "Endocrinology",
                "provider": random.choice(["Dr. M. Patel","Dr. L. Chen"]),
                "location": random.choice(["Oak Valley Medical Office", "Telehealth"]),
                "start_iso": random_iso_within(days),
                "status": random.choice(["Scheduled","Completed"])  # bias to scheduled
            }
        })

    # Sort by ISO timestamp (most recent first)
    evs.sort(key=lambda e: e["ts_iso"], reverse=True)
    return evs


# --- Feature names for ML model ---
FEATURE_NAMES = [
    "recent_discharge",
    "abnormal_lab_count",
    "claims_last_60d",
    "med_refill_count",
    "creatinine_trend",
    "low_spo2_count",
    "high_bp_count"
]


# --- Extract ML features from a list of events ---
def extract_features(events: List[Dict[str, Any]]):
    """
    Extract ML features from a list of events.
    Features:
        1. Recent discharge (last 30 days)
        2. Abnormal lab count
        3. Claims in last 60 days
        4. Medication refill count
        5. Creatinine trend (slope over last 3 values)
        6. Count of vitals with SpO2 < 94
        7. Count of vitals with BP systolic > 140 or diastolic > 90
    Args:
        events (List[Dict[str, Any]]): List of event dicts.
    Returns:
        List[float]: List of feature values.
    """
    # 1. Was there a recent discharge (last 30 days)?
    recent_discharge = int(any(e["type"]=="ADT_DISCHARGE" and (datetime.utcnow() - datetime.strptime(e["ts"], "%Y-%m-%d")).days <= 30 for e in events))
    # 2. How many abnormal labs?
    abn_count = sum(1 for e in events if e["type"]=="LAB_RESULT" and e["payload"].get("flag") in ["H","L"])
    # 3. How many claims in last 60 days?
    claims_60 = sum(1 for e in events if e["type"]=="CLAIM" and (datetime.utcnow() - datetime.strptime(e["ts"], "%Y-%m-%d")).days <= 60)
    # 4. How many medication refills?
    refill_count = sum(1 for e in events if e["type"]=="MED_REFILL")

    # 5. Creatinine trend (slope over last 3 values, if available)
    creatinine_labs = [
        (e["ts_iso"], e["payload"]["value"])
        for e in events
        if e["type"] == "LAB_RESULT" and e["payload"].get("test_name") == "Creatinine"
        and isinstance(e["payload"].get("value"), (int, float))
    ]
    # Sort by timestamp descending (most recent first)
    creatinine_labs.sort(key=lambda x: x[0], reverse=True)
    last3 = creatinine_labs[:3]
    if len(last3) >= 2:
        # Convert ISO times to ordinal for regression
        from datetime import datetime as dt
        xs = []
        ys = []
        for ts, val in last3:
            try:
                xs.append(dt.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").timestamp())
                ys.append(val)
            except Exception:
                continue
        if len(xs) >= 2:
            # Fit simple linear regression (slope)
            xarr = np.array(xs)
            yarr = np.array(ys)
            xmean = xarr.mean()
            ymean = yarr.mean()
            denom = ((xarr - xmean) ** 2).sum()
            slope = float(((xarr - xmean) * (yarr - ymean)).sum() / denom) if denom > 0 else 0.0
            # Scale slope to "per day" for interpretability
            slope_per_day = slope * 86400 if slope != 0 else 0.0
            creatinine_trend = slope_per_day
        else:
            creatinine_trend = 0.0
    else:
        creatinine_trend = 0.0

    # 6. Count of vitals events with SpO2 < 94
    low_spo2_count = sum(
        1 for e in events
        if e["type"] == "VITALS" and isinstance(e["payload"].get("spo2"), (int, float)) and e["payload"]["spo2"] < 94
    )

    # 7. Count of vitals events with BP systolic > 140 or diastolic > 90
    high_bp_count = sum(
        1 for e in events
        if e["type"] == "VITALS"
        and (
            (isinstance(e["payload"].get("bp_systolic"), (int, float)) and e["payload"]["bp_systolic"] > 140) or
            (isinstance(e["payload"].get("bp_diastolic"), (int, float)) and e["payload"]["bp_diastolic"] > 90)
        )
    )
    return [
        recent_discharge,
        abn_count,
        claims_60,
        refill_count,
        creatinine_trend,
        low_spo2_count,
        high_bp_count
    ]


# --- Train an XGBoost model and SHAP explainer on synthetic data ---
def train_model(seed: int = 42):
    """
    Train an XGBoost multiclass classifier and SHAP explainer on synthetic data.
    The model predicts risk class (LOW, MEDIUM, HIGH) based on extracted features.
    Returns:
        model: Trained XGBoost model.
        explainer: SHAP TreeExplainer for model interpretability.
    """
    random.seed(seed); np.random.seed(seed)
    X, y = [], []
    # Weighted sum of features to simulate risk with noise
    weights = np.array([2.0, 1.5, 1.5, 1.0, 2.5, 1.2, 1.2])  # arbitrary weights for realism
    for _ in range(800):
        ev = gen_events(14)
        f = extract_features(ev)
        f_arr = np.array(f)
        # Add some noise to simulate real-world label fuzziness
        noise = np.random.normal(0, 1.0)
        score = float(np.dot(weights, f_arr)) + noise
        # Risk class thresholds (tune as needed)
        if score > 8.5:
            label = 2  # HIGH
        elif score > 4.5:
            label = 1  # MEDIUM
        else:
            label = 0  # LOW
        X.append(f)
        y.append(label)
    X = np.array(X); y = np.array(y)
    # Train XGBoost model
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    params = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss", "max_depth": 3, "eta": 0.3}
    model = xgb.train(params, dtrain, num_boost_round=50)
    # Create SHAP explainer for model interpretability
    explainer = shap.TreeExplainer(model)
    return model, explainer


# --- Cache model and explainer so they're only trained once ---
@lru_cache(maxsize=1)
def get_model_and_explainer():
    """
    Cache and return the trained model and SHAP explainer.
    Ensures training only happens once per process.
    Returns:
        Tuple[model, explainer]: Trained model and explainer.
    """
    return train_model()


# --- Response schemas for API endpoints ---

class SnapshotResponse(BaseModel):
    """
    Response schema for the /snapshot endpoint.
    Contains member ID, timestamp, risk prediction, and summary.
    """
    member_id: str
    generated_at: str
    risk: Dict[str, Any]
    summary: Dict[str, Any]


class TimelineResponse(BaseModel):
    """
    Response schema for the /timeline endpoint.
    Contains member ID and a list of events.
    """
    member_id: str
    events: List[Dict[str, Any]]


# --- API endpoint: Get timeline of synthetic events for a member ---
@app.get("/api/members/{member_id}/timeline", response_model=TimelineResponse)
def get_timeline(member_id: str, days: int = Query(120, ge=1, le=365), n: int = Query(12, ge=1, le=200)):
    """Return an enriched synthetic timeline with optional window and count."""
    profile = generate_member_profile(member_id)
    return {"member_id": member_id, "events": gen_events(n=n, days=days, profile=profile)}


@app.get("/api/members/{member_id}/tick")
def tick(member_id: str, days: int = Query(3, ge=1, le=120)):
    """Emit a single newest event for polling demos (near real-time feel)."""
    profile = generate_member_profile(member_id)
    return {"member_id": member_id, "event": gen_events(n=1, days=days, profile=profile)[0]}


# --- API endpoint: Get risk snapshot and summary for a member ---
@app.get("/api/members/{member_id}/snapshot", response_model=SnapshotResponse)
def get_snapshot(member_id: str):
    import json
    import os
    import io
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Generate member profile for this member
    profile = generate_member_profile(member_id)
    # Generate synthetic events for this member
    events = gen_events(14, days=120, profile=profile)
    # Save synthetic events to a JSON file in ../frontend/src/ with member name
    frontend_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/src'))
    os.makedirs(frontend_src_dir, exist_ok=True)
    json_path = os.path.join(frontend_src_dir, f'{member_id}.json')
    with open(json_path, 'w') as f:
        json.dump(events, f, indent=2)
    """
    API endpoint: Get risk snapshot and summary for a member.
    Generates synthetic events, extracts features, predicts risk,
    computes SHAP values for feature impact, and builds a summary.
    Args:
        member_id (str): Member identifier.
    Returns:
        dict: Snapshot response with risk and summary.
    """
    # Get cached model and explainer
    model, explainer = get_model_and_explainer()
    # Extract features for ML model
    feats = extract_features(events)
    fvec = np.array(feats).reshape(1,-1)
    dtest = xgb.DMatrix(fvec, feature_names=FEATURE_NAMES)
    # Predict risk probabilities for each class
    probs = model.predict(dtest)[0]  # [LOW, MEDIUM, HIGH] probs
    pred_class = int(np.argmax(probs))
    label_map = {0:"LOW",1:"MEDIUM",2:"HIGH"}
    label = label_map[pred_class]

    # Compute SHAP values and build human-readable top factors
    shap_plot_b64 = None
    narratives = []
    try:
        # SHAP values explain the impact of each feature on the prediction
        shap_values = explainer.shap_values(fvec, check_additivity=False)
        shap_class = shap_values[pred_class] if isinstance(shap_values, list) else shap_values
        sv = np.ravel(np.asarray(shap_class)).astype(float)
        order = np.argsort(np.abs(sv))[::-1]
        top_idx = [int(i) for i in order[:3]]
        factors = [f"{FEATURE_NAMES[i]}: impact {sv[i]:.2f}" for i in top_idx]

        # SHAP summary plot for this prediction (single row)
        plt.figure(figsize=(6, 2.5))
        shap.summary_plot(shap_class, fvec, feature_names=FEATURE_NAMES, show=False, plot_type="bar")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        shap_plot_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # Human-readable narratives based on top features and their values
        for i in top_idx:
            fname = FEATURE_NAMES[i]
            val = feats[i]
            impact = sv[i]
            if fname == "recent_discharge" and val > 0:
                narratives.append("Recent discharge")
            elif fname == "abnormal_lab_count" and val > 0:
                narratives.append(f"{int(val)} abnormal lab results")
            elif fname == "claims_last_60d" and val > 0:
                narratives.append(f"{int(val)} claims in last 60 days")
            elif fname == "med_refill_count" and val > 0:
                narratives.append(f"{int(val)} medication refills")
            elif fname == "creatinine_trend":
                if val > 0.03:
                    narratives.append("Rising creatinine levels")
                elif val < -0.03:
                    narratives.append("Falling creatinine levels")
            elif fname == "low_spo2_count" and val > 0:
                narratives.append(f"{int(val)} low SpO₂ readings (<94%)")
            elif fname == "high_bp_count" and val > 0:
                narratives.append(f"{int(val)} high blood pressure readings")
        # Fallback in case no narrative is found
        if not narratives:
            narratives.append("No major risk factors identified")
    except Exception:
        # Graceful fallback if SHAP has an issue (e.g., version mismatch)
        feats = extract_features(events)
        labels = [
            "recent discharge (≤30d)",
            "abnormal labs count",
            "claims in last 60d",
            "med refills count",
            "creatinine trend",
            "low SpO2 count",
            "high BP count"
        ]
        order = np.argsort(np.abs(np.asarray(feats)))[::-1]
        top_idx = [int(i) for i in order[:3]]
        factors = [f"{labels[i]}: value {feats[i]}" for i in top_idx]
        narratives = [labels[i] for i in top_idx]
        shap_plot_b64 = None

    # Build a simple summary for the frontend
    abnormal_labs = [{
        "name": e["payload"]["test_name"],
        "value": e["payload"]["value"],
        "unit": e["payload"]["unit"],
        "flag": e["payload"]["flag"]
    } for e in events if e["type"]=="LAB_RESULT" and e["payload"].get("flag") in ["H","L"]][:5]

    summary = {
        "recent_events": events[:5],  # Most recent 5 events
        "key_findings": [
            f"{len(abnormal_labs)} abnormal lab(s) in last 120 days",
            f"{sum(1 for e in events if e['type']=='ADT_ADMIT')} admissions; {sum(1 for e in events if e['type']=='ADT_DISCHARGE')} discharges"
        ],
        "abnormal_labs": abnormal_labs,
        "recommended_next_steps": ["Follow-up per care plan"]
    }

    # Return the full snapshot response
    resp = {
        "member_id": member_id,
        "profile": profile,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "risk": {
            "score": float(max(probs)),
            "label": label,
            "factors": factors,
            "narratives": narratives
        },
        "summary": summary
    }
    if shap_plot_b64 is not None:
        resp["risk"]["shap_plot"] = shap_plot_b64
    return resp


# --- Health check endpoint for monitoring ---
@app.get("/healthz")
def healthz():
    """
    Health check endpoint for monitoring.
    Returns:
        dict: Always {"ok": True}
    """
    return {"ok": True}
