# Line-by-Line Explanation of `main.py`

This document explains every line and block in `main.py` for beginners. Each section is described in simple terms.

---

## 1. Imports

```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # stabilize OpenMP on macOS
```
- `import os`: Lets you interact with your operating system.
- `os.environ.setdefault(...)`: Sets an environment variable to help with stability on macOS.

### Standard Library Imports
```python
from functools import lru_cache
from hashlib import sha1
from datetime import datetime, timedelta
import uuid, io, base64, json, random
```
- These are built-in Python modules for caching, hashing, working with dates, generating unique IDs, and handling data.

### Typing & FastAPI/Pydantic
```python
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
```
- `typing`: Helps define types for variables and functions.
- `FastAPI`: A web framework for building APIs.
- `CORSMiddleware`: Allows your API to be accessed from other domains (like your frontend).
- `BaseModel`: Used for data validation.

### ML/Math/Visualization
```python
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```
- These are libraries for machine learning, math, and plotting graphs.

---

## 2. FastAPI App Setup

```python
app = FastAPI(title="FCC Snapshot API (ML, FHIR, Batch)", version="0.4.0")
```
- Creates the FastAPI app with a title and version.

### CORS Middleware
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[...],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
- Allows your API to be called from specific frontend URLs.

---

## 3. Constants and Dictionaries

Lists of event types, lab tests, codes, and other data used to generate synthetic (fake) medical data.

---

## 4. Utility Functions

### `random_date_within(days)`
Returns a random date within the last `days` days.

### `random_iso_within(days)`
Returns a random ISO-formatted date/time within the last `days` days.

### `_seed_from_member(member_id)`
Creates a random seed based on a member's ID for reproducible results.

### `generate_member_profile(member_id)`
Creates a fake member profile with age, sex, and health conditions.

### `sample_lab_value(...)`
Generates a fake lab test result based on the member's profile.

### `gen_events(...)`
Generates a list of fake medical events (lab results, appointments, etc.) for a member.

---

## 5. Feature Engineering

### `extract_features(events)`
Extracts numerical features from the events (like counts of abnormal labs, recent discharges, etc.) for use in machine learning.

---

## 6. Model Training and Caching

### `train_model(seed)`
Trains a synthetic XGBoost model using generated data.

### `get_model_and_explainer()`
Caches the trained model and SHAP explainer for reuse.

---

## 7. Data Models (Pydantic)

Defines the structure of API responses and requests using Pydantic classes:
- `SnapshotResponse`
- `TimelineResponse`
- `BatchRequest`

---

## 8. SHAP and Narrative Helpers

### `shap_image_base64(...)`
Creates a bar chart image showing feature impacts and encodes it as base64.

### `build_narratives(...)`
Generates human-readable explanations for the model's predictions.

---

## 9. Recommendations Helper

### `build_next_steps(...)`
Suggests next steps for care based on the member's profile and events.

---

## 10. API Endpoints

### `/api/members/{member_id}/timeline`
Returns a timeline of events for a member.

### `/api/members/{member_id}/tick`
Returns a single event for a member.

### `/api/members/{member_id}/snapshot`
Returns a risk snapshot for a member, including model predictions and explanations.

### `/api/members/{member_id}/fhir`
Returns a FHIR bundle (standard medical data format) for a member.

### `/api/members/snapshot/batch`
Returns risk snapshots for a batch of members.

### `/healthz`
Health check endpoint (returns `{"ok": True}` if the server is running).

---

## 11. FHIR Export Helper

### `to_fhir_bundle(...)`
Converts member data and events into a FHIR-compliant bundle for interoperability.

---

## Summary
- The code creates a synthetic medical API using FastAPI.
- It generates fake member profiles and events, extracts features, trains a model, and provides predictions and explanations.
- The API supports multiple endpoints for timelines, snapshots, batch processing, and FHIR export.

If you want a deeper explanation of any specific function or block, let me know!
