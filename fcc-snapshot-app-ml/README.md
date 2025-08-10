
# Family Care Central — AI Health Record Summarizer & Risk Predictor (ML)

Full-stack demo (React + FastAPI) with **XGBoost** risk prediction and **SHAP** explanations.

## Run

### Backend
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd ../frontend
npm install
npm run dev
```
Open http://localhost:5173

> Frontend expects API at http://localhost:8000
