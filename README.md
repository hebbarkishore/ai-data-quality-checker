# 🧠 AI Data Quality Checker

Streamlit web app to assess data quality: missingness, duplicates, schema/type issues, numeric outliers, drift against a reference dataset, and potential PII using regex patterns.

## ✨ Features
- Missing values & % by column
- Duplicate row count + examples
- Schema & sample value per column
- Cardinality & constant-column checks
- Numeric summary (count, mean, std, min/max, missing %)
- Outliers via Isolation Forest (configurable contamination) or IQR
- Drift vs. reference dataset (PSI for numeric, Jensen–Shannon for categorical)
- PII scan (emails, phones, SSN, credit-card patterns)
- One-click CSV/JSON exports
- Dockerized, with tests

## 🚀 Quickstart
```bash
# 1) Clone
git clone https://github.com/YOUR-USER/ai-data-quality-checker.git
cd ai-data-quality-checker

# 2) (Option A) Local
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

# 2) (Option B) Docker
docker build -t dqc:latest .
docker run --rm -p 8501:8501 dqc:latest
```

Open http://localhost:8501, upload a **current** CSV and (optionally) a **reference** CSV to compute drift.

## 📦 Sample Data
Two toy CSVs in `sample_data/` show anomalies (missing, outliers, duplicates, PII) for demo screenshots.

## 🧪 Tests
```bash
pytest -q
```

## 🧰 Tech Stack
Python, Streamlit, Pandas, NumPy, Scikit‑learn, SciPy

## 🔒 Notes
- PII scan uses simple regex heuristics; validate with your compliance/legal team before production use.
- PSI/JSD thresholds are defaults; tune to your domain.

## 📜 License
MIT
