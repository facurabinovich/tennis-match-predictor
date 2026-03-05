# 🎾 ATP Tennis Match Predictor

> End-to-end machine learning system for ATP tennis match prediction — from raw data ingestion and feature engineering through model training, MySQL storage, and a live Streamlit application.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-FINAL%20MODEL-2ecc71?style=flat)](https://lightgbm.readthedocs.io)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=flat&logo=mysql&logoColor=white)](https://mysql.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)

---

## Overview

This project builds a production-ready ATP tennis match prediction system trained on **66,703 matches (2000–2025)**. It predicts the winner of any ATP match given pre-match information — rankings, Elo ratings, recent form, surface performance, H2H history, and serve/return statistics.

**Final model: LightGBM with GridSearch tuning**
- **67.59% test accuracy** on completely unseen 2025 data
- **0.7424 ROC-AUC**
- 85 engineered features
- Val-test gap of 0.67% — excellent temporal stability

The system includes a live data pipeline that pulls current ATP matches daily and a five-page Streamlit app for match prediction and player/season analytics.

---

## Project Structure

```
├── tennis-notebook.ipynb     # Full ML pipeline: EDA → feature engineering → modeling
├── app_database.py           # Streamlit application (5 pages)
├── data_updater.py           # Daily ATP data pipeline (tennismylife.org)
├── load_initial_data.py      # One-time historical data loader
├── config.py                 # DB connection settings
├── models/
│   ├── lgbm_final.pkl        # Trained LightGBM model
│   ├── feature_set.pkl       # Feature list + metadata
│   ├── model_metrics.pkl     # Performance metrics
│   └── feature_importance.pkl
└── tml-data/                 # Raw ATP match CSVs (2000–2026)
```

---

## ML Pipeline

### Problem Framing
Player A / Player B assignment is **randomized** at training time, ensuring the model learns genuine predictive signal rather than position artifacts. Target = 1 if Player A wins, 0 if Player B wins. ~50% balance by design.

### Feature Engineering (85 features)

| Category | Features | Notes |
|----------|----------|-------|
| **Elo ratings** | Overall + surface Elo, Elo diffs | Variable K-factor: GS=40, Masters=36, 500=32, 250=28 |
| **Form** | 5-match overall, surface, level form | Tournament-tier form captures Slam vs. 250 specialists |
| **Extended surface form** | 20-match surface window | Identifies surface specialists missed by short windows |
| **Serve stats** | 1st serve %, won %, BP save %, ace rate, DF rate | Career averages from full match history |
| **Return stats** | 1st/2nd return won %, BP conversion % | Derived from opponent serve stats |
| **H2H** | Career + last-3 H2H wins | Surface-breakdown available |
| **Context** | Rank, points, inactivity, indoor, surface, round, level | Ordinal-encoded tournament level |
| **Age flags** | `is_teen` (< 20), `is_veteran` (> 35) | Binary flags replace continuous age — see Design Decisions |

### Model Progression

| Model | Val Acc | AUC | Notes |
|-------|---------|-----|-------|
| Rank Baseline (heuristic) | 64.01% | — | Always pick better-ranked player |
| Logistic Regression | 65.10% | 0.7152 | +1.09pp over baseline |
| XGBoost (tuned + Elo) | 66.94% | 0.7483 | Elo features drive improvement |
| XGBoost (tuned + Advanced) | 67.32% | 0.7503 | |
| LightGBM (default) | 67.47% | 0.7602 | |
| LightGBM (tuned) | 67.58% | 0.7576 | |
| LR (13 significant features) | 65.55% | 0.7267 | Feature selection experiment |
| LightGBM GridSearch (continuous age) | 69.35% | 0.7667 | Biased — see below |
| **LightGBM GridSearch (age flags)** | **68.26%** | **0.7599** | **Production model** |

### Design Decisions

**Age feature revision:** The continuous `age_diff` feature ranked #2 in feature importance but introduced a systematic bias — the model always heavily favoured younger players regardless of form or Elo. A 25-year-old ranked #77 with 2 wins in his last 10 was predicted to beat a 32-year-old ranked #64 with 5 wins in his last 10 at **84.1% confidence**. The root cause: career-stage trends in 25 years of training data don't generalise to individual match predictions.

Replacing continuous age with binary `is_teen` / `is_veteran` flags cost -1.09pp in validation accuracy but reduced the val-test gap from 1.68pp to **0.67pp** and eliminated the directional bias. On unseen 2025 data the gap is just 0.08pp (67.59% vs 67.67%).

**Prediction ceiling:** ~68% accuracy is considered near-optimal for pre-match ATP prediction. The remaining ~32% error rate is largely driven by upsets — 57.1% of errors were high-confidence predictions where the lower-ranked player won. No pre-match tabular feature captures in-match momentum, injury, or psychological factors.

---

## Database Architecture (MySQL)

```sql
players        — 1,647 ATP players (nationality, hand, height, birth_date)
matches        — 66,703 matches with full serve stats (ace, DF, BP saved/faced)
match_features — 85 pre-computed features per match (model input)
player_stats   — current metrics + career serve/return averages per player
player_elo     — Elo ratings by surface, updated after each match
h2h_history    — head-to-head records with surface breakdown
```

Age is stored as `birth_date` in the `players` table and computed on-the-fly via `TIMESTAMPDIFF(YEAR, birth_date, CURDATE())` — never stored as a static column.

---

## Streamlit Application

Five-page app built with Streamlit + Plotly:

| Page | Description |
|------|-------------|
| 🏠 Home | Season overview, recent matches, live tournament tracker |
| 🎾 Predict Match | Pre-match analysis + LightGBM prediction with win probability |
| 👤 Player Stats | Elo progression chart, surface breakdown, season record |
| 📅 Season Dashboard | Monthly activity, Grand Slam winners, serve/return leaders, upsets |
| ℹ️ About | Model documentation and tech stack |

The **Predict Match** page includes a full pre-match report before the prediction: season record, recent matches, inactivity warnings, tournament history, and H2H breakdown.

---

## Data Pipeline

`data_updater.py` runs daily against [tennismylife.org](https://stats.tennismylife.org) — a zero-lag ATP data source with 50 columns and official ATP player IDs. It:

1. Fetches ongoing tournament CSV
2. Identifies new matches not yet in the database
3. Computes Elo updates, form metrics, and all 85 features
4. Inserts into MySQL transactionally

Player IDs from tennismylife.org match the historical dataset exactly, eliminating name-variant issues (e.g. "Alcaraz" vs "Carlos Alcaraz").

---

## Setup

### Requirements
```
Python 3.12
MySQL 8.0
```

### Install
```bash
pip install -r requirements.txt
```

### Configure
```bash
# Create config.py with your MySQL credentials
DB_HOST     = "localhost"
DB_USER     = "your_user"
DB_PASSWORD = "your_password"
DB_NAME     = "tennis_db"
```

### Run
```bash
# Load historical data (one-time)
python load_initial_data.py

# Update with latest matches
python data_updater.py

# Launch app
streamlit run app_database.py
```

---

## Key Learnings

**Data leakage is subtle.** Early models hit ~100% accuracy because the Winner/Loser column structure leaked the outcome directly. Restructuring to a neutral Player A/B format with random assignment was the first critical fix.

**Feature engineering beats model complexity.** The same LightGBM architecture went from 67.58% to 68.26% purely through better features — variable K-factor Elo, tournament-level form, and extended surface windows. Hyperparameter tuning contributed less than expected.

**High importance ≠ good feature.** `age_diff` was #2 by importance but was creating harmful predictions. Feature importance measures how often a feature is used, not whether it's being used correctly.

**Schema integrity matters early.** Catching and fixing schema mismatches (missing columns, stale rankings, empty tables) before the pipeline is fully built prevents compounding issues downstream.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| ML | LightGBM, scikit-learn, XGBoost |
| Data | pandas, numpy |
| Database | MySQL 8.0 |
| App | Streamlit, Plotly |
| Environment | WSL2 / Ubuntu |

---

## Author

**Facundo Rabinovich**
Systems Engineer with ~2 years in data analytics, transitioning into Data Science.
Building projects in analytics, ML, and data engineering.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-facundo--rabinovich-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/facundo-rabinovich-9a2541242)
[![GitHub](https://img.shields.io/badge/GitHub-facurabinovich-181717?style=flat&logo=github&logoColor=white)](https://github.com/facurabinovich)