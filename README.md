# 🏨 StaySmart Hotels — Cancellation Risk Prediction
### Data Preprocessing & Feature Engineering Capstone 

> **Graded Assignment — BS in Data Science & AI** 
> Predicting hotel booking cancellations (`is_canceled`) using the Hotel Bookings dataset.  
> Core goal: prove that **feature engineering + preprocessing drives performance** — not model complexity.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VkxsXzo0ibb-xCJyqLP3Rn90QX5UtjsE?usp=sharing)

---

## 🎯 Final Results at a Glance

| Stage | Features | Preprocessing | Model | ROC-AUC | F1 |
|-------|----------|---------------|-------|---------|-----|
| ① Baseline | Raw (label-encoded) | Simple impute | Logistic Regression | 0.8563 | 0.6912 |
| ② Pipeline | All numeric + OHE | Impute + Scale + OHE | Logistic Regression | 0.9903 | 0.9450 |
| ③ + Construction | +8 constructed features | Full pipeline + RF | Random Forest (all) | 0.9967 | 0.9772 |
| ④ + Selection | Top 17 features | Full pipeline + RF | Random Forest (selected) | **0.9974** | **0.9770** |

> 📈 **ROC-AUC lift from baseline → final: +0.1411 (+16.5% relative improvement)**

---

## 📁 Repo Structure

```
FeatureEngineering_Capstone/
│
├── FeatureEngineering_Capstone.ipynb   ← Main notebook (run this!)
│
├── src/
│   ├── __init__.py                     ← Package init
│   └── helpers.py                      ← Reusable utilities:
│                                           evaluate_model()
│                                           safe_agg_feature()
│                                           plot_roc_curve()
│                                           bar_compare()
│
├── report/
│   ├── README.md                       ← Report folder guide
│   └── Report.pdf                      ← Full write-up (export from notebook)
│
├── requirements.txt                    ← Python dependencies
├── .gitignore                          ← Standard Python ignores
└── README.md                           ← This file
```

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended, zero setup)

Click the **Open in Colab** badge above, then:
```
Runtime → Run all
```
The dataset loads automatically — no file uploads needed.

### Option 2 — Run Locally

**Step 1 — Clone**
```bash
git clone https://github.com/cybersd12321/FeatureEngineering_Capstone.git
cd FeatureEngineering_Capstone
```

**Step 2 — Virtual environment**
```bash
# macOS / Linux
python3 -m venv venv && source venv/bin/activate

# Windows
python -m venv venv && venv\Scripts\activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Launch Jupyter**
```bash
jupyter notebook FeatureEngineering_Capstone.ipynb
```

**Step 5 — Run all cells**
```
Kernel → Restart & Run All
```

> ⚡ No local data file needed. Dataset is fetched automatically from:
> ```
> https://raw.githubusercontent.com/swapnilsaurav/Dataset/refs/heads/master/hotel_bookings.csv
>
> Expected runtime: ~3–5 minutes on a standard laptop.

---

## 📋 Tasks Covered

| # | Task | Key Output | 
|---|------|-----------|---------------
| 1 | Baseline Model + "What is a Feature?" | Confusion matrix · ROC-AUC = 0.8563 | 
| 2 | Curse of Dimensionality Demo | Distance distribution plots (2 → 200 dims) | 
| 3 | Numeric Preprocessing | Binning · binarisation · scaler comparison | 
| 4 | Distance/Proximity Metrics | KNN: No scaling vs Standard vs Robust + Euclidean vs Manhattan |
| 5 | End-to-End Numeric Pipeline | ColumnTransformer + 5-fold CV → AUC = 0.9903 | 
| 6 | Feature Extraction | 7 DateTime features + OHE vs Target Encoding | 
| 7 | Feature Construction | 8 constructed features + leakage prevention section | 
| 8 | Feature Importance + Selection | RF + MI + Permutation + Chi-square · final 17 features | 
| ★ | **Final Comparison** | 4-stage ROC-AUC / F1 dashboard | 


---

## 🔧 Helper Functions (`src/helpers.py`)

```python
from src.helpers import evaluate_model, safe_agg_feature, plot_roc_curve, bar_compare

# Fit and evaluate any classifier in one line
results = evaluate_model("My Model", clf, X_train, X_test, y_train, y_test)
# → {'Model': ..., 'Accuracy': ..., 'ROC-AUC': ..., 'F1': ...}

# Leakage-safe group aggregation (train-only)
df, agg = safe_agg_feature(train_idx, df, "country", "is_canceled", "country_cancel_rate")

# Overlay multiple ROC curves on one axes
plot_roc_curve(ax, [res1, res2, res3], title="ROC Curves")

# Grouped bar chart for model comparison
bar_compare(results_df, title="Before vs After")
```

---

## 📊 Executive Summary

The biggest performance gains came from **preprocessing and feature extraction**, not from the model.

- **Biggest single jump — Task 5 (Pipeline):** ROC-AUC jumped from 0.8563 → 0.9903. Proper imputation + scaling + OHE explained most of the lift before any hand-crafted features.
- **Feature construction (Task 7)** pushed further to 0.9967 — domain-aware features like `lead_time_bucket`, `price_per_person`, and `country_cancel_rate` genuinely carry cancellation signal.
- **Feature selection (Task 8)** trimmed from 27 → 17 features with **zero AUC loss**, proving many raw columns become redundant once constructed features exist.

**Most business-actionable features:**
- `lead_time` / `lead_time_bucket` → require deposits for 30–90 day booking window (highest cancel risk)
- `country_cancel_rate` → proactively contact guests from high-risk origin countries
- `price_per_person` → offer early-bird incentives to budget-sensitive bookings
- `booking_changes` → flag frequent changers for personalised retention outreach

---

## 📦 Dependencies

| Package | Version | Used for |
|---------|---------|----------|
| `pandas` | 2.1.4 | Data loading & manipulation |
| `numpy` | 1.26.4 | Numerical operations |
| `matplotlib` | 3.8.2 | All visualisations |
| `scikit-learn` | 1.4.2 | Models, pipelines, metrics, preprocessing |
| `scipy` | 1.13.0 | Distance metrics (`cdist`) |
| `jupyter` | 1.0.0 | Notebook runtime (local) |

---

## ✅ Submission Checklist

- [x] GitHub repo is **public**
- [x] Notebook **runs end-to-end** (execution counts 1→11, zero errors)
- [x] `requirements.txt` present
- [x] `/report` folder present
- [x] **Final comparison table included** (4-stage dashboard in Final Task cell)
- [x] `README.md` with full local run instructions
- [x] `/src` folder with reusable helper functions
- [ ] Add your Google Colab share link below
- [ ] Upload `Report.pdf` to `/report/`
- [ ] Submit PDF to Lumen with both links

---

## 📎 Links

| Resource | URL |
|----------|-----|
| 📓 Google Colab | https://colab.research.google.com/drive/1VkxsXzo0ibb-xCJyqLP3Rn90QX5UtjsE?usp=sharing |
| 🗄️ Dataset | [hotel_bookings.csv](https://raw.githubusercontent.com/swapnilsaurav/Dataset/refs/heads/master/hotel_bookings.csv) |
| 👤 GitHub Profile | [cybersd12321](https://github.com/cybersd12321) |

---

*Submitted as part of Week 7 Graded Assignment — BS Data Science & AI, Year 1.*
