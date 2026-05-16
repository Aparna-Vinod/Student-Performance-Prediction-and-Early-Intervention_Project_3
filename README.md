# Student-Performance-Prediction-and-Early-Intervention_Project_3
# 🎓 Student Performance Prediction & Early Intervention

---

## 👥 Team Members

| Member | Register Number |
|--------| -------- |
| **Abhi** | 253226 |
| **Aparna** | 253015 |
| **Jia** | 253116 |

---

## 📌 Problem Statement

Student dropout and academic failure remain persistent challenges in secondary education. Early identification of at-risk students allows counselors and educators to intervene before irreversible damage occurs to a student's academic trajectory.

**Goal:** Build a binary classification system that predicts whether a student will **pass or fail** their final exam using demographics, study habits, attendance, family background, and interim academic performance.

**Why This Matters:**
- Early warning systems can reduce failure rates by 20–30% *(Romero & Ventura, 2010)*
- Counselors have limited bandwidth — ML helps prioritize students who need help most
- Personalized intervention recommendations increase counselor effectiveness

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/student+performance) |
| **File** | `student-mat.csv` (Mathematics course) |
| **Records** | 395 students |
| **Features** | 33 attributes (demographics, grades, lifestyle, family) |
| **Target** | Binary: `pass = 1` if G3 ≥ 10, else `pass = 0` |
| **Class Distribution** | ~52% Pass / ~48% Fail |

**Key Features Used:**
- `G1`, `G2` — First and second period exam grades
- `studytime` — Weekly study time (1–4 scale)
- `failures` — Number of past class failures
- `absences` — School absences count
- `higher` — Wants to pursue higher education
- `Medu`, `Fedu` — Mother and father education levels
- **Engineered:** `avg_grade`, `avg_parent_edu`, `alcohol_total`, `social_index`

---

## 🔄 Data Science Life Cycle

All 10 stages are fully implemented in the Jupyter notebook:

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Problem Definition & Literature Review | ✅ |
| 2 | Data Collection & Data Understanding | ✅ |
| 3 | Data Preprocessing & Cleaning | ✅ |
| 4 | Exploratory Data Analysis (EDA) | ✅ |
| 5 | Feature Engineering & Selection | ✅ |
| 6 | Model Building & Training | ✅ |
| 7 | Model Evaluation & Comparison | ✅ |
| 8 | Model Interpretation & Explainability (LIME) | ✅ |
| 9 | Deployment (Streamlit) | ✅ |
| 10 | Documentation (README, PPT, GitHub) | ✅ |

---

## 🤖 Models & Results

### Models Trained
1. **Decision Tree** — max_depth=5, min_samples_leaf=5
2. **Logistic Regression** — C=1.0, max_iter=1000
3. **Gradient Boosting** ⭐ Best — 100 estimators, lr=0.1

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Decision Tree | 0.820 | 0.800 | 0.790 | 0.795 | 0.840 |
| Logistic Regression | 0.850 | 0.840 | 0.830 | 0.835 | 0.890 |
| **Gradient Boosting** | **0.880** | **0.870** | **0.860** | **0.865** | **0.920** |

### Class Imbalance Handling
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance training data
- Evaluated on unmodified test set for realistic performance measurement

### Feature Selection
- Used **Mutual Information** to score all features
- Selected features with MI score > 0.02
- Added 4 **engineered features**: `avg_grade`, `avg_parent_edu`, `alcohol_total`, `social_index`

### Explainability
- **LIME (Local Interpretable Model-agnostic Explanations)** used for per-student predictions
- Counselors receive specific, actionable reasons for each at-risk student prediction

---

## 🚀 Deployment

### Live App
🌐 **https://tneg733ztk9xbdz8e42x7q.streamlit.app/**

### App Features
- Input student details via interactive sidebar sliders
- View Pass/Fail prediction with confidence probability bar
- Read LIME-based intervention tips tailored to each student
- Compare model performance metrics
- Handles edge cases gracefully



---

## 🛠️ Setup & Run Locally

### Prerequisites
- Python 3.10+
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-team/student-performance-prediction.git
cd student-performance-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Go to: https://archive.ics.uci.edu/ml/datasets/student+performance
# Download student-mat.csv and place it in /data/

# 4. Run the notebook (trains and saves the model)
jupyter notebook notebooks/student_performance_prediction.ipynb

# 5. Launch Streamlit app
cd streamlit_app
streamlit run app.py
```

---

## 📁 Repository Structure

```
student-performance-prediction/
├── data/
│   └── student-mat.csv              # UCI dataset (download separately)
├── models/
│   ├── gradient_boosting_model.pkl  # Saved best model
│   ├── scaler.pkl                   # Feature scaler
│   └── selected_features.pkl        # Feature list
├── notebooks/
│   └── student_performance_prediction.ipynb  # Full pipeline
├── streamlit_app/
│   └── app.py                       # Streamlit deployment
├── docs/
│   ├── class_distribution.png
│   ├── grade_distributions.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── lime_explanation_fail.png
│   └── gb_feature_importance.png
├── individual_profiles/
│   ├── abhi_github_activity.png
│   ├── aparna_github_activity.png
│   └── jia_github_activity.png
├── Student_Performance_Prediction.pptx
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
lime>=0.2.0
xgboost>=1.7.0
joblib>=1.2.0
streamlit>=1.28.0
jupyter>=1.0.0
```

---

## 📊 Key Findings

1. **G1 & G2 are the strongest predictors** — Mid-term performance is highly predictive of final outcomes
2. **Past failures significantly increase risk** — Each failure reduces pass probability by ~15-20%
3. **Study time matters** — Students studying >5h/week are 30% more likely to pass
4. **Higher education aspiration** — Students aiming for higher education perform significantly better
5. **Absenteeism is a warning sign** — >10 absences strongly correlates with failing

---

## 📚 References

1. Romero, C., & Ventura, S. (2010). Educational data mining: A review of the state of the art. *IEEE TSMCS.*
2. Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. *EUROSIS.*
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). LIME: Local interpretable model-agnostic explanations. *KDD.*

---

Thank You

# Team Members Contribution
| Member | Contribution |
| --- | --- |
| Jia | Step 1 : Problem Definition & Litrature Review , Stage 2: Data Collection & Data Understanding , Stage 3: Data Preprocessing & Cleaning , Stage 4: Exploratory Data Analysis (EDA) , Stage 10: Documentation (README) |
| Abhi | Stage 5: Feature Engineering & Selection , Stage 6: Model Building & Training , Stage 7: Model Evaluation & Comparison , Stage 10: Documentation (README, PPT) |
| Aparna | Stage 8: Model Interpretation & Explainability , Stage 9: Deployment (Streamlit or equivalent) , Stage 10: Documentation (README, GitHub) |



