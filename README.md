# 🏥 KTAS Triage Priority Prediction

### EDA • Data Leakage Audit • XGBoost • CatBoost • LightGBM • Cost-Sensitive Classification

> **Can machine learning predict emergency-department triage priority using only information available at the moment a patient arrives?**

This project explores that question using **KTAS (Korean Triage and Acuity Scale)** data from adult emergency-department patients.

The goal isn't simply to build a model with a high accuracy score. The focus is on building a **realistic triage decision-support pipeline** by carefully handling missing data, class imbalance, categorical variables, and—most importantly—**data leakage**.

---

## 🚨 Why This Project Matters

In emergency medicine, not all classification errors have the same consequences.

Predicting a patient as **less urgent than they actually are** can result in under-triage, while predicting them as more urgent can result in over-triage and additional resource use.

That makes this a particularly interesting machine-learning problem:

* The target has **5 ordered urgency levels**
* The classes are **strongly imbalanced**
* Several variables contain substantial missingness
* Some variables are only available **after** the triage decision
* The dataset contains both numerical and categorical information
* Under-triage and over-triage have different practical implications

This project therefore evaluates the models using more than ordinary accuracy.

---

# 🎯 Project Objective

Predict:

```text
KTAS_expert
```

where:

| KTAS | Meaning      |
| ---: | ------------ |
| 🔴 1 | Most urgent  |
| 🟠 2 | Very urgent  |
| 🟡 3 | Urgent       |
| 🟢 4 | Less urgent  |
| 🟢 5 | Least urgent |

The target was determined by a panel of **three triage experts**, independently of the treating nurse's real-time triage decision.

The central modeling constraint is:

> **Only information available at the moment of triage should be used as a model feature.**

This prevents the model from achieving artificially high performance by using information that would not actually be available when the triage decision is made.

---

# 📊 Dataset

The dataset contains:

* **1,267 adult emergency-department patients**
* Data collected from **two Korean emergency departments**
* Collection period: **October 2016 – September 2017**
* **24 recorded variables**

The data includes information such as:

* Age
* Sex
* Arrival mode
* Injury status
* Mental status
* Pain
* Numeric pain score
* Blood pressure
* Heart rate
* Respiratory rate
* Body temperature
* Oxygen saturation
* Chief complaint
* Patient arrival/crowding information
* KTAS expert assessment

### Ground Truth

The prediction target is:

```text
KTAS_expert
```

rather than:

```text
KTAS_RN
```

This distinction is important because `KTAS_RN` represents the nurse's own triage decision, while `KTAS_expert` represents the expert-panel reference used as the target.

---

# 🔍 Exploratory Data Analysis

The notebook performs a detailed EDA before modeling.

### The analysis includes:

* Dataset structure and data types
* Categorical variable interpretation
* Missing-value analysis
* Target-class distribution
* Feature distributions across KTAS levels
* Correlation analysis
* Feature relationships with triage severity
* Missingness patterns across KTAS classes

One of the most interesting findings is the substantial missingness in some clinical variables.

For example:

```text
Saturation    ~55% missing
NRS_pain      ~44% missing
```

More importantly, missingness is **not uniformly distributed across KTAS classes**.

Instead of simply imputing these values and throwing away the missingness information, the project creates explicit missing-value indicators.

---

# ⚠️ Data Leakage Audit

One of the most important parts of this project is the **data-leakage audit**.

Several variables in the raw dataset are only known **after** the patient's ED visit or are directly related to the target.

Examples include:

```text
KTAS_RN
Diagnosis in ED
Disposition
Error_group
Length of stay_min
KTAS duration_min
mistriage
```

Using these variables during training would make the model appear more accurate than it could realistically be at triage time.

Therefore, these post-triage variables are excluded from the modeling feature set.

### The principle

> **If the information would not be available when the triage decision is made, the model should not see it.**

This makes the experiment substantially closer to a real-world clinical decision-support scenario.

---

# 🧠 Feature Engineering

The project uses several preprocessing and feature-engineering strategies.

### 1. Missingness indicators

For important clinical variables, additional binary features are created:

```text
Saturation_missing
NRS_pain_missing
SBP_missing
DBP_missing
HR_missing
RR_missing
BT_missing
```

This allows the models to learn whether **the absence of a measurement itself** contains useful information.

---

### 2. Chief complaint grouping

The raw dataset contains approximately **417 distinct chief-complaint values**.

With only ~1,000 training examples, using every raw category would create a high-cardinality feature with considerable overfitting risk.

The project therefore groups less frequent complaints into:

```text
other
```

while retaining frequently occurring complaints.

---

### 3. Categorical features

Categorical variables are explicitly represented as categorical data rather than treating their numerical codes as continuous measurements.

Examples include:

```text
Sex
Arrival mode
Injury
Mental
Pain
Group
Chief_complain_grp
```

---

# ⚖️ Class Imbalance

The KTAS target is heavily imbalanced.

In particular, the most critical category:

```text
KTAS 1
```

represents only a small fraction of the dataset.

This means that a model could achieve seemingly good overall accuracy while performing poorly on the classes that matter most.

For this reason, the project focuses heavily on:

* **Macro-F1**
* **Quadratic-weighted Cohen's Kappa**
* Confusion matrices
* Under-triage rate
* Over-triage rate
* Severe-class recall

Class weighting is also used during training.

The weighting strategy uses square-root inverse frequency weighting to give rare classes additional importance without excessively dominating the training process.

---

# 🤖 Models

Three gradient-boosting algorithms are compared under the same experimental setup.

## 1. XGBoost

[XGBoost](https://xgboost.readthedocs.io/) is used with native categorical support and hyperparameter optimization through randomized search.

The search evaluates parameters including:

* Number of estimators
* Tree depth
* Learning rate
* Subsampling
* Column subsampling
* Minimum child weight
* Regularization
* Gamma

Optimization objective:

```text
Macro-F1
```

using **5-fold stratified cross-validation**.

---

## 2. CatBoost

[CatBoost](https://catboost.ai/) is particularly interesting for this dataset because of its relatively small sample size and substantial categorical information.

Categorical features are passed directly to CatBoost rather than being manually one-hot encoded.

A randomized parameter search is performed over:

* Iterations
* Tree depth
* Learning rate
* L2 regularization
* Border count
* Bagging temperature

---

## 3. LightGBM

[LightGBM](https://lightgbm.readthedocs.io/) is used as a third gradient-boosting baseline.

The experiment tunes parameters including:

* Number of estimators
* Number of leaves
* Maximum depth
* Learning rate
* Feature subsampling
* Row subsampling
* Minimum child samples
* L1/L2 regularization

---

# 🧪 Experimental Setup

The dataset is divided using a stratified train/test split:

```text
80% → Training
20% → Testing
```

with:

```python
random_state = 42
```

The training data uses:

```text
5-fold Stratified Cross-Validation
```

and model selection is based on:

```text
Macro-F1
```

This ensures that rare KTAS classes have a meaningful influence on model selection.

---

# 🩺 Cost-Sensitive Triage

A standard classifier normally chooses the class with the highest predicted probability.

But triage isn't a normal classification problem.

Consider:

```text
True KTAS = 1
Predicted KTAS = 5
```

versus:

```text
True KTAS = 5
Predicted KTAS = 4
```

These two mistakes are not equivalent from a triage perspective.

Therefore, the project introduces a **cost-sensitive prediction rule**.

Under-triage is given a higher penalty:

```python
UNDER_COST = 1.5
```

The model then selects the prediction with the lowest expected cost rather than simply choosing the most probable class.

Conceptually:

```text
             Prediction
              1  2  3  4  5
True 1        0  1  2  3  4
True 2        1  0  1  2  3
True 3        1  1  0  1  2
...
```

with under-triage errors receiving a larger penalty.

### Important

The `1.5×` penalty is **not a universal clinical standard**.

It is a configurable modeling assumption demonstrating how the decision threshold can be adjusted depending on the operational cost assigned to under-triage.

---

# 📏 Evaluation Metrics

The project evaluates both raw model predictions and cost-sensitive predictions.

### Macro-F1

Macro-F1 gives every class equal importance:

```text
F1₁ + F1₂ + F1₃ + F1₄ + F1₅
--------------------------------
               5
```

This is especially useful because the KTAS classes are imbalanced.

---

### Quadratic-Weighted Cohen's Kappa

KTAS levels are ordered.

A prediction of:

```text
1 → 2
```

is not as far away from the truth as:

```text
1 → 5
```

Quadratic-weighted Cohen's Kappa accounts for this ordinal structure.

---

### Exact Accuracy

The percentage of predictions where:

```text
Predicted KTAS == True KTAS
```

---

### Under-Triage Rate

Predicted urgency is lower than the true urgency.

This is treated as the more costly error direction in the experiment.

---

### Over-Triage Rate

Predicted urgency is higher than the true urgency.

---

### Severe-Class Recall

The project also monitors performance on the most urgent classes:

```text
KTAS 1–2
```

because performance on these classes can be obscured by aggregate metrics.

---

# 📈 Model Comparison

The notebook compares:

```text
XGBoost
CatBoost
LightGBM
```

under both:

```text
Raw prediction
```

and:

```text
Cost-sensitive prediction
```

The comparison includes:

* Macro-F1
* Quadratic-weighted Cohen's Kappa
* Exact accuracy
* Under-triage rate
* Over-triage rate
* Severe-class recall
* Confusion matrices

This produces a more complete picture than reporting accuracy alone.

---

# 💡 Key Findings

The experiments indicate that **CatBoost achieved the strongest performance among the three tested boosting approaches on this particular dataset**, including under the cost-sensitive prediction setup.

The notebook also identifies several influential signals, including:

* Respiratory rate
* Mental status
* Chief complaint
* Pain information
* Whether vital signs were recorded

An important observation is that **missing clinical measurements themselves can contain predictive information**.

However, these findings should be interpreted carefully because the dataset is relatively small.

---

# ⚠️ Limitations

This project is a research/learning experiment, **not a clinically validated triage system**.

Important limitations include:

### Small dataset

Only:

```text
1,267 patients
```

are available for a 5-class classification problem.

More data would provide more reliable estimates of generalization performance.

### Class imbalance

The most critical KTAS categories contain relatively few observations.

Performance on rare classes can therefore be unstable.

### Missing clinical information

Some variables have substantial missingness.

Although missingness indicators are intentionally retained, the underlying missing-data mechanism may reflect clinical workflow rather than purely patient characteristics.

### Limited generalization

The data comes from two Korean emergency departments during a specific time period.

Performance on another hospital, healthcare system, country, or time period may differ.

### Cost function is hypothetical

The:

```text
1.5× under-triage penalty
```

is a modeling choice used for experimentation.

A real deployment would require clinically validated cost assumptions.

### No prospective validation

A model intended for clinical decision support would require much more extensive validation, monitoring, safety analysis, and clinical evaluation before real-world use.

---

# 🗂️ Project Structure

A simple repository structure could look like:

```text
KTAS-Triage-Prediction/
│
├── KTAS_EDA_XGBoost.ipynb
├── data.csv
├── README.md
├── requirements.txt
└── .gitignore
```

> **Note:** If the dataset has redistribution restrictions, do not commit the raw `data.csv` to GitHub. Instead, provide instructions describing how users can obtain the dataset.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/KTAS-Triage-Prediction.git
cd KTAS-Triage-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost catboost lightgbm jupyter
```

---

# ▶️ Running the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
KTAS_EDA_XGBoost.ipynb
```

Make sure the dataset is available at:

```text
./data.csv
```

The notebook will then:

```text
Load data
   ↓
Clean corrupted/missing values
   ↓
Explore the dataset
   ↓
Audit data leakage
   ↓
Engineer features
   ↓
Split train/test data
   ↓
Apply class weighting
   ↓
Tune XGBoost
   ↓
Tune CatBoost
   ↓
Tune LightGBM
   ↓
Evaluate predictions
   ↓
Apply cost-sensitive decision rule
   ↓
Compare models
```

---

# 🛠️ Tech Stack

| Technology      | Purpose                             |
| --------------- | ----------------------------------- |
| 🐍 Python       | Core programming language           |
| 🐼 Pandas       | Data manipulation                   |
| 🔢 NumPy        | Numerical computation               |
| 📊 Matplotlib   | Visualization                       |
| 🎨 Seaborn      | Statistical visualization           |
| 📐 SciPy        | Statistical analysis                |
| 🤖 Scikit-learn | Splitting, tuning & evaluation      |
| ⚡ XGBoost       | Gradient boosting                   |
| 🐱 CatBoost     | Categorical-aware gradient boosting |
| 💡 LightGBM     | Gradient boosting                   |
| 📓 Jupyter      | Interactive analysis                |

---

# 📚 What This Project Demonstrates

This project goes beyond simply fitting a machine-learning model.

It demonstrates a complete applied ML workflow:

```text
                    ┌─────────────────┐
                    │   Raw Dataset   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Data Cleaning   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │      EDA        │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Leakage Audit   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Feature         │
                    │ Engineering     │
                    └────────┬────────┘
                             ↓
                 ┌───────────┼───────────┐
                 ↓           ↓           ↓
             XGBoost      CatBoost    LightGBM
                 │           │           │
                 └───────────┼───────────┘
                             ↓
                    ┌─────────────────┐
                    │ Cost-Sensitive  │
                    │ Prediction      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Model Evaluation│
                    └─────────────────┘
```

---

# 🔬 Future Improvements

Several directions could make the project more robust:

* [ ] External validation on an independent hospital dataset
* [ ] Larger multi-center dataset
* [ ] Repeated cross-validation
* [ ] Calibration analysis
* [ ] Probability calibration
* [ ] Ordinal classification methods
* [ ] Explainable AI with SHAP
* [ ] Robustness testing across hospital sites
* [ ] Temporal validation
* [ ] More rigorous missing-data analysis
* [ ] Clinically validated cost matrices
* [ ] Decision-curve analysis
* [ ] Prospective evaluation

---

# ⚠️ Disclaimer

This repository is intended for **educational, research, and machine-learning experimentation purposes**.

It is **not a medical device**, and its predictions should not be used to make clinical decisions or determine patient treatment.

Any real-world clinical deployment would require independent validation, clinical oversight, regulatory review where applicable, and appropriate safety monitoring.

---

# 👨‍💻 Author

**Moe**

If you found this project interesting, feel free to ⭐ the repository or explore the notebook.

---

## ⭐ If You Like This Project

Consider giving the repository a star!

```text
⭐ Star → 🔎 Explore → 🧠 Learn → 🚀 Build
```

---

### Keywords

`Machine Learning` · `Healthcare AI` · `Emergency Medicine` · `KTAS` · `Triage` · `XGBoost` · `CatBoost` · `LightGBM` · `Classification` · `EDA` · `Data Science` · `Python` · `Clinical Decision Support`
