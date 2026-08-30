# CreditWise Loan System

An intelligent, Machine Learning–powered loan approval system built for **SecureTrust Bank** to automate and improve the accuracy of personal and home loan decisions.

---

##  Problem Statement

SecureTrust Bank is a mid-sized financial company offering personal and home loans to customers across urban and rural regions of India. Every day, hundreds of customers apply for loans through online and branch channels.

Historically, loan approvals have relied on a **manual verification process**, where loan officers evaluate applications by checking income proofs, employment details, credit history, and other supporting documents. This process is:

- Time-consuming
- Inconsistent across officers
- Prone to human bias

As a result, the bank faces two major challenges:

1. **Good customers sometimes get rejected** → loss of business
2. **High-risk customers sometimes get approved** → financial losses (defaults)

### Goal

Design and build an **intelligent loan approval system** using Machine Learning that automatically analyzes applicant details and predicts whether a loan should be **Approved** or **Rejected**, before final human verification — providing accurate, fast, and unbiased decisions.

---

##  Dataset Description

Each row in the dataset represents a single **loan applicant**, described by personal, financial, and credit-related attributes.

| Column | Description |
|---|---|
| `Applicant_ID` | Unique applicant ID |
| `Applicant_Income` | Monthly income of applicant |
| `Coapplicant_Income` | Monthly income of co-applicant |
| `Employment_Status` | Salaried / Self-Employed / Business |
| `Age` | Applicant age |
| `Marital_Status` | Married / Single |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Number of already running loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Savings balance |
| `Collateral_Value` | Value of collateral provided |
| `Loan_Amount` | Loan amount requested |
| `Loan_Term` | Loan duration (months) |
| `Loan_Purpose` | Home / Education / Personal / Business |
| `Property_Area` | Urban / Semi-Urban / Rural |
| `Education_Level` | Graduate / Postgraduate / Undergraduate |
| `Gender` | Male / Female |
| `Employer_Category` | Govt / Private / Self |
| `Loan_Approved` (Target) | `1` = Approved, `0` = Rejected |

---

##  Project Objective

Build a supervised classification model that learns hidden patterns from historical loan application records and predicts `Loan_Approved` (1/0) for new applicants — supporting SecureTrust Bank's loan officers with a fast, consistent, and data-driven first-pass decision.

---

##  Tech Stack

- **Language:** Python
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn
  - Logistic Regression
  - Naive Bayes (GaussianNB)
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
- **Model Selection & Tuning:** `GridSearchCV`, `Pipeline`, `StandardScaler`
- **Evaluation:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report

---

##  Project Workflow

1. **Data Understanding & Cleaning**
   - Handle missing values
   - Encode categorical features (`Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Education_Level`, `Gender`, `Employer_Category`)
   - Feature scaling where required (Logistic Regression, Naive Bayes, KNN)

2. **Exploratory Data Analysis (EDA)**
   - Distribution of income, credit score, DTI ratio
   - Class balance of `Loan_Approved`
   - Correlation between features and approval outcome

3. **Model Building**
   - Train baseline models: Logistic Regression, Naive Bayes, KNN, Decision Tree
   - Apply **Decision Tree pruning**:
     - **Pre-pruning** — grid search over `max_depth` and `min_samples_split`
     - **Post-pruning** — cost-complexity pruning using `ccp_alpha`

4. **Hyperparameter Tuning**
   - `GridSearchCV` (5-fold cross-validation) with `f1_weighted` scoring across all models

5. **Model Evaluation**
   - Compare models on Accuracy, Precision, Recall, and F1 Score
   - Review Confusion Matrix and Classification Report for each model
   - Select the best-performing model based on F1 Score

6. **Best Model Selection**
   - Final model chosen and stored for deployment / future predictions

## 📈 Results

### Baseline models (before tuning)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.72 | 0.58 | 0.25 | 0.34 |
| Naive Bayes | 0.74 | 0.57 | 0.54 | 0.55 |
| KNN | 0.64 | 0.33 | 0.20 | 0.25 |
| Decision Tree | 0.88 | 0.83 | 0.79 | 0.81 |
| SVC | 0.70 | 0.00 | 0.00 | 0.00 |
| Random Forest | 0.90 | 0.83 | 0.87 | 0.85 |

*SVC's 0.00 precision/recall on the baseline run means it predicted every applicant as "Rejected" — a sign the untuned model needed a different kernel and/or feature scaling, both addressed during hyperparameter tuning below.*

### After hyperparameter tuning (GridSearchCV, 5-fold CV, `f1_weighted` scoring)

| Model | Accuracy | Precision | Recall | F1 Score | Best Parameters |
|---|---|---|---|---|---|
| Logistic Regression | 0.87 | 0.79 | 0.79 | 0.79 | `C=10, penalty='l2'` |
| Naive Bayes | 0.86 | 0.79 | 0.75 | 0.77 | `var_smoothing=0.1` |
| KNN | 0.77 | 0.60 | 0.51 | 0.55 | `n_neighbors=3` |
| SVM | 0.85 | 0.78 | 0.80 | 0.79 | `kernel='linear'` |
| **Decision Tree** | **0.91** | **0.79** | **0.95** | **0.87** | `ccp_alpha=0.01, max_depth=5, min_samples_split=2` |
| Random Forest | 0.91 | 0.83 | 0.89 | 0.86 | `max_depth=9, min_samples_split=2, n_estimators=300` |

Best model: Decision Tree (tuned) — F1 Score = 0.87

Tuning made a substantial difference across the board — every model improved after `GridSearchCV`, and the tuned Decision Tree edged out Random Forest on F1 (0.87 vs 0.86) while achieving the highest recall (0.95) of any model, meaning it misses the fewest genuinely creditworthy applicants — an important property for a bank trying to avoid losing good customers to false rejections.

<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/b14286d5-5c77-4aec-89dd-2c2b364b456c" />


## Live Demo (Streamlit App)

This project includes a Streamlit web app (app.py) that loads the trained model and lets you get an instant loan approval prediction from a simple form.

Run it locally
```
bash
git clone https://github.com/akshay041193/loan-approval-system.git
cd loan-approval-system

pip install -r requirements.txt

streamlit run app.py
```

##  Project Structure

```
loan-approval-system/
│
├── Loan_Approval_System.ipynb   # Main notebook: EDA, preprocessing, model training,
│                                 # pruning (pre & post), hyperparameter tuning, evaluation
├── loan_approval_data.csv       # Dataset used for training and evaluation
├── App.py                       # Streamlit app for interactive predictions
├── requirements.txt             # Python dependencies for local/cloud deployment
├── loan_model.pkl      # Saved trained model (generated by the notebook)
├── feature_orderpkl
├── .gitignore
└── README.md
```

---

##  How to Run the Notebook

```bash
# 1. Clone the repository
git clone https://github.com/akshay041193/loan-approval-system.git
cd loan-approval-system

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook Loan_Approval_System.ipynb
```

---

## ✅ Key Outcomes

- Automated, consistent loan approval predictions replacing manual, bias-prone review
- Reduced risk of approving high-risk applicants and rejecting creditworthy ones
- Best model selected via cross-validated hyperparameter tuning, evaluated with multiple metrics (not accuracy alone) to account for class imbalance
