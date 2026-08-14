# CreditWise Loan System

An intelligent, Machine Learning–powered loan approval system built for **SecureTrust Bank** to automate and improve the accuracy of personal and home loan decisions.

---

## 📌 Problem Statement

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

## 📊 Dataset Description

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

## 🎯 Project Objective

Build a supervised classification model that learns hidden patterns from historical loan application records and predicts `Loan_Approved` (1/0) for new applicants — supporting SecureTrust Bank's loan officers with a fast, consistent, and data-driven first-pass decision.

---

## 🛠️ Tech Stack

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

## 🔄 Project Workflow

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


---
## 📂 Project Structure

```
loan-approval-system/
│
├── Loan_Approval_System.ipynb   # Main notebook: EDA, preprocessing, model training,
│                                 # pruning (pre & post), hyperparameter tuning, evaluation
├── loan_approval_data.csv       # Dataset used for training and evaluation
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

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

---

## 🔮 Future Improvements

- Add ensemble models (Random Forest, XGBoost, Gradient Boosting)

---

## 👤 Author

Machine Learning Engineer — CreditWise Loan System, SecureTrust Bank
