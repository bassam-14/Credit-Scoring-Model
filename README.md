# Credit Scoring Model  

This project builds a **machine learning pipeline** to predict a customer’s **credit score classification** (`Poor`, `Standard`, `Good`) using demographic, financial, and behavioral data.  

The dataset is from Kaggle: [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification).  

---

## 📌 Project Overview  
Banks and financial institutions rely on credit scoring to evaluate the risk of lending money to customers.  
This project aims to:  
- Clean and preprocess raw credit data  
- Engineer additional features to capture financial behavior  
- Train and evaluate multiple ML models  
- Compare model performance on baseline vs engineered datasets  

---

## ⚙️ Data Preprocessing  
1. **Dropped irrelevant columns** (`ID`, `Customer_ID`, `Name`, `SSN`, `Monthly_Inhand_Salary`)  
2. **Handled Age column** → fixed invalid values, capped outliers  
3. **Categorical encoding**  
   - `Credit_Score` → labels (`Poor=0`, `Standard=1`, `Good=2`)  
   - `Occupation`, `Credit_Mix`, `Payment_of_Min_Amount` → one-hot encoding  
4. **Missing values**  
   - Added “missing” indicator columns  
   - Imputed numeric columns with median values  
5. **Complex features**  
   - `Type_of_Loan` → split into multiple binary loan features  
   - `Credit_History_Age` → converted to total months  
   - `Payment_Behaviour` → split into `Spending` (Low/High) & `Payment_Value` (Small/Medium/Large)  
6. **Date handling**  
   - Encoded `Month` cyclically using sine & cosine  

---

## 🛠️ Feature Engineering  
Created an engineered dataset (`df_eng`) with new features:  

- **Debt_to_Income** = Outstanding Debt / Annual Income  
- **Loan_per_Account** = Num_of_Loan / Num_Bank_Accounts  
- **Utilization_Interest** = Credit Utilization Ratio × Interest Rate  
- **EMI_to_Income** = Total EMI per month / Annual Income  
- **Balance_to_Income** = Monthly Balance / Annual Income  
- **Delinquency_Ratio** = Num_of_Delayed_Payment / Num_of_Loan  

---

## 🤖 Models Used  
The following models were trained **on both baseline (`df`) and engineered (`df_eng`) datasets**:  

- **Logistic Regression**  
- **Random Forest Classifier**  
- **XGBoost Classifier**  

---

## 📊 Results  

### Baseline Dataset (`df`)  
*(earlier baseline results — weaker performance, especially Logistic Regression)*  
- Logistic Regression: ~61% accuracy  
- Random Forest: ~80% accuracy  
- XGBoost: ~76% accuracy  

### Engineered Dataset (`df_eng`) – Final Results  
| Model               | Accuracy | Precision | Recall | F1-score | Notes |
|----------------------|----------|-----------|--------|----------|-------|
| Logistic Regression  | **0.6181** | 0.60 | 0.58 | 0.58 | Struggles, but improves with engineered ratios |
| Random Forest        | **0.8024** | 0.79 | 0.79 | 0.79 | Strongest performer, captures complex interactions |
| XGBoost              | **0.7678** | 0.75 | 0.75 | 0.75 | Competitive, benefits from engineered features |

✅ **Engineered features improved all models**, especially Random Forest and XGBoost, while Logistic Regression still lagged behind but became more stable.  

---


## 🚀 How to Run  

1. Clone this repository:
```bash
git clone https://github.com/your-username/Credit-Scoring-Model.git
cd Credit-Scoring-Model
```
2. Install Dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebook:
```bash
jupyter notebook main.ipynb
```
Or run the script (if using main.py):
```bash
python main.py
```
