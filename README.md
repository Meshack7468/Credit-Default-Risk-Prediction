# Credit Default Risk Prediction

## OVERVIEW

The project uses machine learning to predict loan default risk, supporting faster, more consistent, and data-driven credit approval decisions.
It combines data preparation, exploratory data analysis, model development, and model evaluation to identify borrowers who are likely to default based on financial, employment, and loan-related factors.
The goal is to help lending institutions reduce financial losses, improve risk assessment, and support responsible lending practices.

## BUSINESS PROBLEM

Credit providers handle loan applications daily. Still, it is not always straightforward to tell in advance which applicants are likely to repay successfully and which ones may default. Poor risk assessment can lead to financial losses when high-risk borrowers are approved.

Manual assessments can be slow and inconsistent, calling for a more reliable and standardized decision-making process. 

This project uses machine learning to predict default risk early, improve approval consistency, and support faster, more accurate credit decisions

## DATA UNDERSTANDING

The dataset used is the Loan Default Prediction Dataset from Kaggle, containing over 255,000 loan records with borrower details, financial indicators, and loan characteristics.

The target variable is:

Default – Whether an applicant defaulted or repaid (1 = Defaulted, 0 = Repaid) making this a binary classification problem

Key predictive features include:

* Age
* Income
* Credit Score
* Months Employed
* Loan Amount
* Interest Rate
* Loan Term
* Employment Type
* Education
* Debt-to-Income Ratio (DTI)
* Loan Purpose
* Number of Credit Lines

##  METHODS

The project begins with data cleaning, duplicate checks, missing value checks, and preprocessing to prepare the dataset for analysis.
EDA is then performed to identify borrower behavior patterns and understand relationships across loan and borrower features.
Correlation analysis showed that multicollinearity was not a major concern.
SMOTE was applied to handle class imbalance.
Three models were trained and compared: Logistic Regression as the baseline, Random Forest, and XGBoost
Prediction & model evaluation using Precision, Recall, F1-score, and ROC-AUC metrics.
Finally, the Random Forest model was deployed within a Streamlit application to provide an interactive user interface, with Render used for cloud deployment.

## KEY VALUE

The project adds value by:

Identifying high-risk applicants early
Reducing potential loan default losses
Supporting faster and more efficient loan approval decisions
Improving consistency in credit risk assessments
Providing interpretable risk insights for better decision-making

## CONCLUSION & RECOMMENDATIONS

Default risk is strongly influenced by income, credit score, employment duration, age, interest rate, and loan amount. Higher interest rates, higher loan amounts, lower credit scores, and lower income were connected to higher default risk.
Applicants with longer employment histories and higher ages showed a lower likelihood of default.
Random Forest delivered the most balanced performance across Precision, Recall, and F1-score, making it the best model for practical credit risk assessment.
Credit providers should use the model to detect high-risk applicants for further review while speeding up approvals for lower-risk individuals.
Continuous monitoring and regular model updates are recommended to maintain accuracy.
