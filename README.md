Credit Risk Analysis with Logistic Regression & LIME
A machine learning project for credit risk classification using the German Credit Dataset. The model predicts whether a bank customer is a good or bad credit risk, with LIME (Local Interpretable Model-agnostic Explanations) used to explain individual predictions locally.

This project is a companion to the SHAP-based analysis, using a different interpretability method on the same dataset.


📋 Project Overview
This project applies Logistic Regression with L1-based feature selection and explains each prediction using LIME TabularExplainer. Unlike SHAP, LIME generates local explanations — it explains why the model made a specific decision for a specific customer, one instance at a time.

📁 Repository Structure
├── lime_analysis.py                  # Main script: training, feature selection, LIME analysis
├── model_summary.txt                 # Model performance report (accuracy, F1, confusion matrix)
├── lime_explanations.xlsx            # Detailed LIME weights for each test instance
├── lime_feature_importance.xlsx      # Aggregated global feature importance from LIME
├── lime_report.docx                  # Auto-generated Word report with top features table
├── lime_instance_5.zip               # Interactive HTML LIME explanations (instances 1–5)
└── real_data_1.xlsx                  # Preprocessed dataset used for training

🤖 Model Details
ParameterValueAlgorithmLogistic Regression (L2 regularization)Feature SelectionL1 (Lasso) — top 15 featuresSolverLBFGSTest Split20%Accuracy71%
Classification Report
ClassPrecisionRecallF1-ScoreGood Credit (1)0.750.890.81Bad Credit (2)0.520.270.36Weighted Avg0.680.710.68
Confusion Matrix
Predicted GoodPredicted BadActual Good12615Actual Bad4316

🔍 Selected Features (15)
Features chosen via L1 regularization:
#Feature1Duration2Age3Credit_Amount4Installment_Rate5Credit_History_good_history6Credit_History_problem_history7Number_Credits8Housing_rent_or_free9Housing_own10Employment_short_or_unemployed11Employment_long12Savings_high13Savings_low_or_unknown14Other_Installment_Plans_no_other_plans15Other_Installment_Plans_has_other_plans

💡 How LIME Works
LIME explains individual predictions by:

Taking a single test instance (one customer)
Creating perturbed versions of that instance
Fitting a simple linear model locally around it
Using that local model's coefficients as the explanation

This gives per-instance feature weights showing which features pushed the prediction toward good or bad credit.

🚀 How to Run
1. Install dependencies
bashpip install pandas numpy scikit-learn lime matplotlib openpyxl python-docx
2. Place your data file on the Desktop
The script searches for a file named real data 1 (any extension: .csv, .xlsx, .xls) on the Desktop.
3. Run the analysis
bashpython lime_analysis.py
4. Results
All outputs are saved to ~/Desktop/lime.real/:
FileDescriptionlime_explanations.xlsxLIME weights per instancelime_feature_importance.xlsxAggregated global importancelime_report.docxSummary Word reportlime_instance_1.html … lime_instance_5.htmlInteractive HTML explanationsmodel_summary.txtAccuracy, F1-score, confusion matrix

📦 Dataset
German Credit Dataset — 1,000 bank customers described by financial and personal attributes. Target: 1 = Good credit risk, 2 = Bad credit risk.🛠 Tech Stack

Python 3.10+
scikit-learn — Logistic Regression, feature selection, metrics
lime — Local interpretable explanations (LimeTabularExplainer)
pandas / numpy — Data processing
python-docx — Word report generation
openpyxl — Excel output


👤 Author
Alen Baghdasaryan
GitHub: @Alen-Baghdasaryan
