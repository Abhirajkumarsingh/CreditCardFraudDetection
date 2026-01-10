🚀 1. Project Overview

Credit card fraud is extremely rare (only 0.173% of transactions in the dataset), making detection difficult using traditional methods.
This project builds a machine learning model to classify transactions as:
0 → Normal
1 → Fraudulent

The final solution includes a trained ML model + an interactive web interface for real-time predictions.

🧠 2. Dataset Information

Source: Kaggle – Credit Card Fraud Detection Dataset
Rows: 284,807
Features: 31

🧩 Feature Types:

Time

V1–V28 → PCA-transformed features

Amount

Class (Target):

0 → Normal

1 → Fraudulent

❗ Challenge:

The dataset is highly imbalanced:

Class	Count	Percentage
Normal	284,315	99.83%
Fraud	492	0.17%
⚙️ 3. Machine Learning Pipeline
✔️ 1. Data Preprocessing

Remove noise / missing values

Apply StandardScaler to normalize numerical columns

✔️ 2. Train–Test Split

80% training

20% testing

✔️ 3. Handling Imbalance with SMOTE

Oversamples the minority (fraud) class

Applied only on training set to avoid data leakage

✔️ 4. Models Trained

Logistic Regression

Naive Bayes

Random Forest

XGBoost (Final Model)

✔️ 5. Evaluation Metrics

ROC–AUC

Precision / Recall

F1–Score

F-Beta scores (β ∈ {0.5, 1, 2, 3}) → focus on recall for fraud

🏆 Final Model: Gradient Boosting (XGBoost)

Highest ROC-AUC ≈ 0.979

Best recall for fraud cases

Most stable across F-Beta tests

💻 4. Streamlit Web App

The project includes a ready-to-run Streamlit UI:

✔️ Single Transaction Prediction

Enter comma-separated feature values (Time, V1, V2, ..., Amount) → get fraud/no-fraud result.

✔️ Batch CSV Prediction

Upload a CSV containing all features

Get prediction for every row

Download results as predictions.csv

✔️ Uses the pre-trained model:

best_model_gradient_boosting.pkl

📁 5. Project Structure
CreditCardFraudDetection/
│── app.py                     # Streamlit frontend
│── best_model_gradient_boosting.pkl   # Trained ML model
│── requirements.txt           # All dependencies
│── dataset/
│     └── creditcard.csv       # Dataset (if included)
│── figures/                   # Graphs, charts
│── notebook/
│     └── training.ipynb       # Model training notebook
│── test_samples/
│     └── sample_input.csv     # For batch testing
│── README.md                  # Project documentation

▶️ 6. How to Run the Project Locally
Step 1 — Create Conda environment
conda create -n frauddetect python=3.10 -y
conda activate frauddetect

Step 2 — Install required packages
pip install -r requirements.txt

Step 3 — Run Streamlit App
streamlit run app.py


Streamlit will open the app in your browser automatically.

📷 7. App Features Example
🔍 Single Prediction Output

Shows whether transaction is Fraudulent or Normal

📂 CSV Batch Prediction

Returns a dataframe with predictions

Option to download results as .csv

📊 8. Model Performance Summary
Model	ROC-AUC	Precision	Recall	F1
Logistic Regression	~0.971	Medium	Good	Medium
Naive Bayes	~0.965	Medium	Low	Low
Random Forest	~0.968	Good	Medium	Good
Gradient Boosting (XGBoost)	~0.979	High	High	Best

Reason for final choice:
👉 Highest recall for fraud cases (most important business metric).
👉 Stable performance on imbalanced data.

🚀 9. Future Enhancements

Deploy model as a cloud API

Implement real-time fraud monitoring

Add Deep Learning (Autoencoders, LSTMs)

Build dashboard for visualization

Optimize using cost-sensitive learning

🙌 10. Contributors

Abhiraj Kumar Singh (Developer)
