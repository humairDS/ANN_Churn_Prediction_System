## ANN Bank Customer Churn Prediction System

A churn classification system built with an Artificial Neural Network (ANN). It predicts whether a bank customer is likely to leave (churn) based on their profile and account details, and is served through an interactive Streamlit web app.

🔗 Live demo: annchurnpredictionsystem.streamlit.app

# Overview

Customer churn is costly: keeping an existing customer is usually cheaper than acquiring a new one. This project trains a neural network on the Churn_Modelling.csv dataset to estimate each customer's probability of churning, so a business can spot at-risk customers early.

The workflow covers data preprocessing, model training, saving the trained artifacts, and deploying a prediction app.

# ✨ Features
End-to-end ML pipeline: preprocessing → training → evaluation → deployment
ANN binary classifier built with TensorFlow / Keras
Saved encoders and scaler so the app applies identical preprocessing at prediction time
Interactive Streamlit UI: enter customer details and get a churn probability
Training logs saved in log/ for TensorBoard visualisation
🛠️ Tech Stack
Area	Tools
Language	Python
Deep learning	TensorFlow / Keras
Data processing	Pandas, NumPy, scikit-learn
Web app	Streamlit
Experimentation	Jupyter Notebook, TensorBoard


# 📁 Project Structure
ANN_Churn_Prediction_System/
├── Churn_Modelling.csv         # Dataset
├── experiment.ipynb            # Data preprocessing, model building and training
├── prediction.ipynb            # Loads saved model and runs sample predictions
├── app.py                      # Streamlit web app
├── model.h5                    # Trained ANN model
├── scaler.pkl                  # Fitted StandardScaler
├── label_encoder_gender.pkl    # Label encoder for Gender
├── onehot_encoder_geo.pkl      # One-hot encoder for Geography
├── log/                        # TensorBoard training logs
├── requirements.txt            # Python dependencies
└── .gitignore
📊 Dataset

Churn_Modelling.csv contains bank customer records. The target column is Exited (1 = customer left, 0 = customer stayed).

Typical input features include CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember and EstimatedSalary.

# 🧠 Methodology
Cleaning: drop identifier columns that carry no predictive value (e.g. RowNumber, CustomerId, Surname).
Encoding: label-encode Gender and one-hot encode Geography.
Scaling: standardise features with StandardScaler.
Split: divide the data into training and test sets.
Model: a feed-forward ANN with fully connected hidden layers and a sigmoid output for binary classification, trained with Adam and binary cross-entropy.
Monitoring: training tracked with TensorBoard, with early stopping to limit overfitting.
Export: save the model (model.h5) and the preprocessing objects (.pkl) for deployment.

# 🚀 Getting Started
Prerequisites
Python 3.9+
pip
Installation
bash
git clone https://github.com/humairDS/ANN_Churn_Prediction_System.git
cd ANN_Churn_Prediction_System

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
Run the app
bash
streamlit run app.py

Then open the local URL shown in your terminal (usually http://localhost:8501).

Explore the notebooks
bash
jupyter notebook
experiment.ipynb — how the model was built and trained
prediction.ipynb — how to load the model and predict for a new customer
View training logs (optional)
bash
tensorboard --logdir log

# 🖥️ Usage
Open the app (locally or via the live demo).
Enter the customer's details: geography, gender, age, balance, credit score, tenure, number of products, salary, and so on.
The app preprocesses the inputs with the saved encoders and scaler, then displays the churn probability and whether the customer is likely to leave.

# 📈 Results
Metric	Value
Test accuracy	[add yours]
Validation accuracy	[add yours]

(Fill in the numbers from experiment.ipynb.)

🔮 Future Improvements
Handle class imbalance (SMOTE or class weights)
Compare against Random Forest and XGBoost baselines
Add precision, recall, F1 and ROC-AUC reporting
Hyperparameter tuning (layers, units, learning rate)
Add model explainability (e.g. SHAP)
🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, commit your changes and open a pull request.

👤 Author

Humair — @humairDS
