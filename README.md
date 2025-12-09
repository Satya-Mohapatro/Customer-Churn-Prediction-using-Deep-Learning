#  Customer Churn Prediction using Deep Learning (ANN + Embeddings)

An end-to-end deep learning project that predicts telecom customer churn using the **Telco Customer Churn Dataset**.  
This project uses **learned categorical embeddings + dense neural layers**, offering a powerful alternative to traditional one-hot encoded ML models.

---

##  Project Overview

Customer churn is a critical business metric that heavily affects revenue.  
This project builds a deep learning model that predicts whether a customer will leave the telecom service provider based on:

- Demographics  
- Account-level information  
- Internet/phone service subscriptions  
- Billing & payment patterns  

###  Why this project matters
- Reduces customer acquisition cost  
- Helps design targeted retention campaigns  
- Enables proactive monitoring of high-risk users  

###  Project Goals
- Clean and preprocess customer churn data  
- Build an ANN with **categorical embeddings**  
- Train and evaluate using metrics like **AUC**, **Accuracy**, and **Confusion Matrix**  
- Provide deployment-ready inference scripts  

---


##  Model Architecture

### ✔ Embedding layers for categorical features  
Instead of one-hot encoding, each categorical column is assigned its own embedding matrix.

### ✔ Dense neural network for classification  
Final layers consist of:

- Dense(128) → ReLU → BatchNorm → Dropout  
- Dense(64) → ReLU → BatchNorm → Dropout  
- Output: Sigmoid for churn probability  

---

##  Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-Learn** (preprocessing + metrics)
- **FastAPI** (optional deployment)

---

##  Dataset

- **Source:** Telco Customer Churn (Kaggle)  
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- **Target variable:** `Churn` (Yes/No)

Key feature groups include:
- Customer demographic info  
- Phone & internet service subscriptions  
- Contract and billing preferences  
- Monthly & total charges  

---


##  Evaluation Metrics

The model outputs:

- **ROC-AUC score**
- **Accuracy**
- **Precision & Recall**
- **Confusion Matrix**
- **ROC Curve Visualization**

Example results:
```
ROC-AUC: 0.85+
Accuracy: ~80–82%
```

---

##  Key Features

### ✔ Complete preprocessing pipeline  
- Handle missing values  
- Clean TotalCharges  
- Scale numeric features  
- Encode categorical values via LabelEncoders  
- Build TensorFlow-ready input dictionaries  

### ✔ Deep learning with embeddings  
- Better representation than one-hot encoding  
- More compact and expressive  

### ✔ Visualizations  
- Class distribution  
- Loss curves  
- AUC curves  
- Confusion matrix  

### ✔ Deployment-ready  
- Exports SavedModel  
- Includes `serve.py` for prediction  
- Includes FastAPI endpoint for real-time inference  

---

##  Future Improvements

- Add SHAP explainability  
- Hyperparameter tuning with Optuna  
- Try XGBoost/LightGBM for comparison  
- Model ensembling  
- Convert to ONNX / TensorFlow Lite  
- Cloud deployment (AWS Lambda / GCP / Azure)

---

##  Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow>=2.10
joblib
fastapi
uvicorn
shap
xgboost
tf2onnx
```

---

##  Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you’d like to modify.

---


