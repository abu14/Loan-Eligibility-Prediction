# Import Libraries
import pandas as pd
import numpy as np
import sklearn.linear_model as logistic_regression
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold,train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, mean_squared_error
from sklearn.preprocessing import  MinMaxScaler,OneHotEncoder,LabelEncoder,RobustScaler,StandardScaler
import pickle
from sklearn.impute import SimpleImputer

# Setup Envivronment
import os
os.chdir(r'C:\Users\Dell\Loan-Eligibility-Prediction-Project')
#Load Data
input_data = pd.read_csv('data/loan-test.csv')

# Imort modules from functions
from functions import *

# Prepare data
df = prepare_data(input_data)

# Impute missing variables
df_imputed = data_imput(df)

# Dummy Variables
df_impu_dummy = var_dummy(df_imputed)

# Label Encoding
colmns = ['Gender','Married','Education','Self_Employed','Credit_History']
df_impu_dummy_encod = label_encod(df_impu_dummy,colmns)

# scale dataframe
colms = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
df_impu_dummy_encod_scaled = data_scaler(df_impu_dummy_encod,colms)


# Load our saved model
with open('model/best_model.pkl','rb') as f:
    loaded_model = pickle.load(f)


# copy file for saving 
df_predicted = df_impu_dummy_encod_scaler.copy()

# Get predictions
prediction = loaded_model.predict(df_impu_dummy_encod_scaler)
df_predicted['Predictions'] = pd.DataFrame(prediction)

# probability of predictions
probability = loaded_model.predict_proba(df_impu_dummy_encod_scaler)
df_predicted[['Probability_0','Probability_1']] = pd.DataFrame(probability)

#Save the model
df_predicted.to_csv('model/loan_predictions.csv')



