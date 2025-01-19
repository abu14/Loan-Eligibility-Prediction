import pandas as pd
import numpy as np
import sklearn.linear_model as logistic_regression
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold,train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, mean_squared_error
from sklearn.preprocessing import  MinMaxScaler,OneHotEncoder,LabelEncoder,RobustScaler,StandardScaler
import pickle
from sklearn.impute import SimpleImputer


# with open('model\best_model.pkl','rb') as f:     
#     model = pickle.load(f)


def load_and_split_data(input_data):
    # Read the data
    data = pd.read_csv(input_data)
    # Drop the columns that are not needed
    X = data.drop(['Loan_Status','Loan_ID'], axis=1)
    y = data['Loan_Status']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def prepare_data(data):
    """Loads and prepares the data by dropping unnecessary columns.
    Args:
        input_data: The pandas DataFrame containing the raw data.
    Returns:
        A pandas DataFrame with the 'Loan_ID' column dropped.
    """
    data = data.drop(['Loan_ID'], axis=1)
    return data 


def data_imputation(data):
    """Loads and prepares the data by imputing null values
    Args:
        input_data: The pandas DataFrame containing the raw data.
    Returns:
        A pandas DataFrame with the 'Loan_ID' column dropped.
    """    
    object_columns = data.select_dtypes(include=["object"]).columns.tolist()
    float_columns = data.select_dtypes(inlcude=["float","int"]).columns.tolist()
    
    imputer_obj = SimpleImputer(strategy='most_frequent')
    imputer_mum = SimpleImputer(strategy='median')

    for col in object_columns:
        data[col] = imputer_obj.fit_transform(data[[col]]).ravel()

    for col in float_columns:
        data[col] = imputer_mum.fit_transform(data[[col]]).ravel()
    
    return data 






def preprocess_data(X_train, X_test):
    data = pd.concat(X_train,X_test,axis=0)
    object_columns = data.drop(columns=['Loan_Status']).select_dtypes(include=["object"]).columns.tolist()
    float_columns = data.select_dtypes(include=["float","int"]).columns.tolist()     
    #Data Imputation
    imputer_obj = SimpleImputer(strategy='most_frequent')
    imputer_mum = SimpleImputer(strategy='median')
    # Apply the imputer to the train and test sets
    for col in object_columns:
        X_train[col] = imputer_obj.fit_transform(X_train[[col]]).ravel()
        X_test[col] = imputer_obj.fit_transform(X_test[[col]]).ravel()

    for col in float_columns:
        X_train[col] = imputer_mum.fit_transform(X_train[[col]]).ravel()
        X_test[col] = imputer_mum.fit_transform(X_test[[col]]).ravel()

    return X_train, X_test
    

def dummy_variable(X_train, X_test):
    """Creates a dummy variable for two features in the dataset"""
    dummy_cols = ['Dependents','Property_Area']
    X_train = pd.get_dummies(X_train, columns=dummy_cols, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=dummy_cols, drop_first=True, dtype=int)
    return X_train, X_test



def encode_target(y_train, y_test):
    """Encodes the target variable."""
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return y_train_encoded, y_test_encoded



def data_encoding(X_train, X_test, y_train, y_test):
    LabelEncoder = LabelEncoder()
    colms = ['Gender','Married','Education','Self_Employed','Credit_History']
    # note that the target variable is also included in the list
    # label encoder assigns numeracal values based on teh alphabe order so N : 0, Y : 1
    X_train[colms] = X_train[colms].apply(LabelEncoder.fit_transform)
    X_test[colms] = X_test[colms].apply(LabelEncoder.fit_transform)
 
    return X_train, X_test
    

def feature_scaling(X_train, X_test, y_train, y_test):
    scalables = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
    MinMaxScaler = MinMaxScaler()
    X_train[scalables] = MinMaxScaler.fit_transform(X_train[scalables])
    X_test[scalables] = MinMaxScaler.transform(X_test[scalables])
    return X_train, X_test, y_train, y_test










