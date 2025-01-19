import pandas as pd
import numpy as np
import sklearn.linear_model as logistic_regression
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold,train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, mean_squared_error
from sklearn.preprocessing import  MinMaxScaler,OneHotEncoder,LabelEncoder,RobustScaler,StandardScaler
import pickle
from sklearn.impute import SimpleImputer



def prepare_data(data):
    """Loads and prepares the data by dropping unnecessary columns.
    Args:
        input_data: The pandas DataFrame containing the raw data.
    Returns:
        A pandas DataFrame with the 'Loan_ID' column dropped.
    """
    data = data.drop(['Loan_ID'], axis=1)
    return data 



def data_imput(data):
    """Loads and prepares the data by imputing null values
    Args:
        input_data: The pandas DataFrame containing the data with their respective data types as (obj, float,or int)
    Returns:
        A pandas DataFrame with imputed values for missing data.
    """    
    object_columns = data.select_dtypes(include=["object"]).columns.tolist()
    float_columns = data.select_dtypes(include=["float","int"]).columns.tolist()
    
    imputer_obj = SimpleImputer(strategy='most_frequent')
    imputer_mum = SimpleImputer(strategy='median')

    for col in object_columns:
        data[col] = imputer_obj.fit_transform(data[[col]]).ravel()

    for col in float_columns:
        data[col] = imputer_mum.fit_transform(data[[col]]).ravel()
    
    return data 



def var_dummy(data):
    """crates a dummy variable from a data received.
    Args:
        input_data: A pandas DataFrame 
    Returns:
        A pandas DataFrame with dummy variable created for respective columns
    """
    dum_dum = ['Dependents','Property_Area']
    data = pd.get_dummies(data,columns=dum_dum,drop_first=True, dtype=int)
    return data



def label_encod(data,colmns):
    """label encodes specified clumns
    Args:
        input_data: a Frame with the specific columns to be converted.
        variables: be sure to create the object LabelEncoder with the same name. Also specify the colms as colms.
    Returns:
        A pandas DataFrame with new columns created out of the label variables.
    """
    for col in colmns:
        le_2 = LabelEncoder()
        data[col] = le_2.fit_transform(data[col])
    return data


def data_scaler(data, scalables):
    """scales data for specified clumns
    Args:
        input_data: a Frame with the specific columns to be converted.
        variables: be sure to create the object MinMaxScaler with the same name. Also specify the colms as scalables.
    Returns:
        A pandas DataFrame with scaled values of the columns provided.
    """
    for col in scalables:
        scaler = MinMaxScaler()
        scaler.fit(data[col].values.reshape(-1,1))
        data[col] = scaler.transform(data[col].values.reshape(-1,1))
    return data


