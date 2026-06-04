import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data=pd.read_csv('data/parkinsons_preprocessed.csv')

def data_partition(data,target_name):
    X=data.drop(columns=target_name)
    Y=data[target_name]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                                      Y_train,
                                                      test_size=0.25, random_state=15, shuffle=True, stratify=Y_train
                                                      )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

X_train, X_val, X_test, Y_train, Y_val, Y_test=data_partition(data,'status')

print(X_train.shape,Y_train.shape, X_val.shape, Y_val.shape, X_test.shape,Y_test.shape)


