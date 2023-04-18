import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        #data loading
        self.df = pd.read_csv('data/data.csv')
        print(f"{len(self.df)} lines loaded")
        
    def trim(self):
        #converting column names to lowercase and replacing whitespace with _
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')
            
    def validate(self,y_pred,y):
        #calculatin RMSE between ground truth and prediction
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)
    
    def linear_regression(self,X, y):
        #linear regression model
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        return w[0], w[1:]
    
    def prepare_X(self,df,base):
        #data preparation
        df_num = df[base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X
    
    def predict(self,data,w0,w):
        #prediction using linear regression model
        y_pred = w0 + data.dot(w)
        return y_pred

def test():
    carPrice = CarPrice()
    carPrice.trim() 
    df = carPrice.df
    
    np.random.seed(2)
    n = len(df)
    # splitting of data to test(20%), validation(20%) and train(60%)
    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - (n_val + n_test)

    idx = np.arange(n)
    # shuffling the data
    np.random.shuffle(idx)

    df_shuffled = df.iloc[idx]

    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test = df_shuffled.iloc[n_train+n_val:].copy()
    
    # ground truth msrp for each set
    y_train_orig = df_train.msrp.values
    y_val_orig = df_val.msrp.values
    y_test_orig = df_test.msrp.values

    
    y_train = np.log1p(df_train.msrp.values)
    y_val = np.log1p(df_val.msrp.values)
    y_test = np.log1p(df_test.msrp.values)
    
    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
    # preparing training data
    X_train = carPrice.prepare_X(df_train,base)
    # applying linear regression model on training data
    w_0, w = carPrice.linear_regression(X_train, y_train)
    # preparing validation data
    X_val = carPrice.prepare_X(df_val,base)
    # predicting on validation data
    y_pred_val = carPrice.predict(X_val,w_0,w)
    #calculating rmse on validation
    val_rmse = carPrice.validate(y_val, y_pred_val)
    print("The rmse value of predicted MSRP and actual MSRP of validation set is ",val_rmse)
    # preparing test data
    X_test = carPrice.prepare_X(df_test,base)
    # predicting on test data
    y_pred_test = carPrice.predict(X_test,w_0,w)
    #calculating rmse on test
    test_rmse=carPrice.validate(y_test, y_pred_test)
    print("The rmse value of predicted MSRP and actual MSRP of test set is ",test_rmse)
 
    y_pred_MSRP_val = np.expm1(y_pred_val) # expm1 calculates exp(x) - 1

    df_val['msrp_pred'] = y_pred_MSRP_val # Add the column
    print("First 5 cars in Validation Set's original msrp vs. predicted msrp")
    print(df_val.iloc[:,5:].head().to_markdown(), "\n")
    
    y_pred_MSRP_test = np.expm1(y_pred_test) # expm1 calculates exp(x) - 1

    df_test['msrp_pred'] = y_pred_MSRP_test # Add the column
    print("First 5 cars in Test Set's original msrp vs. predicted msrp")
    print(df_test.iloc[:,5:].head().to_markdown(), "\n")

if __name__ == "__main__":
    test()
   