import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100
    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')


def predict_func(data):
    """
    Modify this function to predict closing prices for the next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for the closing price of the next 2 samples
    """
    model = load_model('lstm_Model.h5')
    close_prices = data['Close']
    close_prices.dropna(inplace=True)
    values = close_prices.values
    training_data_len = math.ceil(len(values))
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))
    train_data = scaled_data[:training_data_len, :]

    x_train = scaled_data[-2:].reshape(1, -1, 1)
    # Predict the next 2 samples
    predictions = []
    for i in range(2):
        pred = model.predict(x_train)
        predictions.append(pred[0][0])
        x_train = np.append(x_train[:, 1:, :], pred.reshape(1,1, 1), axis=1)
    
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions.reshape(1, -1)).flatten()
    pred_1 = predictions[0]
    pred_2 = predictions[1]
    return [pred_1, pred_2]
    

if __name__== "__main__":
    evaluate()