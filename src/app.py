from utils import db_connect
engine = db_connect()


import pickle
import numpy as np
import pandas as pd
import os

def predict_wine_quality(data):

    #rute to model
    dir = os.path.dirname(__file__)
    model_path = os.path.join(dir, '..', 'models', 'KNN_red_wines.pkl')
    model_path = os.path.abspath(model_path) #absolute route

    #open both model and scaler
    with open(model_path, 'rb') as archive:
        content = pickle.load(archive)

    model = content['model'] #the model inside pkl dict
    scaler = content['scaler'] #the scaler inside pkl

    #formate the input
    data = np.array(data)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)


    #remade a dataframe with same col names

    cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    data = pd.DataFrame(data, columns= cols)

    scaled_data = scaler.transform(data) #scale the new data

    predict = model.predict(scaled_data)

    #results
    if predict == 0:
        return f'This wine probably is low quality 🍷'
    elif predict == 1:
        return f'This wine seems to have medium quality 🍇'
    else:
        return f'This wine is likely high quality! 🍾'


#print(predict_wine_quality([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]))

