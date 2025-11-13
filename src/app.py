from utils import db_connect
engine = db_connect()


import pickle
import numpy as np

def predict_wine_quality(data):

    #open both model and scaler
    with open('../models/KNN_red_wines.pkl', 'rb') as archive:
        content = pickle.load(archive)

    model = content['model'] #the model inside pkl dict
    scaler = content['scaler'] #the scaler inside pkl

    #formate the input
    data = np.array(data)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    scaled_data = scaler.transform(data) #scale the new data

    predict = model.predict(scaled_data)

    #results
    if predict == 0:
        return f'This wine probably is low quality 🍷'
    elif predict == 1:
        return f'This wine seems to have medium quality 🍇'
    else:
        return f'This wine is likely high quality! 🍾'


print(predict_wine_quality([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]))

