from training_data import input_parser
import numpy as np
import pandas as pd
from statsmodels.api import OLS



cols = ['const', 'male', 'raceA', 'raceB', 'raceC', 'raceD', 'edu1', 'edu2', 'edu3', 'edu4', 'edu5', 'lunch', 'prep']

class Linear_Model:
    def __init__(self, data):
        x = np.array([[1]+input_parser(d[0]) for d in data])
        y_math = np.array([d[1][0]/100.0 for d in data])
        y_reading = np.array([d[1][1]/100.0 for d in data])
        y_writing = np.array([d[1][2]/100.0 for d in data])

        df = pd.DataFrame(x, columns=cols)
        df["math"] = y_math
        df["reading"] = y_reading
        df["writing"] = y_writing

        model_math = OLS(df["math"], df[cols]).fit()
        model_reading = OLS(df["math"], df[cols]).fit()
        model_writing = OLS(df["math"], df[cols]).fit()
        self.models = [model_math, model_reading, model_writing]

    def __call__(self, x):
        x = [1] + input_parser(x)
        x = np.array(x).reshape(1, -1)  # Convert to 2D array with 1 row
        return [model.predict(x)[0] for model in self.models]

#print(model.summary())