import pandas as pd

data = pd.read_csv('Basic Neural Net\exams.csv')
data = data.values.tolist()
data = [(d[:5], d[5:]) for d in data]

validation = pd.read_csv('Basic Neural Net\\validation.csv')
validation = validation.values.tolist()
validation = [(d[:5], d[5:]) for d in validation]

def input_parser(x):
    output = [0.0] * 17
    match x[0]:
        case "male":
            output[0] = 1.0
        case "female":
            output[1] = 1.0
    match x[1]:
        case "group A":
            output[2] = 1.0
        case "group B":
            output[3] = 1.0
        case "group C":
            output[4] = 1.0
        case "group D":
            output[5] = 1.0
        case "group E":
            output[6] = 1.0
    match x[2]:
        case "some high school":
            output[7] = 1.0
        case "high school":
            output[8] = 1.0
        case "some college":
            output[9] = 1.0
        case "associate's degree":
            output[10] = 1.0
        case "bachelor's degree":
            output[11] = 1.0
        case "master's degree":
            output[12] = 1.0
    match x[3]:
        case "standard":
            output[13] = 1.0
        case "free/reduced":
            output[14] = 1.0
    match x[4]:
        case "none":
            output[15] = 1.0
        case "completed":
            output[16] = 1.0
    return output