import sklearn
from joblib import load
import numpy as np
import os

#Return loaded model
def load_model(modelpath):
    print(modelpath)
    clf = load(os.path.join(modelpath,'model.joblib'))
    print("loaded")
    return clf

# return prediction based on loaded model (from the step above) and an input payload
def predict(model, payload):
    print(type(payload))
    try:
        print(np.frombuffer(payload))
        print(np.frombuffer(payload).reshape((1,64)))
        print( model.predict(np.frombuffer(payload).reshape((1,64))) )
        
        out = model.predict(np.frombuffer(payload).reshape((1,64)))
        
    except Exception as e:
        out = [type(payload),str(e)] #useful for debugging!
    
    return out
