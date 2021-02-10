import sklearn
from joblib import load
import numpy as np
import os

#Return loaded model
def load_model(modelpath):
    print(modelpath)
    
    # Either load individually
    print("loading individuals")
    logistic = load(os.path.join(modelpath,'logistic.joblib'))
    cart = load(os.path.join(modelpath,'cart.joblib'))
    svm = load(os.path.join(modelpath,'svm.joblib'))
    
    # Or load the entire ensemble
    print("loading ensemble")
    ensemble = load(os.path.join(modelpath,'ensemble.joblib'))
    print("loaded")
    return ensemble

# return prediction based on loaded model (from the step above) and an input payload
def predict(model, payload):
    try:
        # locally, payload may come in as an np.ndarray
        if type(payload)==np.ndarray:
            out = [str(model.predict(payload.reshape((1,8))))]
        # in remote / container based deployment, payload comes in as a stream of bytes
        else:
            out = [str(model.predict(np.frombuffer(payload[0]['body']).reshape((1,8))))]
    except Exception as e:
        out = [type(payload),str(e)] #useful for debugging!
    
    return out
