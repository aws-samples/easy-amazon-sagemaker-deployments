import os

import numpy as np
from joblib import load


# Return loaded model
def load_model(modelpath):
    print(modelpath)
    clf = load(os.path.join(modelpath, "model.joblib"))
    print("loaded")
    return clf


# return prediction based on loaded model (from the step above) and an input payload
def predict(model, payload):
    try:
        # locally, payload may come in as an np.ndarray
        if type(payload) == np.ndarray:
            out = [str(model.predict(np.frombuffer(payload).reshape((1, 64))))]
        # in remote / container based deployment, payload comes in as a stream of bytes
        else:
            out = [
                str(model.predict(np.frombuffer(payload[0]["body"]).reshape((1, 64))))
            ]
    except Exception as e:
        out = [type(payload), str(e)]  # useful for debugging!

    return out
