import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import json

#Return loaded model
def load_model(modelpath):
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 
    return model

# return prediction based on loaded model (from the step above) and an input payload
def predict(model, payload):
    try:
        if(type(payload) == str):
            data = [payload]
        else:
            data = [payload.decode()]# Multi model endpoints -> [payload[0]['body'].decode()]
            
        out = np.asarray(model(data)).tolist()
    except Exception as e:
        out = str(e)
    return [json.dumps({'output':[out],'tfeager': tf.executing_eagerly()})]