import gluonnlp as nlp; import mxnet as mx;
from joblib import load
import numpy as np
import os
import json

#Return loaded model
def load_model(modelpath):
    model, vocab = nlp.model.get_model('distilbert_6_768_12', dataset_name='distilbert_book_corpus_wiki_en_uncased');
    print("loaded")
    return {'model':model,'vocab':vocab}

# return prediction based on loaded model (from the step above) and an input payload
def predict(modeldict, payload):
    
    #set_trace()
    
    model = modeldict['model']
    vocab = modeldict['vocab']
    
    tokenizer = nlp.data.BERTTokenizer(vocab, lower=True);
    transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=512, pair=False, pad=False);
    
    try:
        # Local
        if type(payload) == str:
            sample = transform(payload);
        elif type(payload) == bytes :
            sample = transform(str(payload.decode()));
        # Remote, standard payload comes in as a list of json strings with 'body' key
        elif type(payload)==list:
            sample = transform(payload[0]['body'].decode());
        else:
            return [json.dumps({'response':"Provide string or bytes string",
                    'payload':str(payload),
                    'type':str(type(payload))})]
        
        words, valid_len = mx.nd.array([sample[0]]), mx.nd.array([sample[1]])
        out = model(words, valid_len)  
        out = json.dumps({'output':out.asnumpy().tolist()})
    except Exception as e:
        out = str(e) #useful for debugging!
    return [out]
