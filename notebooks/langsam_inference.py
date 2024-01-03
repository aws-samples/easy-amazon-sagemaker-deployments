
import os
import json
import base64
from PIL import Image
from lang_sam import LangSAM
import torch
torch.cuda.empty_cache() 

import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=16'

def load_model(modelpath):
    model = LangSAM()
    print("Loaded LangSAM successfully")

    return model

# return prediction based on loaded model (from the step above) and an input payload
def predict(model, payload):
    
    
    # json_payload = '{"size": [100, 100], "image_bytes": "BASE64_ENCODED_IMAGE_DATA", "text_prompt": "wheel"}'

    # Parse the JSON payload
    payload = json.loads(payload)

    # Decode the Base64 image data back into bytes
    image_bytes_base64 = payload['image_bytes']
    image_bytes = base64.b64decode(image_bytes_base64)
    
    size = payload['size']
    text_prompt = payload['text_prompt']
    
    image_pil = Image.frombytes("RGB",size = size, data = image_bytes) 
    
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    
    return json.dumps({'boxes':boxes.tolist()})
