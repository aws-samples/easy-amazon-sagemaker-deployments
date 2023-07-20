import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image


# Return loaded model
def load_model(modelpath):
    model_file = os.path.join(modelpath, "mobilenet_v1_1.0_224_quant.tflite")
    model = tf.lite.Interpreter(model_path=model_file)
    model.allocate_tensors()
    return model


# return prediction based on loaded model (from the step above) and an input payload
def predict(model, payload):
    try:
        img = Image.frombytes(
            "RGB", (224, 224), payload, "raw"
        )  # For Multi model endpoints -> [payload[0]['body'].decode()]

        # img = np.frombuffer(data, dtype=np.uint8).reshape((224, 224))

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # check the type of the input tensor
        floating_model = input_details[0]["dtype"] == np.float32

        # NxHxWxC, H:1, W:2
        input_details[0]["shape"][1]
        input_details[0]["shape"][2]
        # img = Image.open(image_file).resize((width, height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        model.set_tensor(input_details[0]["index"], input_data)

        model.invoke()

        output_data = model.get_tensor(output_details[0]["index"])
        results = np.squeeze(output_data)

        out = results.tolist()
    except Exception as e:
        out = str(e)
    return [json.dumps({"output": [out], "tfeager": tf.executing_eagerly()})]
