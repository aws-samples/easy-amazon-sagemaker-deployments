import json
import os

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# Return loaded model
def load_model(modelpath):
    # (re)Defne model
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(1 + 1),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])
                )
            ),
        ]
    )

    # Load model
    print("1. Listing files in modelpath")
    print(os.listdir(modelpath))

    print("2. Loading h5 file")
    file = h5py.File(os.path.join(modelpath, "reg1.h5"), "r")

    print("3. Loading weights")
    weight = []
    for i in range(len(file.keys())):
        weight.append(file["weight" + str(i)][:])

    model.build(input_shape=(150, 1))
    model.set_weights(weight)

    print("4. Loaded model successfully")

    return model


# return prediction based on loaded model (from the step above) and an input payload
def predict(model, payload):
    try:
        # Note, for Multi model endpoints -> (payload[0]['body'].decode())
        data = np.frombuffer(payload, dtype=np.float32).reshape((150, 1))
        tmpout = model(data)

        # Add outputs here !!

        out = {
            "mean": np.asarray(tmpout.mean()).T.tolist(),
            "mode": np.asarray(tmpout.mode()).T.tolist(),
            "stddev": np.asarray(tmpout.stddev()).T.tolist(),
            "quantile_75": np.asarray(tmpout.quantile(0.75)).T.tolist(),
        }

    except Exception as e:
        out = str(e)
    return [json.dumps({"output": out})]
