from locust import Locust, between, TaskSet, events, task
import os
import sagemaker
import time
import json
import pickle
import sys

with open("src/locustdata.txt") as json_file:
    locustdata = json.load(json_file)

endpoint_name = locustdata["endpoint_name"]
p = sagemaker.predictor.RealTimePredictor(endpoint_name)
target_model = locustdata["target_model"]
input_data = pickle.load(open("src/testdata.p", "rb"))

from locust import Locust, TaskSet, task, between


class MyTaskSet(TaskSet):
    @task
    def my_task(self):
        start_time = time.time()
        try:
            if target_model == "":
                p.predict(input_data)
            else:
                p.predict(input_data, target_model=target_model)

            total_time = int((time.time() - start_time) * 1000)
            events.request_success.fire(
                request_type="sagemaker",
                name="predict",
                response_time=total_time,
                response_length=0,
            )

        except:
            total_time = int((time.time() - start_time) * 1000)
            events.request_failure.fire(
                request_type="sagemaker",
                name="predict",
                response_time=total_time,
                response_length=0,
                exception=sys.exc_info(),
            )


class User(Locust):
    task_set = MyTaskSet
    wait_time = between(0, 1)
