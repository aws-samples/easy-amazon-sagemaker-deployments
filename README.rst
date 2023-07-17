====================================================
Ezsmdeploy - SageMaker custom deployments made easy
====================================================

.. image:: https://img.shields.io/pypi/v/ezsmdeploy.svg
   :target: https://pypi.python.org/pypi/ezsmdeploy
   :alt: Latest Version

.. image:: https://img.shields.io/badge/code_style-black-000000.svg
   :target: https://github.com/python/black
   :alt: Code style: black

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/Made%20With-Love-orange.svg
   :target: https://pypi.python.org/pypi/ezsmdeploy
   :alt: Made With Love

.. image:: https://img.shields.io/badge/Gen-AI-8A2BE2
   :target: https://pypi.python.org/pypi/ezsmdeploy
   :alt: GenAI
   
   

**Ezsmdeploy** python SDK helps you easily deploy Machine learning models on SageMaker. It provides a rich set of features such as deploying models from hubs (like Huggingface or SageMaker Jumpstart), passing one or more model files (even with multi-model deployments), automatically choosing an instance based on model size or based on a budget, and load testing endpoints using an intuitive API. **Ezsmdeploy** uses the SageMaker Python SDK, which is an open source library for training and deploying machine learning models on Amazon SageMaker. This SDK however focuses on further simplifying deployment from existing models, and as such, this is for you if:

1.  you want to quickly deploy and try out foundational language models as an API powered by SageMaker (**New in v 2.0**)
2.  you have a serialized model (a pickle / joblib/ json/ TF saved model/ Pytorch .pth/ etc) file and you want to deploy and test your model as an API endpoint
3. you have a model or multiple models stored as local files, in local folders, or in S3 as tar files (model.tar.gz)
4. you don't want to create a custom docker container for deployment and/or don't want to deal with docker
5. you want to make use of advanced features such as autoscaling, elastic inference, multi-model endpoints, model inference data capture, and locust.io based load testing, without any of the heavy lifting
6. you want to still have control of how do perform inference by passing in a python script

Note for some Sagemaker estimators, deployment from pretrained models is easy; consider the Tensorflow savedmodel format. You can very easily tar your save_model.pb and variables file and use the sagemaker.tensorflow.serving.Model to register and deploy your model. Nevertheless, if your TF model is saved as checkpoints, HDF5 file, or as Tflite file, or if you have deployments needs accross multiple types of serialized model files, this may help standardize your deployment pipeline and avoid the need for building new containers for each model.



V 2.x release notes
-------------------
1. Added support for SageMaker Jumpstart foundational models
2. Added support for Huggingface hub models
3. Added OpenChatKit support for appropriate chat models
4. Tested the following:
    - tiiuae/falcon-40b-instruct, ml.g4dn.12xlarge
    - tiiuae/falcon-7b-instruct, ml.g5.16xlarge
    - WizardLM/WizardLM-7B-V1.0", ml.g5.16xlarge
    - TheBloke/wizardLM-7B-HF, ml.g4dn.4xlarge
    - TheBloke/dromedary-65b-lora-HF, ml.g4dn.4xlarge
    - togethercomputer/RedPajama-INCITE-Chat-3B-v1, ml.g4dn.4xlarge
    - openchat/openchat, ml.g5.24xlarge
    - facebook/galactica-6.7b, ml.g5.16xlarge
    - CalderaAI/30B-Lazarus, ml.g5.16xlarge
    - huggyllama/llama-65b, ml.g5.16xlarge
    - ausboss/llama-30b-supercot, ml.g4dn.4xlarge
    - MetaIX/GPT4-X-Alpasta-30b, ml.g4dn.4xlarge
    - 
    - Also tried several small/tiny models from huggingface on Serverless - (distilbert / dynamic-tinybert / deepset/tinyroberta-squad2 / facebook/detr-resnet-50) 


V 1.x release notes
-------------------
1. Updated to use >v2.x of SageMaker SDK
2. Fixed failing docker builds
3. Tested with test notebook


Table of Contents
-----------------
1. `Installing Ezsmdeploy Python SDK <#installing-the-ezsmdeploy-python-sdk>`__
2. `Key Features <#key-features>`__
3. `Other Features <#other-features>`__
4. `Model script requirements <#model-script-requirements>`__
5. `Sample notebooks <#sample-notebooks>`__
6. `Known gotchas <#known-gotchas>`__

Installing the Ezsmdeploy Python SDK
------------------------------------


The Ezsmdeploy Python SDK is built to PyPI and has the following dependencies sagemaker>=1.55.3, cyaspin==0.16.0,  shortuuid==1.0.1 and locustio==0.14.5. Ezsmdeploy can be installed with pip as follows:

::

    pip install ezsmdeploy

To install locustio for testing, do:


::

    pip install ezsmdeploy[locust]

Cleanest way to install this package is within a virtualenv:


::

    python -m venv env
    
    source env/bin/activate

    pip install ezsmdeploy[locust]


In some cases, installs fail due to an existing package installed called "greenlet". This is not a direct dependency of ezsmdeploy but interferes with the installation. To fix this, either install in a virtualenv as seen above, or do:

::

    pip install ezsmdeploy[locust] --ignore-installed greenlet
    
    
If you have another way to test the endpoint, or want to manage locust on your own, just do:

::

    pip install ezsmdeploy
    
   

Key Features
~~~~~~~~~~~~

At minimum, **ezsmdeploy** requires you to provide:

1. one or more model files
2. a python script with two functions: i) *load_model(modelpath)* - loads a model from a modelpath and returns a model object and ii) *predict(model,input)* - performs inference based on a model object and input data
3. a list of requirements or a requirements.txt file

For example, you can do:

::

    ezonsm = ezsmdeploy.Deploy(model = 'model.pth',
                  script = 'modelscript_pytorch.py',
                  requirements = ['numpy','torch','joblib'])


You can also load multiple models ...

::

    ezonsm = ezsmdeploy.Deploy(model = ['model1.pth','model2.pth'],
                  script = 'modelscript_pytorch.py',
                  requirements = ['numpy','torch','joblib'])    

...  or download tar.gz models from S3
:: 
    
    ezonsm = ezsmdeploy.Deploy(model = ['s3://ezsmdeploy/pytorchmnist/model.tar.gz'],
                  script = 'modelscript_pytorch.py',
                  requirements = 'path/to/requirements.txt')


Other Features
~~~~~~~~~~~~~~~

The **Deploy** class is initialized with these parameters:

::

    class Deploy(object):
    def __init__(
        self,
        model,
        script,
        framework=None,
        requirements=None,
        name=None,
        autoscale=False,
        autoscaletarget=1000,
        wait=True,
        bucket=None,
        session=None,
        image=None,
        dockerfilepath=None,
        instance_type=None,
        instance_count=1,
        budget=100,
        ei=None,
        monitor=False,
    ):


Let's take a look at each of these parameters and what they do:

* You can skip passing in requirements through a file or a list if you choose a **"framework"** in ["tensorflow", "pytorch", "mxnet", "sklearn"]. If you do, these libraries are installed automatically. However it is expected that most people will not use this, given the limited installs, and will usually pass in a custom set of requirements.

 :: 

    ezonsm = ezsmdeploy.Deploy(model = ... ,
                  script = ... ,
                  framework = 'sklearn')

* Pass in a **"name"** if you want to override the random name generated by ezsmdeploy that is used to name your custom ECR image and the endpoint.

 :: 

    ezonsm = ezsmdeploy.Deploy(model = ... ,
                  script = ... ,
                  framework = 'sklearn',
                  name = 'randomname')
                      
                      
* Set **"autoscale"** to True if required to switch on autoscaling for your endpoint. By default, this sets up endpoint autoscaling with the metric *SageMakerVariantInvocationsPerInstance* and a target value of 1000. You can override this value by also passing in a value for autoscaletarget

|

* **"wait**" is set to True by default and can be set to False if you don't want to wait for the endpoint to deploy.

|

* Passing a valid **"bucket"** name will force ezsmdeploy to use this bucket rather than the Sagemaker default session bucket

|

* Pass in a sagemaker **"session"** to override the default session; for most cases this is not necessary. Also, this may interfere with local deployments as the same session cannot be used for tasks such as downloading and uploading files, and for local and remote deployments.

|

* If you already have a prebuild docker image, use the **"image"** argument or pass in a **"dockerfilepath"** if you want ezsmdeploy to use this image. Note that ezsmdeploy will automatically build a custom image with your requirements and the right deployment stack (flask-nginx or MMS) based on the arguments passed in. 

|

* If you do not pass in an **"instance_type"**, ezsmdeploy will choose an instance based on the total size of the model (or multiple models passed in), take into account the multiple workers per endpoint, and also optionally a **"budget"** that will choose instance_type based on a maximum acceptible cost per hour. You can of course, choose an instance as well. We assume you need at least 4 workers and each model is deployed redundantly to every vcpu  available on the selected instance; this eliminates instance tupes with lower number of available vcpus to choose from. If model is being downloaded from a hub (like TF hub or Torch hub or NGC) one should ideally pass in an instance since we don't know the size of model. For all instances that have the same memory per vcpu, what is done to tie break is min (cost/total vpcus). Also 'd' instances are preferred to others for faster load times at the same cost since they have NvMe. 

|

* Passing in an **"instance_count"** > 1 will change the initial number of instances that the model(s) is(are) deployed on.

|

* Pass in a value for **"ei"** or Elastic Inference from this list - ["ml.eia2.medium","ml.eia2.large","ml.eia2.xlarge","ml.eia.medium","ml.eia.large","ml.eia.xlarge"] to add an accelerator to your deployed instance. Read more about Elastic Inference here - https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html

|

* Set **"monitor"** to True if you would like to turn on Datacapture for this endpoint. Currently, a sampling_percentage of 100 is used. Read more about Model monitor here - https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html

|

* You should see an output as follows for a typical deployment:
    
 ::

   0:00:00.143132 | compressed model(s)
   0:00:00.403894 | uploaded model tarball(s) ; check returned modelpath
   0:00:00.404948 | added requirements file
   0:00:00.406745 | added source file
   0:00:00.408180 | added Dockerfile
   0:00:00.409959 | added model_handler and docker utils
   0:00:00.410072 | building docker container
   0:01:59.298091 | built docker container
   0:01:59.647986 | created model(s). Now deploying on ml.m5.xlarge
   0:09:31.904897 | deployed model
   0:09:31.905450 | estimated cost is $0.3 per hour
   0:09:31.905805 | Done! ✔ 


* Once your model is deployed, you can use locust.io to load test your endpoint. The test reports the number of requests, number of failures, average, min, max response time in milliseconds and requests per second reached based on the number of parallel users and hatch rate entered. To load test your model (make sure you have deployed it remotely first), try:
 
 ::

     ezonsm.test(input_data, target_model='model1.tar.gz')
 
 or 

 ::

     ezonsm.test(input_data, target_model='model1.tar.gz',usercount=20,hatchrate=10,timeoutsecs=10)
     
 ... to override default arguments. Read more about locust.io here https://docs.locust.io/en/stable/


Model Script requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure your model script has a load_model() and predict() function. While you can still use sagemaker's serializers and deserializers, assume that you will get a payload in bytes, and that you have to return a prediction in bytes. What you do in between is up to you. For example, your model script may look like:

::

    def load_model(modelpath):
        clf = load(os.path.join(modelpath,'model.joblib'))
        return clf

    def predict(model, payload):
        try:
            # in remote / container based deployment, payload comes in as a stream of bytes
            out = [str(model.predict(np.frombuffer(payload[0]['body']).reshape((1,64))))]
        except Exception as e:
           out = [type(payload),str(e)] #useful for debugging!
    
    return out


Note that when using the Multi model mode, the payload comes in as a dictionary and the raw bytes sent in can be accessed using payload[0]['body']; In flask based deployments, you can just use payload as it is (comes in as bytes)


Large Language models
~~~~~~~~~~~~~~~~~~~~~

EzSMDeploy supports deploying foundation models through Jumpstart as well as huggingface. Genreral guidance:


1. Jumpstart models - `foundation_model=True`
2. Large huggingface models - `foundation_model=True, huggingface_model=True`
3. Small huggingface models - `huggingface_model=True`
4. Tiny models - `serverless=True`


To deploy models using Jumpstart:

::

    ezonsm = ezsmdeploy.Deploy(model = "huggingface-text2text-flan-ul2-bf16",
                               foundation_model=True)
                               
Note that with Jumpstart models, we can automatically retrieve default/suggested instances from SageMaker                               



To deploy a huggingface LLM model (this uses the huggingface llm container):

::

    ezonsm = ezsmdeploy.Deploy(model = "tiiuae/falcon-40b-instruct",
                               foundation_model=True,
                               huggingface_model=True,
                               huggingface_model_task='text-generation',
                               instance_type="ml.g4dn.12xlarge"
                               )
                               
(See release notes for models we have tested so far with instances that worked)

Note that at the time of writing this, officially supported model architectures for LLMs on Huggingface are currently:

    - BLOOM / BLOOMZ
    - MT0-XXL
    - Galactica
    - SantaCoder
    - GPT-Neox 20B (joi, pythia, lotus, rosey, chip, RedPajama, open assistant)
    - FLAN-T5-XXL (T5-11B)
    - Llama (vicuna, alpaca, koala)
    - Starcoder / SantaCoder
    - Falcon 7B / Falcon 40B





Serverless inference
~~~~~~~~~~~~~~~~~~~~

Simply do `serverless=True`. Make sure you size your serverless endpoint correctly using `serverless_memory` and `serverless_concurrency`. You can combine other features as well, for example, to deploy a huggingface model on serverless use:

::

    ezonsm = ezsmdeploy.Deploy(model = "distilbert-base-uncased-finetuned-sst-2-english",
                               huggingface_model=True,
                               huggingface_model_task='text-classification',
                               serverless=True
                               )


Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ezsmdeploy SDK has been tested on Unix/Linux.

Supported Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~~

Ezsmdeploy SDK has been tested on Python 3.6; should run in higher versions!

AWS Permissions
~~~~~~~~~~~~~~~
Ezsmdeploy uses the  Sagemaker python SDK.

As a managed service, Amazon SageMaker performs operations on your behalf on the AWS hardware that is managed by Amazon SageMaker.
Amazon SageMaker can perform only operations that the user permits.
You can read more about which permissions are necessary in the `AWS Documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__.

The SageMaker Python SDK should not require any additional permissions aside from what is required for using SageMaker.
However, if you are using an IAM role with a path in it, you should grant permission for ``iam:GetRole``.

Licensing
~~~~~~~~~
Ezsmdeploy is licensed under the MIT license and uses the SageMaker Python SDK. SageMaker Python SDK is licensed under the Apache 2.0 License. It is copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at: http://aws.amazon.com/apache2.0/ 

Sample Notebooks
~~~~~~~~~~~~~~~~~
https://github.com/aws-samples/easy-amazon-sagemaker-deployments/tree/master/notebooks

Known Gotchas
~~~~~~~~~~~~~~~~~~
* Ezsmdeploy uses the sagemaker python sdk under the hood, so any limitations / limits / restrictions are expected to be carried over

|

* Ezsmdeploy builds your docker container on the fly, and uses two types of base containers - a flask-nginx deployment stack or the Multi model server. Sending in a single model, or choosing to use a GPU instance will default to the flask-nginx stack. You can force the use of the MMS stack if you pass in a single model as a list, for example, ['model1.joblib']

|

* Ezsmdeploy uses a local 'src' folder as a staging folder which is reset at the beginning of every deploy. So consider using the package in separate project folders so there is no overlap/ overwriting  of staging files.

|

* Ezsmdeploy uses Locust to do endpoint testing - any restrictions of the locustio package are also expected to be seen here.

|

* Ezsmdeploy has been tested from Sagemaker notebook instances (both GPU and non-GPU). 

|

* The payload comes in as bytes; you can also use Sagemaker's serializer and deserializers to send in other formats of input data

|

* Not all feature combinations are tested; any contributions testing, for example, budget constraints are welcome!

|

* If you are doing local testing in a container, make sure you kill any running containers, since any invocations hit the same port. to do this, run:

::

    docker container stop $(docker container ls -aq) >/dev/nul

* If your docker push fails, chances are that your disk is full. Try. clearing some docker images:

::

    docker system prune -a

* If you encounter an "image does not exist" error, try running this script that exists after an unsuccessful run, but manually. For this, do:

::

   ./src/build-docker.sh 

* Locust load testing on local endpoint has not been tested (and may not make much sense). Please use the .test() for remote deployment

|

* Use instance_type "local" if you would like to test locally (this lets you test using the MMS stack). If you intend to finally deploy your model to a GPU instance, use "local_gpu" - this launches the flask-nginx stack locally and the same stack when you deploy to a GPU.

|

* At the time of writing this guide, launching a multi-model server from sagemaker does not support GPUs (but the open source MMS repository has no such restrictions). Ezsmdeploy checks the number of models passed in, the instance type and other parameters to decide which stack to build for your endpoint.


CONTRIBUTING
------------

Please submit a pull request to the packages git repo



