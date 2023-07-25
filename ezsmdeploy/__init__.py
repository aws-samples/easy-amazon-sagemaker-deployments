import sagemaker
import shortuuid
from yaspin import yaspin
from yaspin.spinners import Spinners
import time
import datetime
import tarfile
import re
import boto3
import glob
import os
import shutil
import pkg_resources
import subprocess
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.model import Model
import ast
import csv
import json
import pickle
import time
import cmd
from typing import List, Optional
from sagemaker.predictor import Predictor

_model_env_variable_map = {
    "huggingface-text2text-flan-t5-xxl": {"TS_DEFAULT_WORKERS_PER_MODEL": "1"},
    "huggingface-text2text-flan-t5-xxl-fp16": {"TS_DEFAULT_WORKERS_PER_MODEL": "1"},
    "huggingface-text2text-flan-t5-xxl-bnb-int8": {"TS_DEFAULT_WORKERS_PER_MODEL": "1"},
    "huggingface-text2text-flan-t5-xl": {"MMS_DEFAULT_WORKERS_PER_MODEL": "1"},
    "huggingface-text2text-flan-t5-large": {"MMS_DEFAULT_WORKERS_PER_MODEL": "1"},
    "huggingface-text2text-flan-ul2-bf16": {"TS_DEFAULT_WORKERS_PER_MODEL": "1"},
    "huggingface-text2text-bigscience-t0pp": {"TS_DEFAULT_WORKERS_PER_MODEL": "1"},
    "huggingface-text2text-bigscience-t0pp-fp16": {"TS_DEFAULT_WORKERS_PER_MODEL": "1"},
    "huggingface-text2text-bigscience-t0pp-bnb-int8": {"TS_DEFAULT_WORKERS_PER_MODEL": "1"},
    "huggingface-textgeneration2-gpt-neoxt-chat-base-20b-fp16":{"TS_DEFAULT_WORKERS_PER_MODEL": "1"}
}



class utils(object):
    def __init__(self):
        pass
    def list_foundation_models(self, filter_value="task == text2text"):
        from sagemaker.jumpstart.notebook_utils import list_jumpstart_models
        # print('\n'.join(list(_model_env_variable_map.keys())))
        text_generation_models = list_jumpstart_models(filter=filter_value)
        print("List of foundation models in Jumpstart: \n")
        print("\n".join(text_generation_models))
    
class Deploy(object):
    def __init__(
        self,
        model,
        script=None,
        framework=None,
        requirements=None,
        name=None,
        autoscale=False,
        autoscaletarget=1000,
        serverless=False,
        serverless_memory=4096,
        serverless_concurrency=10,
        wait=True,
        bucket=None,
        prefix='',
        session=None,
        image=None,
        dockerfilepath=None,
        instance_type=None,
        instance_count=1,
        budget=100,
        ei=None,
        monitor=False,
        foundation_model=False,
        foundation_model_version='*',
        huggingface_model=False,
        huggingface_model_task=None,
        huggingface_model_quantize=None
    ):

        self.frameworklist = ["tensorflow", "pytorch", "mxnet", "sklearn","huggingface"]
        self.frameworkinstalls = {
            "tensorflow": ["tensorflow"],
            "pytorch": ["torch"],
            "mxnet": ["mxnet", "gluon"],
            "sklearn": ["sklearn"],
        }

        self.wait = wait
        self.budget = budget
        self.instance_count = instance_count
        self.instance_type = instance_type
        self.image = image
        self.dockerfilepath = dockerfilepath
        self.ei = ei
        self.prefix = prefix
        self.monitor = monitor
        self.deployed = False
        self.autoscaletarget = autoscaletarget
        self.foundation_model = foundation_model
        self.foundation_model_version = foundation_model_version
        self.huggingface_model = huggingface_model
        self.huggingface_model_task = huggingface_model_task
        self.serverless = serverless
        
        if serverless:
            self.serverless_config = ServerlessInferenceConfig(
                memory_size_in_mb=serverless_memory, max_concurrency=serverless_concurrency)
        else:
            self.serverless_config = None
            
        self.huggingface_model_quantize = huggingface_model_quantize

        # ------ load cost types dict ---------
        costpath = pkg_resources.resource_filename("ezsmdeploy", "data/cost.csv")
        self.costdict = {}
        with open(costpath, mode="r") as infile:
            reader = csv.reader(infile)
            for rows in reader:
                # cost for each instance
                self.costdict[rows[0]] = float(rows[1])

        # ------- basic instance type check --------

        if (
            self.instance_type == None
        ):  # since we will not select a a GPU instance in automatic instance selection
            self.gpu = False
            self.multimodel = True
        else:

            if (
                (self.instance_type in list(self.costdict.keys()))
                or "local" in self.instance_type
            ) and self.instance_type != None:

                if "local" in self.instance_type:
                    if (
                        self.instance_type == "local_gpu"
                    ):  # useful if you intend to do local testing. No change vs. local
                        self.gpu = True
                        self.multimodel = False
                        self.instance_type == "local"
                    else:
                        self.gpu = False
                        self.multimodel = True

                else:
                    if self.instance_type.split(".")[1][0] in [
                        "p",
                        "g",
                    ]:  # if gpu instance
                        self.gpu = True
                        self.multimodel = False
                    else:
                        self.gpu = False
                        self.multimodel = (
                            True  # multi model works well with local endpoints ....
                        )

            else:  # throw wrong instance error
                raise ValueError(
                    "Please choose an instance type in",
                    list(self.costdict.keys()),
                    ", or choose local for local testing. Don't pass in any instance or pass in None if you want to automatically choose an instance type.",
                )

        # ------- Model checks --------
        if type(model) == str:
            self.model = [model]
            self.multimodel = False

        elif type(model) == list:
            self.model = model
            self.multimodel = True
        elif model == None:  # assume you are loading from a hub or from a dockerfile
            with open("tmpmodel", "w") as fp:
                pass
            self.model = ["tmpmodel"]
            self.multimodel = False
        else:
            raise ValueError(
                "model must be a single serialized file (like 'model.pkl') or a \
                list of files ([model.pkl, model2.pkl]). If you are downloading a model in the script \
                or packaging with the container, pass in model = None"
            )
            
        # if self.huggingface_model:
        #     # if self.huggingface_model_task == None:
        #     #     print("Using None as task type")
        #     # else:
        #     #     pr
                # raise ValueError("Please specify huggingface_model_task. For example: question-answering")

        # ------- Script checks ---------
        if not (self.foundation_model or self.huggingface_model):
            if script[-2:] != "py":
                raise ValueError(
                    "please provide a valid python script with .py extension. "
                    + script
                    + " is invalid"
                )
            else:
                self.script = script

            filename = self.script
            with open(filename) as file:
                node = ast.parse(file.read())
                functions = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

            if ("load_model" not in functions) and ("predict" not in functions):
                raise ValueError(
                    "please implement a load_model(modelpath) that \
                    returns a loaded model, and predict(inputdata) function that returns a prediction in your"
                    + script
                )

        # ------- session checks --------
        if session == None:
            self.session = sagemaker.session.Session()
        else:
            self.session = session  # leave session as none since users may want to do local testing.

        # ------- name checks --------
        if name == None:
            self.name = shortuuid.uuid().lower()
        elif type(name) == str:
            self.name = name
            if name.islower() == False:
                raise ValueError(
                    "please enter a name with lower case letters; we will be using this name for s3 bucket prefixes, model names, ECR repository names etc. that have various restrictions"
                )

        else:
            raise ValueError(
                "enter string for a name or don't pass in a name; type of name passed in is "
                + str(type(name))
            )

        # ------- bucket checks --------
        if bucket == None:
            self.bucket = self.session.default_bucket()
        else:
            self.bucket = bucket

        self.requirements = requirements

        # ------- framework --------
        if requirements == None and framework in self.frameworklist:
            self.framework = framework
            self.requirements = self.frameworkinstalls[framework]
        elif requirements == None and framework not in self.frameworklist and not self.foundation_model and not self.huggingface_model:
            raise ValueError(
                "If requirements=None, please provide a value for framework; \
                    choice should be one of 'tensorflow','pytorch','mxnet','sklearn'"
            )

        self.autoscale = autoscale

        self.wait = wait
        self.deploy()

    
    
    def deploy_huggingface_model(self):
        
        if self.instance_type == None and not self.serverless:
            raise ValueError("Please enter a valid instance type, not [None]")
        
        # from sagemaker.huggingface.model import HuggingFaceModel, get_huggingface_llm_image_uri
        from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
        
        self.model = self.model[0]
        hub = {
            'HF_MODEL_ID':self.model,                # model_id from hf.co/models
        }
        
        
        if self.huggingface_model_task is not None:
             hub['HF_TASK'] = self.huggingface_model_task    # NLP task you want to use for predictions
        
        if not self.serverless: #ignore instance type checks, and ignore fallback. Also ignore quantization
            try:
                ec2_client = boto3.client("ec2")
                resp = ec2_client.describe_instance_types(InstanceTypes=[self.instance_type.strip("ml.")])['InstanceTypes'][0]
                if 'GpuInfo' in resp:
                    hub['SM_NUM_GPUS']= json.dumps(resp['GpuInfo']['Gpus'][0]['Count'])
                else:
                    pass
            except Exception as e:
                print(e, end=" ... ")
                print("Trying fallback to figure out number of GPUs in the instance type you chose - ")
                hub['SM_NUM_GPUS']= json.dumps(0) # Use at least 0 GPU by default. else:
                if 'g' in self.instance_type:
                    hub['SM_NUM_GPUS']= json.dumps(1)
                    if '12x' in self.instance_type:
                        hub['SM_NUM_GPUS']= json.dumps(4)
                    elif '24x' in self.instance_type:
                        hub['SM_NUM_GPUS']= json.dumps(4)
                    elif '48x' in self.instance_type:
                        hub['SM_NUM_GPUS']= json.dumps(4)
                elif 'p2' in self.instance_type:
                    hub['SM_NUM_GPUS']= json.dumps(1)
                    if '8x' in self.instance_type:
                        hub['SM_NUM_GPUS']=json.dumps(8)
                    elif '16x' in self.instance_type:
                        hub['SM_NUM_GPUS']=json.dumps(16)
                elif 'p3' in self.instance_type:
                    hub['SM_NUM_GPUS']= json.dumps(1)
                    if '8x' in self.instance_type:
                        hub['SM_NUM_GPUS']=json.dumps(4)
                    elif '16x' in self.instance_type:
                        hub['SM_NUM_GPUS']=json.dumps(8)
                    elif '24x' in self.instance_type:
                        hub['SM_NUM_GPUS']=json.dumps(8)
                elif 'p4' in self.instance_type:
                    hub['SM_NUM_GPUS']=json.dumps(8)




            if self.huggingface_model_quantize in ['bitsandbytes', 'gptq']:
                hub['HF_MODEL_QUANTIZE'] = self.huggingface_model_quantize
            elif self.huggingface_model_quantize==None:
                pass
            else:
                raise ValueError(f"huggingface_model_quantize needs to be one of bitsandbytes, gptq, not {self.huggingface_model_quantize}")

        
        
        aws_role = sagemaker.get_execution_role()
        endpoint_name = name="hf-model-" + self.name
        
        # create Hugging Face Model Class
        if not self.serverless and self.foundation_model and self.huggingface_model: #Basically just for large models
            self.sagemakermodel = HuggingFaceModel(
               image_uri=get_huggingface_llm_image_uri("huggingface"),
               env=hub,                                                # configuration for loading model from Hub
               role=aws_role,                                          # IAM role with permissions to create an endpoint
               name=endpoint_name,
               transformers_version="4.26",                             # Transformers version used
               pytorch_version="1.13",                                  # PyTorch version used
               py_version='py39',                                      # Python version used
            )
        else:
            self.sagemakermodel = HuggingFaceModel(
                env=hub,                      # configuration for loading model from Hub
                role=aws_role,                    # iam role with permissions to create an Endpoint
                transformers_version="4.26",  # transformers version used
                pytorch_version="1.13",        # pytorch version used
                py_version='py39',            # python version used
                )
    
    def deploy_foundation_model(self):
        # Assume foundation model on Jumpstart here

        
        from sagemaker import image_uris, instance_types, model_uris, script_uris
        self.model = self.model[0]
        # print("self.model = ", self.model)
        instance_type = instance_types.retrieve_default(
            model_id=self.model, model_version=self.foundation_model_version, scope="inference"
        )
        if self.instance_type == None:
            self.instance_type = instance_type
        # self.costperhour = self.costdict[self.instance_type]

        # Retrieve the inference docker container uri. 
        deploy_image_uri = sagemaker.image_uris.retrieve(
            region=None,
            framework=None,  # automatically inferred from model_id
            image_scope="inference",
            model_id=self.model,
            model_version=self.foundation_model_version,
            instance_type=instance_type,
        )

        # Retrieve the inference script uri. This includes all dependencies and scripts for model loading, inference handling etc.
        deploy_source_uri = sagemaker.script_uris.retrieve(
            model_id=self.model, model_version=self.foundation_model_version, script_scope="inference"
        )

        # Retrieve the model uri.
        model_uri = sagemaker.model_uris.retrieve(
            model_id=self.model, model_version=self.foundation_model_version, model_scope="inference"
        )
        aws_role = sagemaker.get_execution_role()
        endpoint_name = "model-" + self.name
        
        # Create the SageMaker model instance
        if self.model in _model_env_variable_map:
            # For those large models, we already repack the inference script and model
            # artifacts for you, so the `source_dir` argument to Model is not required.
            
            
            self.sagemakermodel = sagemaker.model.Model(
                image_uri=deploy_image_uri,
                model_data=model_uri,
                role=aws_role,
                predictor_cls=sagemaker.predictor.Predictor,
                name=endpoint_name,
                env=_model_env_variable_map[self.model],
            )
        else:
            from sagemaker.jumpstart.model import JumpStartModel
            
            self.sagemakermodel =  JumpStartModel(model_id = self.model,
                                             model_version = self.foundation_model_version,
                                             role=aws_role)

            
    def get_sagemaker_session(self,local_download_dir='src') -> sagemaker.Session:
        """Return the SageMaker session."""

        sagemaker_client = boto3.client(
            service_name="sagemaker", region_name=boto3.Session().region_name
        )

        session_settings = sagemaker.session_settings.SessionSettings(
            local_download_dir=local_download_dir
        )

        # the unit test will ensure you do not commit this change
        session = sagemaker.session.Session(
            sagemaker_client=sagemaker_client, settings=session_settings
        )

        return session
        
    def process_instance_type(self):
        # ------ instance checks --------
        self.instancedict = {}
    
        if self.instance_type == None:
            # ------ load instance types dict ---------
            instancetypepath = pkg_resources.resource_filename(
                "ezsmdeploy", "data/instancetypes.csv"
            )
            with open(instancetypepath, mode="r") as infile:
                reader = csv.reader(infile)
                for rows in reader:  # memGb / vcpu, cost, cost/memGb-per-vcpu
                    self.instancedict[rows[0]] = (
                        float(rows[2]) / (2 * float(rows[1])),
                        self.costdict[rows[0]],
                        self.costdict[rows[0]] / float(rows[2]) / (2 * float(rows[1])),
                    )

            # ------ auto instance selection ---------
            self.choose_instance_type()

        else:

            if (self.instance_type in list(self.costdict.keys())) or (
                self.instance_type in ["local", "local_gpu"]
            ):
                if self.instance_type not in ["local", "local_gpu"]:
                    self.costperhour = self.costdict[self.instance_type]

                    if self.ei != None:
                        eicosts = {
                            "ml.eia2.medium": 0.12,
                            "ml.eia2.large": 0.24,
                            "ml.eia2.xlarge": 0.34,
                            "ml.eia.medium": 0.13,
                            "ml.eia.large": 0.26,
                            "ml.eia.xlarge": 0.52,
                        }
                        self.costperhour = self.costperhour + eicosts[self.ei]

                else:
                    self.costperhour = 0
            else:
                raise ValueError(
                    "Please choose an instance type in",
                    list(self.costdict.keys()),
                    ", or choose local for local testing.",
                )

    def choose_instance_type(self):
        # TO DO : add heuristic for auto selection of instance size
        
        if self.prefix =='':
            tmppath = "ezsmdeploy/model-" + self.name + "/"
        else:
            tmppath = self.prefix+"/ezsmdeploy/model-" + self.name + "/"

        size = self.get_size(self.bucket, tmppath )

        self.instancetypespath = pkg_resources.resource_filename(
            "ezsmdeploy", "data/instancetypes.csv"
        )

        # Assume you need at least 4 workers, each model is deployed redundantly to every vcpu.
        # So we base this decision on memory available per vcpu. If model is being downloaded from a hub
        # one should ideally pass in an instance since we don't know the size of model.
        # list includes some extremely large CPU instance and all GPU instances. For all instances that have the same
        # memory per vcpu, what is done to tie break is min (cost/total vpcus). Also 'd' instances are preferred to others for
        # faster load times at the same cost since they have NvMe. If budget is supplied, we can try to satisfy this.

        choseninstance = None
        mincost = 1000

        for instance in list(self.instancedict.keys()):
            # cost and memory per worker
            memperworker = self.instancedict[instance][0]
            cost = self.instancedict[instance][1]
            costpermem = self.instancedict[instance][2]
            #
            if self.budget == 100:
                # even though budget is unlimited, minimize cost
                if memperworker > size and cost < mincost:
                    mincost = cost
                    choseninstance = instance
                    # print("instance ={}, size={}, memperworker={}, choseninstance = {}, mincost = {}".format(instance, size, memperworker, choseninstance,mincost))
            else:
                if memperworker > size and cost <= self.budget:
                    choseninstance = instance
                    break

        if choseninstance == None and self.budget != 100:
            raise ValueError(
                "Could not find an instance that satisfies your budget of "
                + str(self.budget)
                + " per hour and can host your models with a total size of "
                + str(size)
                + " Gb. Please choose a higher budget per hour."
            )
        elif choseninstance == None and self.budget == 100:
            raise ValueError(
                "You may be using large models with a total size of "
                + str(size)
                + " Gb. Please choose a high memory GPU instance and launch without multiple models (if applicable)"
            )

        self.instance_type = choseninstance

        self.costperhour = self.costdict[self.instance_type]

    def add_model(self, s3path, relativepath):
        self.sagemakermodel.add_model(s3path, relativepath)

    def create_model(self):

        if not self.multimodel:

            self.sagemakermodel = Model(
                name="model-" + self.name,
                model_data=self.modelpath[0],
                image_uri=self.image,
                role=sagemaker.get_execution_role(),
                # sagemaker_session=self.session,
                predictor_cls=sagemaker.predictor.Predictor,
            )

        else:

            self.sagemakermodel = MultiDataModel(
                name="model-" + self.name,
                model_data_prefix="/".join(self.modelpath[0].split("/")[:-1]) + "/",
                image_uri=self.image,
                role=sagemaker.get_execution_role(),
                # sagemaker_session=self.session,
                predictor_cls=sagemaker.predictor.Predictor,
            )

            for path in self.modelpath:
                self.add_model(path, "serving/")

            self.ei = False

    def deploy_model(self):

        if self.monitor:
            from sagemaker.model_monitor import DataCaptureConfig
            
            
            if prefix == '':
                tmps3uri = "s3://{}/ezsmdeploy/model-{}/datacapture".format(
                    self.bucket, self.name
                )
            else:
                tmps3uri = "s3://{}/{}/ezsmdeploy/model-{}/datacapture".format(
                    self.bucket, self.prefix, self.name
                )
            
            data_capture_config = DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=100,
                destination_s3_uri=tmps3uri
            )
        else:
            data_capture_config = None
        
        if self.instance_type is not None:
            volume_size = None
            if "p3" in self.instance_type or "p4" in self.instance_type or "16x" in self.instance_type or "24x" in self.instance_type or "48x" in self.instance_type or self.foundation_model:
                volume_size = 256 
            
            if "g5" in self.instance_type:
                volume_size = None
                    
        else:
            volume_size = None 
            
        
        if self.foundation_model and not self.huggingface_model:
            # deploy the Model. Note that we need to pass Predictor class when we deploy model through Model class,
            # for being able to run inference through the sagemaker API.

            self.predictor = self.sagemakermodel.deploy(
                initial_instance_count=self.instance_count,
                instance_type=self.instance_type,
                # predictor_cls=sagemaker.predictor.Predictor, # Have to remove this since the new JumpstartModel SDK fails
                endpoint_name="ezsm-foundation-endpoint-" + self.name,
                volume_size=volume_size,
                # serverless_inference_config=self.serverless_config, #ignoring serverless inference
                wait=self.wait
            )
        elif self.foundation_model and self.huggingface_model:
            
            
            self.predictor = self.sagemakermodel.deploy(
                initial_instance_count=self.instance_count,
                instance_type=self.instance_type,
                endpoint_name="ezsm-hf-endpoint-" + self.name,
                volume_size=volume_size,
                wait=self.wait,
                container_startup_health_check_timeout=300,
            )
            
        elif self.huggingface_model and not self.foundation_model:
            
            
            self.predictor = self.sagemakermodel.deploy(
                initial_instance_count=self.instance_count,
                instance_type=self.instance_type,
                endpoint_name="ezsm-hf-endpoint-" + self.name,
                volume_size=volume_size,
                serverless_inference_config=self.serverless_config,
                wait=self.wait
            )
                
            
        else:
            self.predictor = self.sagemakermodel.deploy(
                initial_instance_count=self.instance_count,
                instance_type=self.instance_type,
                accelerator_type=self.ei,
                endpoint_name="ezsm-endpoint-" + self.name,
                update_endpoint=False,
                wait=self.wait,
                volume_size=volume_size,
                data_capture_config=data_capture_config,
                serverless_inference_config=self.serverless_config,
                container_startup_health_check_timeout=300,
            )

        self.endpoint_name = self.predictor.endpoint_name

    def get_size(self, bucket, path):
        s3 = boto3.resource("s3")
        my_bucket = s3.Bucket(bucket)
        total_size = 0.0

        for obj in my_bucket.objects.filter(Prefix=path):
            total_size = total_size + obj.size

        return total_size / ((1024.0) ** 3)

    def upload_model(self):
        i = 1
        if self.prefix == '':
            tmppath = "ezsmdeploy/model-"
        else:
            tmppath = self.prefix + "/ezsmdeploy/model-"
        self.modelpath = []
        for name in self.model:
            self.modelpath.append(
                self.session.upload_data(
                    path="model{}.tar.gz".format(i),
                    bucket=self.bucket,
                    key_prefix=tmppath + self.name,
                )
            )
            i += 1

    def tar_model(self):

        i = 1
        for name in self.model:

            if "tar.gz" in name and 's3' in name:
                # download and uncompress
                self.session.download_data(
                    path="./downloads/{}".format(i),
                    bucket=name.split("/")[2],
                    key_prefix="/".join(name.split("/")[3:]),
                )
                
                with tarfile.open(
                    glob.glob("./downloads/{}/*.tar.gz".format(i))[0]
                ) as tar:
                    tar.extractall("./extractedmodel/{}/".format(i))

                name = "extractedmodel/{}/".format(i)
                
            elif 'tar.gz' in name and 's3' not in name:
                
                self.makedir_safe("./downloads/{}/".format(i))
                shutil.copy(name, "./downloads/{}/".format(i))
                
                with tarfile.open(
                    glob.glob("./downloads/{}/*.tar.gz".format(i))[0]
                ) as tar:
                    tar.extractall("./extractedmodel/{}/".format(i))

                name = "extractedmodel/{}/".format(i)

            tar = tarfile.open("model{}.tar.gz".format(i), "w:gz")
            if "/" in name:
                tar.add(name, arcname=".")
            else:
                tar.add(name)
            tar.close()
            i += 1

    def makedir_safe(self, directory):

        try:
            shutil.rmtree(directory)
        except:
            pass

        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError as err:
            if err.errno != 17:
                print(err.errno)
                raise

    def handle_requirements(self):
        # ------- requirements checks -------
        self.makedir_safe("src")

        if type(self.requirements) == str:
            if os.path.exists(self.requirements):
                # move file to src

                shutil.copy(self.requirements, "src/requirements.txt")

            else:
                raise (self.requirements + " does not exist!")

        elif type(self.requirements) == list:
            f = open("src/requirements.txt", "w")
            l1 = map(lambda x: x + "\n", self.requirements)
            f.writelines(l1)
            f.close()

        else:
            raise ValueError(
                "pass in a path/to/requirements.txt or a list of requirements ['scikit-learn',...,...]"
            )



    def build_docker(self):
        cmd = "chmod +x src/build-docker.sh  & sudo ./src/build-docker.sh {}"
        
        with open('src/dockeroutput.txt', 'w') as f:
            #print("Start process")
            p = subprocess.Popen(cmd.format(self.name), stdout=f, shell=True)
        
        #print("process running in background")
        
        acct = (
            os.popen("aws sts get-caller-identity --query Account --output text")
            .read()
            .split("\n")[0]
        )
        region = os.popen("aws configure get region").read().split("\n")[0]
        self.image = "{}.dkr.ecr.{}.amazonaws.com/ezsmdeploy-image-{}".format(
            acct, region, self.name
        )

        while not os.path.exists("src/done.txt"):
            time.sleep(3)
        
        self.dockeroutput = "Please see src/dockeroutput.txt" 

    def autoscale_endpoint(self):
        response = boto3.client("sagemaker").describe_endpoint(
            EndpointName=self.endpoint_name
        )

        in1 = response["EndpointName"]
        in2 = response["ProductionVariants"][0]["VariantName"]

        client = boto3.client("application-autoscaling")
        response = client.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId="endpoint/{}/variant/{}".format(in1, in2),
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=1,
            MaxCapacity=10,
        )

        response = client.put_scaling_policy(
            PolicyName="scaling-policy-{}".format(self.name),
            ServiceNamespace="sagemaker",
            ResourceId="endpoint/{}/variant/{}".format(in1, in2),
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": self.autoscaletarget,
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
                },
                "ScaleOutCooldown": 600,
                "ScaleInCooldown": 600,
                "DisableScaleIn": False,
            },
        )

        self.scalingresponse = response

    def test(
        self, input_data, target_model=None, usercount=10, hatchrate=5, timeoutsecs=5
    ):

        if self.multimodel and target_model == None:
            raise ValueError(
                "since this is a multimodel endpoint, please pass in a target model that you wish to test"
            )

        if self.deployed:

            path1 = pkg_resources.resource_filename("ezsmdeploy", "data/smlocust.py")
            shutil.copy(path1, "src/smlocust.py")

            start = datetime.datetime.now()

            with yaspin(Spinners.point, color="green", text="") as sp:

                sp.hide()
                sp.write(
                    str(datetime.datetime.now() - start)
                    + " | Starting test with Locust"
                )
                sp.show()

                if self.multimodel:
                    with open("src/locustdata.txt", "w") as outfile:
                        json.dump(
                            {
                                "endpoint_name": self.endpoint_name,
                                "target_model": "model1.tar.gz",
                            },
                            outfile,
                        )
                else:
                    with open("src/locustdata.txt", "w") as outfile:
                        json.dump(
                            {"endpoint_name": self.endpoint_name, "target_model": ""},
                            outfile,
                        )

                pickle.dump(input_data, open("src/testdata.p", "wb"))

                cmd = "locust -f src/smlocust.py --no-web -c {} -r {} --run-time {}s --csv=src/locuststats; touch src/testdone.txt".format(
                    usercount, hatchrate, timeoutsecs
                )
                p = os.system(cmd)
                while not os.path.exists("src/testdone.txt"):
                    time.sleep(3)

                os.remove("src/testdone.txt")

                sp.hide()
                sp.write(
                    str(datetime.datetime.now() - start)
                    + " | Done! Please see the src folder for locuststats* files"
                )
                sp.show()

        else:
            raise ValueError("Deploy model to endpoint first before testing")

    def deploy(self):
        # print(self.__dict__)
        start = datetime.datetime.now()

        with yaspin(Spinners.point, color="green", text="") as sp:

            if not (self.foundation_model or self.huggingface_model) :
                try:
                    shutil.rmtree("src/")
                except:
                    pass

                # compress model files
                self.tar_model()
                sp.hide()
                if self.model == ["tmpmodel"]:
                    sp.write(
                        str(datetime.datetime.now() - start)
                        + " | No model was passed. Assuming you are downloading a model in the script or in the container"
                    )
                else:
                    sp.write(
                        str(datetime.datetime.now() - start) + " | compressed model(s)"
                    )
                sp.show()

                # upload model file(s)
                self.upload_model()

                # Process instance type
                self.process_instance_type()
                sp.hide()
                sp.write(
                    str(datetime.datetime.now() - start)
                    + " | uploaded model tarball(s) ; check returned modelpath"
                )
                sp.show()

                #                 if self.gpu and self.image == None:
                #                     raise ValueError("The default container image used here is based on the multi-model server which does not support GPU instances. Please provide a docker image (ECR repository link) to proceed with model build and deployment.")

                # else:
                # handle requirements
                if self.requirements == None:
                    rtext = (
                        str(datetime.datetime.now() - start)
                        + " | no additional requirements found"
                    )
                    self.makedir_safe("src")
                else:
                    self.handle_requirements()
                    rtext = (
                        str(datetime.datetime.now() - start) + " | added requirements file"
                    )
                sp.hide()
                sp.write(rtext)
                sp.show()

                # move script to src
                shutil.copy(self.script, "src/transformscript.py")
                sp.hide()
                sp.write(str(datetime.datetime.now() - start) + " | added source file")
                sp.show()

                # ------ Dockerfile checks -------
                if self.dockerfilepath == None and self.multimodel == True:
                    self.dockerfilepath = pkg_resources.resource_filename(
                        "ezsmdeploy", "data/Dockerfile"
                    )
                elif self.dockerfilepath == None and self.multimodel == False:
                    self.dockerfilepath = pkg_resources.resource_filename(
                        "ezsmdeploy", "data/Dockerfile_flask"
                    )

                # move Dockerfile to src
                shutil.copy(self.dockerfilepath, "src/Dockerfile")
                sp.hide()
                sp.write(str(datetime.datetime.now() - start) + " | added Dockerfile")
                sp.show()

                # move model_handler and build scripts to src

                if self.multimodel:
                    # Use multi model
                    path1 = pkg_resources.resource_filename(
                        "ezsmdeploy", "data/model_handler.py"
                    )
                    path2 = pkg_resources.resource_filename(
                        "ezsmdeploy", "data/dockerd-entrypoint.py"
                    )
                    path3 = pkg_resources.resource_filename(
                        "ezsmdeploy", "data/build-docker.sh"
                    )

                    shutil.copy(path1, "src/model_handler.py")
                    shutil.copy(path2, "src/dockerd-entrypoint.py")
                    shutil.copy(path3, "src/build-docker.sh")

                    self.ei = None

                else:
                    # Use Flask stack
                    path1 = pkg_resources.resource_filename("ezsmdeploy", "data/nginx.conf")
                    path2 = pkg_resources.resource_filename(
                        "ezsmdeploy", "data/predictor.py"
                    )
                    path3 = pkg_resources.resource_filename("ezsmdeploy", "data/serve")
                    path4 = pkg_resources.resource_filename("ezsmdeploy", "data/train")
                    path5 = pkg_resources.resource_filename("ezsmdeploy", "data/wsgi.py")
                    path6 = pkg_resources.resource_filename(
                        "ezsmdeploy", "data/build-docker.sh"
                    )

                    shutil.copy(path1, "src/nginx.conf")
                    shutil.copy(path2, "src/predictor.py")
                    shutil.copy(path3, "src/serve")
                    shutil.copy(path4, "src/train")
                    shutil.copy(path5, "src/wsgi.py")
                    shutil.copy(path6, "src/build-docker.sh")

                    if self.gpu and self.ei != None:
                        self.ei = None
                        sp.hide()
                        sp.write(
                            str(datetime.datetime.now() - start)
                            + " | Setting Elastic Inference \
                        to None since you selected a GPU instance"
                        )
                        sp.show()

                sp.hide()
                sp.write(
                    str(datetime.datetime.now() - start)
                    + " | added model_handler and docker utils"
                )
                sp.show()

                # build docker container
                if self.image == None:
                    sp.write(
                        str(datetime.datetime.now() - start)
                        + " | building docker container"
                    )
                    self.build_docker()
                    sp.hide()
                    sp.write(
                        str(datetime.datetime.now() - start) + " | built docker container"
                    )
                    sp.show()

                # create sagemaker model
                self.create_model()
            else:
                if self.foundation_model and not self.huggingface_model:
                    self.deploy_foundation_model()
                elif self.huggingface_model:
                    self.deploy_huggingface_model()
                else:
                    raise ValueError("Did not find model artifact, or foundation/huggingface model")
                
            sp.hide()
            
            if not self.serverless:
                sp.write(
                    str(datetime.datetime.now() - start)
                    + " | created model(s). Now deploying on "
                    + self.instance_type
                )
            else:
                sp.write(
                    str(datetime.datetime.now() - start)
                    + " | created model(s). Now deploying on Serverless!"
                )
            sp.show()

            # deploy model
            self.deploy_model()
            sp.hide()
            sp.write(str(datetime.datetime.now() - start) + " | deployed model")
            sp.show()

            if self.autoscale and self.instance_type not in ["local", "local_gpu"]:
                self.autoscale_endpoint()
                sp.hide()
                sp.write(str(datetime.datetime.now() - start) + " | set up autoscaling")
                sp.show()
            elif self.autoscale and self.instance_type in ["local", "local_gpu"]:
                sp.hide()
                sp.write(
                    str(datetime.datetime.now() - start)
                    + " | not setting up autoscaling; deploying locally"
                )
                sp.show()

            # if self.instance_type not in ["local", "local_gpu"]:
            #     sp.hide()
            #     sp.write(
            #         str(datetime.datetime.now() - start)
            #         + " | estimated cost is $"
            #         + str(self.costperhour)
            #         + " per hour"
            #     )
            #     sp.show()

            if self.monitor:
                sp.hide()
                sp.write(
                    str(datetime.datetime.now() - start)
                    + " | model monitor data capture location is "
                    + "s3://{}/ezsmdeploy/model-{}/datacapture".format(
                        self.bucket, self.name
                    )
                )
                sp.show()

            # finalize
            sp.green.ok(str(datetime.datetime.now() - start) + " | " "Done! ✔")

            self.deployed = True

            try:
                # Cleanup
                os.remove("src/done.txt")
                os.remove("src")
                os.remove("downloads")
                os.remove("extractedmodel")
                os.remove("tmpmodel")
            except:
                pass

            return self.predictor
        
    def chat(self):
        # if self.foundation_model and 'chat' in self.model or 'Chat' in self.model :
        OpenChatKitShell(
            predictor=self.predictor,
            model_name = self.model,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.6,
            top_k=40
        ).cmdloop()
        
        # else:
        #     print("Sorry, you can only use the chat functionality with gpt-neoxt-chat-base-20b or the RedPajama chat models for now")



MEANINGLESS_WORDS = ['<pad>', '</s>', '<|endoftext|>']
PRE_PROMPT = """\
Current Date: {}
Current Time: {}

"""

def clean_response(response):
    for word in MEANINGLESS_WORDS:
        response = response.replace(word, "")
    response = response.strip("\n")
    return response

class Conversation:
    def __init__(self, human_id, bot_id):
        cur_date = time.strftime('%Y-%m-%d')
        cur_time = time.strftime('%H:%M:%S %p %Z')

        self._human_id = human_id
        self._bot_id = bot_id
        self._prompt = PRE_PROMPT.format(cur_date, cur_time)

    def push_context_turn(self, context):
        # for now, context is represented as a human turn
        self._prompt += f"{self._human_id}: {context}\n"

    def push_human_turn(self, query):
        self._prompt += f"{self._human_id}: {query}\n"
        self._prompt += f"{self._bot_id}:"

    def push_model_response(self, response):
        has_finished = self._human_id in response
        bot_turn = response.split(f"{self._human_id}:")[0]
        bot_turn = clean_response(bot_turn)
        # if it is truncated, then append "..." to the end of the response
        if not has_finished:
            bot_turn += "..."

        self._prompt += f"{bot_turn}\n"

    def get_last_turn(self):
        human_tag = f"{self._human_id}:"
        bot_tag = f"{self._bot_id}:"
        turns = re.split(f"({human_tag}|{bot_tag})\W?", self._prompt)
        return turns[-1]

    def get_raw_prompt(self):
        return self._prompt

    @classmethod
    def from_raw_prompt(cls, value):
        self._prompt = value



class OpenChatKitShell(cmd.Cmd):
    intro = (
        "EzSMdeploy Openchatkit shell -  Type /help or /? to "
        "list commands. For example, type /quit to exit shell.\n"
    )
    prompt = ">>> "
    human_id = "<human>"
    bot_id = "<bot>"

    def __init__(self, predictor: Predictor, model_name: str, cmd_queue: Optional[List[str]] = None, **kwargs):
        super().__init__()
        self.predictor = predictor
        self.model = model_name
        self.payload_kwargs = kwargs
        self.payload_kwargs["stopping_criteria"] = [self.human_id]
        if cmd_queue is not None:
            self.cmdqueue = cmd_queue

    def preloop(self):
        self.conversation = Conversation(self.human_id, self.bot_id)

    def precmd(self, line):
        command = line[1:] if line.startswith("/") else "say " + line
        return command

    def do_say(self, arg):
        self.conversation.push_human_turn(arg)
        prompt = self.conversation.get_raw_prompt()
        if 'neoxt-chat' in self.model:
            # For neoxt - chatbase - 20B
            payload = {"text_inputs": prompt, **self.payload_kwargs}
            response = self.predictor.predict(payload)
            output = response[0][0]["generated_text"][len(prompt) :]
        elif '-Chat' in self.model or 'chat' in self.model: #for RedPajama chat models, experimental for other chat models like openchat/openchat
            
            payload = {"inputs":prompt,
           "parameters":{"max_new_tokens":100}}
            
            
            response = self.predictor.predict(payload) 
            last = response[0]['generated_text'].rfind("\n<human>") #returns -1 if the human token hasn't been found yet, which works
            output = response[0]["generated_text"][len(prompt) :last]
            
        else:
            output = "I don't recognize the output from this chat model"
            
        self.conversation.push_model_response(output)
        print(self.conversation.get_last_turn())

    def do_reset(self, arg):
        self.conversation = Conversation(self.human_id, self.bot_id)

    def do_hyperparameters(self, arg):
        print(f"Hyperparameters: {self.payload_kwargs}\n")

    def do_quit(self, arg):
        return True