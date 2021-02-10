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
from sagemaker.model import Model
import ast
import csv
import json
import pickle


# from .model_handler import * # this works but make sure you have all the packages mentioned in the model_handler import; i.e., make this generic


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

        self.frameworklist = ["tensorflow", "pytorch", "mxnet", "sklearn"]
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
        self.monitor = monitor
        self.deployed = False
        self.autoscaletarget = autoscaletarget

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

        # ------- Script checks ---------
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
        elif requirements == None and framework not in self.frameworklist:
            raise ValueError(
                "If requirements=None, please provide a value for framework; \
                    choice should be one of 'tensorflow','pytorch','mxnet','sklearn'"
            )

        self.autoscale = autoscale

        self.wait = wait

        self.deploy()

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
        size = self.get_size(self.bucket, "ezsmdeploy/model-" + self.name + "/")

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

            data_capture_config = DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=100,
                destination_s3_uri="s3://{}/ezsmdeploy/model-{}/datacapture".format(
                    self.bucket, self.name
                ),
            )
        else:
            data_capture_config = None

        self.predictor = self.sagemakermodel.deploy(
            initial_instance_count=self.instance_count,
            instance_type=self.instance_type,
            accelerator_type=self.ei,
            endpoint_name="ezsmdeploy-endpoint-" + self.name,
            update_endpoint=False,
            wait=self.wait,
            data_capture_config=data_capture_config,
        )

        self.endpoint_name = "ezsmdeploy-endpoint-" + self.name

    def get_size(self, bucket, path):
        s3 = boto3.resource("s3")
        my_bucket = s3.Bucket(bucket)
        total_size = 0.0

        for obj in my_bucket.objects.filter(Prefix=path):
            total_size = total_size + obj.size

        return total_size / ((1024.0) ** 3)

    def upload_model(self):
        i = 1
        self.modelpath = []
        for name in self.model:
            self.modelpath.append(
                self.session.upload_data(
                    path="model{}.tar.gz".format(i),
                    bucket=self.bucket,
                    key_prefix="ezsmdeploy/model-" + self.name,
                )
            )
            i += 1

    def tar_model(self):

        i = 1
        for name in self.model:

            if "s3" in name:
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
            sp.hide()
            sp.write(
                str(datetime.datetime.now() - start)
                + " | created model(s). Now deploying on "
                + self.instance_type
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

            if self.instance_type not in ["local", "local_gpu"]:
                sp.hide()
                sp.write(
                    str(datetime.datetime.now() - start)
                    + " | estimated cost is $"
                    + str(self.costperhour)
                    + " per hour"
                )
                sp.show()

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
