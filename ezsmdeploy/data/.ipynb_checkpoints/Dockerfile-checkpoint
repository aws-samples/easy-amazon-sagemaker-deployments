FROM ubuntu:22.04

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

RUN apt -y update && apt -y upgrade && \
    apt-get -y install curl && \
    curl -sL https://deb.nodesource.com/setup_18.x | bash - && \
    apt install nodejs -y && \
    npm install -g @bazel/bazelisk
    
# Install necessary dependencies for MMS and SageMaker Inference Toolkit

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3 \
    python3-pip \
    python3-setuptools \
    nginx \
    ca-certificates \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/* 

RUN python3 -V
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

RUN pip3 install --upgrade pip

# Install MXNet, MMS, and SageMaker Inference Toolkit to set up MMS
RUN pip3 --no-cache-dir install mxnet \
                                multi-model-server \
                                sagemaker-inference \
                                retrying

ADD . /usr/local/bin/

RUN ls /usr/local/bin/

#COPY requirements.txt /usr/local/bin/requirements.txt

RUN pip3 install -r /usr/local/bin/requirements.txt

# Copy entrypoint script to the image
COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
COPY model_handler.py /home/model-server/model_handler.py

COPY transformscript.py /home/model-server/transformscript.py

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]
