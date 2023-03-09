# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:22.04

RUN apt -y update && apt -y upgrade && \
    apt-get -y install curl && \
    curl -sL https://deb.nodesource.com/setup_18.x | bash - && \
    apt install nodejs -y && \
    npm install -g @bazel/bazelisk

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
    gcc \
    libpq-dev \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/* 

RUN pip3 install --upgrade pip
RUN python3 -V
# Here we get all python packages.
RUN pip3 install wheel

RUN pip3 --no-cache-dir install setuptools \
                                numpy \
                                pandas \
                                flask gevent gunicorn \
                                mxnet \
                                multi-model-server \
                                sagemaker-inference \
                                retrying

COPY requirements.txt /usr/local/bin/requirements.txt

RUN pip3 install -r /usr/local/bin/requirements.txt

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

ADD . /opt/program/

RUN ls /opt/program/

RUN chmod +x /opt/program/train

RUN chmod +x /opt/program/serve

WORKDIR /opt/program
