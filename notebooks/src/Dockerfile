# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3-dev \
    nginx \
    ca-certificates \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py
    

# Here we get all python packages.

RUN pip3 --no-cache-dir install numpy \
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

# Set up the program in the image
#COPY transformscript.py /opt/program/transformscript.py

#COPY nginx.conf /opt/program/nginx.conf

#COPY predictor.py /opt/program/predictor.py

#COPY serve /opt/program/serve

#COPY train /opt/program/train

#COPY wsgi.py /opt/program/wsgi.py

RUN chmod +x /opt/program/train

RUN chmod +x /opt/program/serve

WORKDIR /opt/program

