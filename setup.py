import os
from glob import glob
import sys

from setuptools import setup, find_packages


def read(fname):
    """
    Args:
        fname:
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

extras = {
    'locust': [
        'locustio==0.14.5'
    ],
    'transfomers': [
        'transformers==4.49.0',
        'peft==0.14.0'
    ]
}

setup(name='ezsmdeploy',
      version='2.23.5',
      description='Amazon SageMaker and Bedrock custom model deployments made easy',
      url='https://pypi.python.org/pypi/ezsmdeploy',
      #scripts=['Dockerfile','dockerd-entrypoint.py','model_handler.py','build-docker.sh'],
      author='Shreyas Subramanian',
      author_email='subshrey@amazon.com',
      license='MIT',
      packages=['ezsmdeploy'],
      package_data={'ezsmdeploy': ['data/*']},
      include_package_data=True,
      extras_require = extras,
      install_requires=["sagemaker==2.238.0","yaspin==0.16.0","shortuuid==1.0.1","sagemaker-studio-image-build==0.5.0", "boto3>=1.14.12", "huggingface_hub==0.28.1"],
      zip_safe=False,
      classifiers=['Development Status :: 5 - Production/Stable',
                   "Intended Audience :: Developers",
                   "Natural Language :: English",
                   "License :: OSI Approved :: Apache Software License",
                   "Programming Language :: Python"],
      long_description=read("README.rst")
     )
