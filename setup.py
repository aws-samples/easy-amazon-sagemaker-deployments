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
    ]
}

setup(name='ezsmdeploy',
      version='0.2.6',
      description='SageMaker custom deployments made easy',
      url='https://pypi.python.org/pypi/ezsmdeploy',
      #scripts=['Dockerfile','dockerd-entrypoint.py','model_handler.py','build-docker.sh'],
      author='Shreyas Subramanian',
      author_email='subshrey@amazon.com',
      license='MIT',
      packages=['ezsmdeploy'],
      package_data={'ezsmdeploy': ['data/*']},
      extras_require = extras,
      install_requires=["sagemaker>=1.55.3","yaspin==0.16.0","shortuuid==1.0.1"],
      zip_safe=False,
      classifiers=['Development Status :: 3 - Alpha',
                   "Intended Audience :: Developers",
                   "Natural Language :: English",
                   "License :: OSI Approved :: Apache Software License",
                   "Programming Language :: Python"],
      long_description=read("README.rst")
     )