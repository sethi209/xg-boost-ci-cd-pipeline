from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['gcsfs>=2021.4.0']

setup(
    name='my_training_package',
    version='0.2',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='MLOps Task'
)