from setuptools import find_packages
from setuptools import setup
import os
from pathlib import Path

# Packages that are required for this module to be executed
#here = os.path.abspath(os.path.dirname(__file__))
#lib_path = Path.cwd().resolve() / 'requirements.txt'

# def list_reqs(fname="f.txt"):
#     with open(fname) as fd:
#         return fd.read().splitlines()

# with open("requirements.txt", "r") as f:
#     rq = f.read().splitlines()

# REQUIRED_PACKAGES = ['pandas<=1.3.5','tensorflow<=2.6.2','tensorflow-io<=0.23.1','tensorflow-addons<=0.15.0']

REQUIRED_PACKAGES = ['pandas>=1.1.5,<1.2.0','tensorflow>=2.6.2,<2.7.0','tensorflow-addons>=0.14.0,<0.15.0','tensorflow-io>=0.21.0,<0.22.0']

#print("*-*-*-*-*-*-*-*-HERE*-*-*-*-*-*-*---*")
#here = os.path.abspath(os.path.dirname(__file__))
#print(lib_path)
#REQUIRED_PACKAGES = ['some_PyPI_package>=1.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
