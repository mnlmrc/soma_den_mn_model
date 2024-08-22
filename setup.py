from setuptools import setup, find_packages

from soma_den_mn_model import __version__

def fetch_requirements():
    with open("requirements.txt", "r", encoding="utf-8", errors="ignore") as f:
        reqs = f.read().strip().split("\n")
    return reqs

setup(
    name='soma_den_mn_model',
    version=__version__,
    author='Irene Mendez Guerra',
    author_email='irene.mendez17@imperial.ac.uk',
    description='soma_den_mn_model is a package to simulate motor neurons.',

    url='https://github.com/imendezguerra/soma_den_mn_model',

    packages=find_packages(),
    install_requires=fetch_requirements(),
)