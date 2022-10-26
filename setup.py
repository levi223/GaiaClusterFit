from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Matching gaia clustered stars to known clusters'
LONG_DESCRIPTION = 'A package that allows to build simple streams of video, audio and camera data.'

# Setting up
setup(
    name="GaiaClusterFit",
    version=VERSION,
    author="Levi van Es",
    author_email="<levi2234@hotmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy>=1.23.3', "more-itertools>=9.0.0","hdbscan==0.8.28",'astropy>=5.1.1',"scipy>=1.9.2", 'joblib==1.1.0',"sklearn>=0.0"],
    keywords=[],
    classifiers=[

    ]
)