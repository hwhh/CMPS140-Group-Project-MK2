from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['Keras==2.1.6', 'h5py==2.7.1', 'tensorflow==1.7.0', 'scikit-learn==0.19.1', 'pandas==0.22.0', 'numpy==1.14.3', 'librosa==0.6.0',
                     'soundfile']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)
