from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras', 'h5py', 'tensorflow', 'scikit-learn', 'pandas', 'numpy', 'librosa']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)
