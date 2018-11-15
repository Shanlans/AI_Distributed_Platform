'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'numpy',
    'Pillow',
    'requests==2.19.1',
    'uuid',
    'pandas',
    'tensorflow-gpu==1.11.0'
]

setup(name='AI-DISTRIBUTED-PLATFORM',
      version='0.1',
      packages=find_packages(),
      include_package_data=True,
      description='Train model on Cloud ML Engine',
      author='SHEN SHANLAN',
      author_email='shenshanlan@gmail.com   ',
      license='MIT',
      install_requires=REQUIRED_PACKAGES,
      zip_safe=False)   