from distutils.core import setup

from setuptools import find_packages

setup(name='ARLO',
      packages=find_packages('.'),
      version='0.0.1',
      license='MIT',      
      description='ARLO: Automated RL Optimizer.',
      long_description='ARLO is a Python library for automating all the stages making up an Automated RL pipeline.',
      author='arloreinforcement',
      author_email='arloreinforcement@gmail.com',
      url='https://arlo-lib.github.io/arlo-lib/',
      install_requires=['catboost', 'cloudpickle', 'gym', 'joblib', 'matplotlib', 'mushroom_rl', 'numpy', 'optuna',
                        'plotly', 'scikit_learn', 'scipy', 'torch', 'xgboost'],
      classifiers=['Programming Language :: Python :: 3',
                   'License :: MIT License',
                   'Operating System :: OS Independent'
                   ]
      )