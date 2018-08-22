from setuptools import setup, find_packages

setup(name="rlpack",
      version="0.1",
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
          "tensorflow-gpu>=1.7.0",
          "numpy",
          "pyzmq",
          "ray"
      ]
)
