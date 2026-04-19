from setuptools import setup, find_packages

# This file makes our src/ folder importable as a package
# So we can do: from src.logger import logger — from anywhere in the project

setup(
    name="dropout_prediction_and_counseling_system",
    version="1.0.0",
    author="Subhadip Pan",
    author_email="subhadip.pan.24@aot.edu.in",
    description="Student Dropout Prediction and Counseling System",
    packages=find_packages(),
    install_requires=[]   # dependencies are in requirements.txt
)
