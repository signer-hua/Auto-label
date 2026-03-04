# DINOv3 推理最小包
from setuptools import setup, find_packages

setup(
    name="dinov3",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "omegaconf",
        "ftfy",
        "regex",
    ],
)
