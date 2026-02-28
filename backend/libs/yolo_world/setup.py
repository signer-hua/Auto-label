# YOLO-World 推理最小包
from setuptools import setup, find_packages

setup(
    name="yolo_world",
    version="0.4.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "mmengine>=0.10.3",
        "mmdet>=3.0.0",
        "transformers>=4.36.0",
    ],
)
