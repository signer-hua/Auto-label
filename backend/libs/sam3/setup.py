# SAM3 推理最小包
from setuptools import setup, find_packages

setup(
    name="sam3",
    version="0.0.1",
    packages=find_packages(),
    package_data={"sam3": ["assets/*.gz"]},
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "timm>=1.0.17",
        "huggingface_hub",
        "tqdm",
        "ftfy==6.1.1",
        "regex",
        "numpy>=1.26,<2",
        "typing_extensions",
    ],
)
