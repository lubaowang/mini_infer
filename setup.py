from setuptools import setup, find_packages

setup(
    name="mini_infer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1",
        "triton>=2.2",
    ],
    python_requires=">=3.10",
)
