from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="carladriver",
    version="2.0.0",
    description="Library for controlling ego vehicle using carla simulator python API",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    include_package_data=True, 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/out0/carladriver",
    author="Cristiano Oliveira",
    author_email="cristianoo@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    #install_requires=["numpy>=1.21.0", "carla>=0.9.15"],
    install_requires=["numpy>=1.21.0"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)
