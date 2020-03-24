import setuptools

def readme():
    with open("README.md", "r") as fh:
        return fh.read()

setuptools.setup(
    name="fcmbci",
    version="0.0.1",
    author="Samvel Mkhitaryan",
    author_email="mkhitarian.samvel@gmail.com",
    description="Fuzzy Cognitive Maps for Behavior Change Interventions and Evaluation",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SamvelMK/FcmBci.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU AFFERO GENERAL PUBLIC LICENSE V3 OR LATER (AGPLV3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)