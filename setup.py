import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="lstm_ner",
    version="0.0.1",
    author="Jakob Steinbauer",
    author_email="jakob_steinbauer@hotmail.com",
    description="This repo holds lstm-based tensorflow models for the task of named entity recognition ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JSteinbauer/lstm_ner",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.0.1",
)
