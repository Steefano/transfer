import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name = "transfer",
    author = "Stefano Rando",
    url = "git@github.com/Steefano/transfer",
    author_email = "ste.rando.97@gmail.com",
    packages = find_packages(),
    install_requires = ["numpy", "scikit-learn", "cvxpy"],
    version = "0.0.1",
    license = "MIT",
    description = "A package containing methods and tools for transfer learning.",
    long_description = README,
    long_description_content_type = "text/markdown",
    classifiers = [
        "Programming language :: Python :: 3"
    ]
)
