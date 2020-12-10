from setuptools import setup

setup(
    name = "transfer",
    author = "Stefano Rando",
    url = "git@github.com/Steefano/transfer",
    author_email = "ste.rando.97@gmail.com",
    packages = setuptools.find_packages(),
    install_requires = ["numpy", "scikit-learn", "cvxpy"],
    version = "0.0.1"
    license = "MIT",
    description = "A package containing methods and tools for transfer learning."
)
