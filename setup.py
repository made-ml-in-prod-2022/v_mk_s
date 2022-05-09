from setuptools import find_packages, setup


setup(
    name="ml_project",
    packages=find_packages(),
    version="1.0.0",
    description="Production-ready ML project",
    author="Vladislav Melnichuk",
    install_requires=["scikit-learn==0.24.1", "pandas==1.1.5",], # !! change
    license="MIT",
)
