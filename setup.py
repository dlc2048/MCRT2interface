from setuptools import setup, find_packages

setup(
    name='rt2',
    version='1.5.4',
    packages=find_packages(where="src"),
    package_dir={'': 'src'}
)
