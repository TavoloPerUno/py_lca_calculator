import setuptools
import pathlib

HERE = pathlib.Path(__file__).parent
INSTALL_REQUIRES = (HERE / "requirements.txt").read_text().splitlines()

__version__ = "0.0.0"

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='py_lca_calculator',
    version=__version__,
    author='Research Computing Group',
    description='LCA Class Membership & Outcome Probability Calculator',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    packages=setuptools.find_packages(),
    include_package_data=True,
    license_file='LICENSE',
    install_requires=[
        'confuse',
        'streamlit',
        'click',
        'pandas',
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_data={'py_lca_calculator': ['config_default.yaml']}
)
