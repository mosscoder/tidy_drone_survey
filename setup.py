from setuptools import setup, find_packages

setup(
    name='tidysurvey',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        'rasterio',
        'numpy',
        'scipy',
        'geopandas',
        'shapely',
        'argparse' # argparse is in standard library for Python 2.7 and 3.2+, but good to list for clarity or older versions
    ],
    test_suite='tests',
) 