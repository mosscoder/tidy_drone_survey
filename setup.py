from setuptools import setup, find_packages

setup(
    name='tidysurvey',
    version='0.2.0.dev0',
    description='Config-driven drone-survey pipeline: stitch, align, calibrate, report',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'rasterio',
        'numpy',
        'scipy',
        'geopandas',
        'shapely',
        'affine',
        'joblib',
        'opencv-python',
        'tomli; python_version < "3.11"',   # TOML parser (stdlib tomllib from 3.11)
    ],
    extras_require={
        # LoFTR matching (align + seam walk) and the Sentinel-2 reference
        'match': ['torch', 'kornia'],
        'sentinel': ['earthengine-api', 'google-auth', 'requests', 'pyproj'],
        'report': ['matplotlib'],
        'tiles': ['pmtiles', 'Pillow'],
    },
    entry_points={
        'console_scripts': [
            'tidysurvey = tidysurvey.cli:main',
        ],
    },
    test_suite='tests',
)
