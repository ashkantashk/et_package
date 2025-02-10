from setuptools import setup, find_packages

setup(
    name='et_package',
    version='0.1',
    description='A package for ET_Func with gaze data processing',
    author='ashta',
    packages=['et_package'],
    install_requires=[
        'pandas',
        'numpy==1.26.4',
        'I2MC',
        'matplotlib',
        'scipy',
        'bokeh',
        'datetime'        
    ],
)
