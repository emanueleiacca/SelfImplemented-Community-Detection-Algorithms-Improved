from setuptools import setup, find_packages

setup(
    name='SelfImplemented-Community-Detection-Algorithms-Improved',
    version='0.1.0',
    packages=find_packages(where='functions'),
    package_dir={'': 'functions'},
    install_requires=[
        'heapq',
        'collections',
        'matplotlib',
        'numpy',
        'copy',
        'networkx'
    ],
    description='A custom library for community detection algorithms with optimized performance and flexibility',
    author='Emanuele Iaccarino',
    author_email='emanueleiaccarino.ei@gmail.com',
    url='https://github.com/your-username/your-library',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
