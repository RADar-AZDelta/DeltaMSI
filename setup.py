#! /usr/bin/env python
from setuptools import setup, find_packages

version = '1.0.0'
dl_version = 'master' if 'dev' in version else '{}'.format(version)

setup(
    name='DeltaMSI',
    version=version,
    author='Koen Swaerts',
    author_email='koen.swaerts@azdelta.be',
    description="DeltaMSI: artificial intelligence-based modeling of microsatellite instability scoring on next-generation sequencing data",
    long_description=__doc__,
    keywords=['bioinformatics', 'biology', 'sequencing', 'NGS', 'next generation sequencing',
              'MSI', 'microsatellite instability'],
    download_url='https://github.com/RADar-AZDelta/DeltaMSI/archive/refs/tags/v{}.tar.gz'.format(
        dl_version),
    license='GNU General Public License v3.0',
    packages=find_packages('.'),
    python_requires='>=3.7',
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'pandas',
        'numpy',
        'pysam',
        'matplotlib',
        'scikit-learn',
        'seaborn',
        'tqdm',
        'openpyxl'
    ],
    entry_points={
        'console_scripts': ['DeltaMSI = deltamsi.app:main']
    }
)