import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='pandas_zookeeper',
    version='0.0.1',
    author='Shawn Smith',
    author_email='ssmith161803@gmail.com',
    description='Automate common data cleaning tasks for Pandas Dataframes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/integral-ssmith/pandas_zookeeper',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
