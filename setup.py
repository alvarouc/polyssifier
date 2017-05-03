from setuptools import setup
setup(
    name='polyssifier',
    packages=['polyssifier'],
    version='0.2.2',
    install_requires=[
        'mlp',
        'pandas',
        'sklearn',
        'numpy'],
    description='Data exploration tool for assessing optimal classification methods',
    author='Alvaro Ulloa',
    author_email='alvarouc@gmail.com',
    url='https://github.com/alvarouc/polyssifier',
    download_url='https://github.com/alvarouc/polyssifier/tarball/0.2.2',
    keywords=['classification', 'machine learning', 'data science'],
    classifiers=[],
)
