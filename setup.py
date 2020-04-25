import setuptools


setup_requires = ['numpy>=1.16.0']
install_requires = setup_requires + ['matplotlib>=3.0.0',
                                     'scipy>=1.0.0']

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="myml-connor-mcisaac",
    version="0.0.1",
    author="Connor McIsaac",
    author_email="connor.mcisaac@outlook.com",
    description="A package of ML and stat tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/connor-mcisaac/myml",
    setup_requires=setup_requires,
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
