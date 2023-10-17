import setuptools


with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="simsad",  # Replace with your own username
    version="0.1.0",
    author="Equipe CJP",
    author_email="pierre-carl.michaud@hec.ca",
    description="Modele de projection des soins a domicile du Quebec",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://rsi-models.github.io/SimSAD/",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
   'pandas',
   'numba',
   'numpy',
   'xlrd',
   'srpp'
    ],
    python_requires='>=3.9',
)