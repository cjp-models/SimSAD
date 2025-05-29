import setuptools


with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="simsad-cjp",  # Replace with your own username
    version="1.1.0",
    author="Equipe CJP",
    author_email="pierre-carl.michaud@hec.ca",
    description="Modele de projection du soutien à l'autonomie du Quebec",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://cjp-models.github.io/SimSAD/",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
   'pandas',
   'numba>=0.58',
   'numpy>=1.24',
   'xlrd',
   'xlsxwriter',
    ],
    python_requires='>=3.9',
)