import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="alectio_sdk",
    version="0.0.1",
    author="Hongshan Li",
    author_email="hongshan.li@alectio.com",
    description="Integrate customer side ML application with Alectio Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Anything in favor of us",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
)

