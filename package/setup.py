import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
 
setuptools.setup(
    name="umucv",
    version="0.3",
    license="BSD 3-Clause",
    author="Alberto Ruiz",
    author_email="aruiz@um.es",
    description="computer vision tools",
    url="https://github.com/albertoruiz",
    platforms="any",
    keywords="vision image",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Computer Vision",
        "Programming Language :: Python :: 3",
        "License :: BSD 3-Clause",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown"
)

