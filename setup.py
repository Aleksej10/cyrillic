import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cyr", 
    version="0.0.1",
    author="Aleksej",
    author_email="mr.strale@gmain.com",
    description="smart ascii to unicode cyrillic converter",
    keywords="cyrillic converter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aleksej10/cyrillic",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "torch",
    ],
    data_files=[
        ("share/man/man1", [
            "doc/cyr.1",
        ])
    ],
    entry_points={
        "console_scripts": [
            "cyr=cyr.cyr:main",
        ]
    },
)
