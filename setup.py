import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# parse_requirements() returns generator of pip.req.InstallRequirement objects

if __name__ == "__main__":
    setuptools.setup(
        name="logic_crf",  # Replace with your own username
        version="0.0.1",
        license='BSD 3-Clause',
        author="Perceval Wajsburt",
        author_email="perceval.wajsburt@sorbonne-universite.fr",
        description="Logic Conditional Random Field",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/perceval/logic_crf",
        packages=setuptools.find_packages(),
        package_data={},
        install_requires=[
            'opt_einsum>=3.1.0',
            'numpy>=1.17.4',
            'pandas>=0.24.1',
            'pyeda>=0.28.0',
            'torch>=1.3.1',
        ],
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
    )
