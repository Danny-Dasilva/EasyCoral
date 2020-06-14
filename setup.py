import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="Easy_Coral",
    version="0.0.10",
    author="James Warszycki, Devin Willis",
    author_email="jwarszycki2017@fau.edu",
    description="Google Coral AI and Camera Manager",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
)