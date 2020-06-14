import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="Easy_Coral",
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="Google Coral AI and Camera Manager",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ReefVision/EasyCoral",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)