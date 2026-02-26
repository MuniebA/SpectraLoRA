from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spectralora", # This is what people will type: pip install spectralora
    version="0.1.0",
    author="Munieb Abdelrahman",
    author_email="muniebawad@gmail.com",
    description="Physics-Aware Parameter-Efficient Fine-Tuning for Geospatial Foundation Models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MuniebA/SpectraLoRA",
    packages=find_packages(include=['spectra_lora', 'spectra_lora.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "rasterio>=1.3.0",
        "transformers>=4.30.0",
        "huggingface-hub>=0.16.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0"
    ],
)