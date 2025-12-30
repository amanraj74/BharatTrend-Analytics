"""
BharatTrend - AI-Powered Market Trend Analysis System
Setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="bharattrend",
    version="1.0.0",
    author="Anand CT",
    author_email="anandct.contact@gmail.com",
    description="AI-Powered E-commerce Market Trend Analysis System for Indian Markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ANANDCT05/BharatTrend_Analysis",
    project_urls={
        "Bug Tracker": "https://github.com/ANANDCT05/BharatTrend_Analysis/issues",
        "Documentation": "https://github.com/ANANDCT05/BharatTrend_Analysis/blob/main/README.md",
        "Source Code": "https://github.com/ANANDCT05/BharatTrend_Analysis",
    },
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
extras_require={
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "jupyter>=1.0.0",
    ],
    "api": [
        "fastapi>=0.108.0",
        "uvicorn>=0.25.0",
    ],
    "dashboard": [
        "streamlit>=1.29.0",
    ],
},
entry_points={
    "console_scripts": [
        # Fixed entry points - must be in format: command=module:function
        "bharattrend-process=src.data_processing:main",
        "bharattrend-train=src.ml_models:main",
        "bharattrend-dashboard=src.run_dashboard:main",  # Now this will work!'
    ],
},
    include_package_data=True,
    package_data={
        "bharattrend": [
            "data/external/.gitkeep",
            "data/processed/.gitkeep",
            "data/visualizations/.gitkeep",
            "configs/*.yaml",
            "configs/*.py",
        ],
    },
    keywords=[
        "machine learning",
        "data science",
        "market analysis",
        "trend prediction",
        "e-commerce",
        "india",
        "price optimization",
        "artificial intelligence",
        "sklearn",
        "pandas",
    ],
    zip_safe=False,
)
