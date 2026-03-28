from setuptools import find_packages, setup

setup(
    name="faseroh-prototype",
    version="0.1.0",
    description="FASEROH prototype for symbolic seq2seq learning",
    author="GSoC Applicant",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "numpy",
        "sympy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "tensorboard",
        "pytest",
    ],
    python_requires=">=3.9",
)
