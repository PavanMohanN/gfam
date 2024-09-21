# Machine Learning Approaches for Efficient FFT Approximation in OFDM Systems: A Data-Driven Analysis

# Can Global Function Approximation Methods Approximate Fast Fourier Transforms?

## Overview of the Project
In this repository, we explore the potential of machine learning models to approximate Fast Fourier Transforms (FFT) using two distinct methodologies. This project investigates how global function approximation techniques, specifically neural networks and the Adaline model, can replace conventional FFT processes in Orthogonal Frequency Division Multiplexing (OFDM) systems. Our goal is to provide a comprehensive analysis of these approaches, including the performance metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE), alongside visual comparisons through various plots.

## Dataset Files
This repository includes two datasets generated from the FFT methods:

1. **`dataset1.csv`**: Contains data generated from the **Standard FFT approach** (FFT1), created using the `FFT_1_method.mlx` MATLAB script.
2. **`dataset2.csv`**: Contains data generated from the **Windowed Data Processing technique** (FFT2), created using the `FFT_2_method.mlx` MATLAB script.

These datasets serve as input for the machine learning models implemented in this project.

## Required Libraries and Dependencies

### MATLAB
To run the MATLAB scripts, ensure you have the following:
- MATLAB R2020a or later
- Signal Processing Toolbox

### Python
For the Python notebooks, the following libraries are required:
- TensorFlow (>=2.0)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

You can install the necessary Python libraries using pip:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

