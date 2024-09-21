# Machine Learning Approaches for Efficient FFT Approximation in OFDM Systems: A Data-Driven Analysis

### Can Global Function Approximation Methods Approximate Fast Fourier Transforms?

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

<code>
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn</code>


<h2>Description of Files</h2>

<h3>MATLAB Files</h3>
<ul>
    <li><strong>FFT_1_method.mlx</strong>: 
        <p>This file implements the FFT1 method, generating a dataset for the standard FFT approach. It creates synthetic signals and computes their FFTs, which serve as the ground truth for model training.</p>
    </li>
    <li><strong>FFT_2_method.mlx</strong>: 
        <p>This file implements the FFT2 method, generating data for the windowed data processing technique. Similar to FFT1, it produces datasets that reflect a different processing approach, aimed at training machine learning models.</p>
    </li>
</ul>

<h3>Python Files</h3>
<ul>
    <li><strong>method1.ipynb</strong>: 
        <p>This notebook implements a dense neural network model to approximate the FFT operation based on the dataset generated from the FFT1 method. It includes data preprocessing, model training, and evaluation, alongside visualizations of training and validation losses.</p>
    </li>
    <li><strong>method2.ipynb</strong>: 
        <p>This notebook implements the Adaline model for approximating the FFT based on the dataset from the FFT2 method. It follows a similar structure to <code>method1.ipynb</code>, emphasizing a different machine learning architecture while presenting comparative results.</p>
    </li>
</ul>
