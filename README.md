# abide-fmri-harmonization-classifier
A PyTorch-based CNN-LSTM model with neuroHarmonize for classifying Autism Spectrum Disorder from ABIDE II raw fMRI data(not preprocessed).

This repository presents a deep learning pipeline designed to classify Autism Spectrum Disorder (ASD) using resting-state fMRI (rs-fMRI) data from the ABIDE dataset. A central focus is on mitigating "batch effects" or site-specific variations, which are common in multi-site neuroimaging studies, through the integration of the neuroHarmonize library. The goal is to enhance the generalizability and robustness of ASD classification models.

Table of Contents

1. Project Overview
2. Motivation
3. Dataset
4. Data Preprocessing and Initial Chunking
5. Lazy Data Loading & Caching
6. Feature Extraction (CNN-LSTM)
7. Harmonization with neuroHarmonize
8. Classification
9. Requirements
10. Results
11. Contributing
12. License
13. Acknowledgements

1. Project Overview
This project implements an end-to-end pipeline for ASD classification from fMRI data, comprising data preprocessing, efficient lazy loading, deep learning-based feature extraction (CNN-LSTM), and a crucial harmonization step to correct for scanner-specific biases. The pipeline concludes with training a classifier on the harmonized features to predict ASD diagnosis.

2. Motivation
Autism Spectrum Disorder (ASD) research significantly benefits from large neuroimaging datasets like ABIDE II, which provide statistical power to identify brain-based biomarkers. However, data collected from multiple sites (different MRI scanners, protocols) inherently contain non-biological variations known as "batch effects." These effects can:

Confound scientific findings.

Limit the generalizability of machine learning models trained on such data.

This project directly addresses this challenge by integrating state-of-the-art deep learning for feature extraction with established harmonization techniques, aiming to build more reliable and generalizable ASD diagnostic tools.

3. Dataset
ABIDE II (Autism Brain Imaging Data Exchange II): This project leverages data from ABIDE II, a large-scale, open-access, multi-site initiative providing brain imaging and comprehensive phenotypic data.
Data Type: The primary input is resting-state functional MRI (rs-fMRI), used to study intrinsic brain functional connectivity patterns. Each fMRI scan is a 4D volume (X, Y, Z, Time).
Labels: Diagnostic labels (ASD vs. Typically Developing controls) are used for classification.

4. Methodology
The pipeline is structured into two main scripts, data_preprocessing.py and main_pipeline.py, to handle data efficiently and perform the multi-stage analysis.
a. Data Preprocessing and Initial Chunking 
This initial script prepares the raw ABIDE fMRI data for the main pipeline:
File Discovery & Loading: Scans a specified base directory for .mat files (assumed to contain HDF5 fMRI data).
Data Extraction & Masking: Extracts 4D fMRI volumes (fMRIdata/orig), applies provided brain masks (fMRIdata/mask), and retrieves diagnostic labels (fMRIdata/dx).
Initial Chunking: To manage large data volumes, the raw, masked fMRI data is saved into smaller, manageable .pkl files (chunks) in the D:\Abide\Chunks directory.
NaN Inpainting & Validation: Iterates through these initial chunks, identifies images with a low percentage (less than 5%) of NaN (Not a Number) values, and performs Gaussian inpainting to interpolate these missing values spatially. These "valid" and cleaned images are then saved into new .pkl chunks in the D:\Abide\Valid_Chunks directory, which serves as the input for the main pipeline.

b. Deep Learning Pipeline with Harmonization 
This is the core script that trains the classification model and applies harmonization:

5. Lazy Data Loading & Caching:
It reads from the D:\Abide\Valid_Chunks directory. Instead of loading all images into memory, a torch.utils.data.Dataset (Variable4DDataset) uses lazy loading to load only the necessary data for each batch.
A small in-memory chunk cache is implemented to store recently accessed chunks, striking a balance between memory efficiency and loading speed.
On-the-fly, each fMRI volume's voxel time series is z-normalized (mean 0, standard deviation 1) across its time dimension.

6. Feature Extraction (CNN-LSTM):
A custom CNN_LSTM_Harmonizer deep learning model is designed to process 4D fMRI data:
SliceCNN: A 2D Convolutional Neural Network processes individual (X, Y) slices across all Z-dimensions and time points of an fMRI volume. This extracts fixed-size, spatial features from each slice.
Aggregation: Features from all Z-slices at a given time point are averaged to create a single feature vector representing the overall brain state at that specific time.
LSTM: A Long Short-Term Memory (LSTM) network then processes these time-series feature vectors. The LSTM captures temporal dependencies within the fMRI signal and outputs a compact, informative feature vector (its final hidden state) for the entire fMRI scan.
This CNN-LSTM model is initially trained as a feature extractor using a standard classification objective.

7. Harmonization with neuroHarmonize:
After the CNN-LSTM feature extractor is trained, the learned LSTM features (the output of the LSTM layer) are extracted for both the training and testing datasets.
The neuroHarmonize library, which implements ComBat harmonization, is then applied:
harmonizationLearn: Learns the site-specific "batch effects" from the extracted features of the training set.
harmonizationApply: Applies the learned harmonization model to both the training and testing features, effectively correcting for non-biological scanner variations while preserving biological signal.

8. Classification:
A simple linear classifier is then trained on the harmonized features (from the LSTM output) to predict the ASD status of each subject.
This separate fine-tuning step on the harmonized features allows for a direct assessment of the impact of the harmonization step on classification performance and generalizability.

9. Requirements
Python 3.x
PyTorch (torch)
NumPy (numpy)
SciPy (scipy)
h5py (h5py)
scikit-learn (sklearn)
neuroHarmonize (neuroHarmonize)
You can install most of these dependencies using pip:```bash
pip install torch numpy scipy h5py scikit-learn neuroHarmonize

10. Results

The script will output training loss values for both the CNN-LSTM feature extractor and the final linear classifier. The ultimate metric reported will be the **final test accuracy after harmonization**.

It is expected that the harmonization step will contribute to improved classification performance and robustness, by effectively removing site-specific noise and allowing the model to learn more generalizable brain patterns related to ASD.

11. Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

12. License

This project is licensed under the [MIT License](LICENSE).

13 Acknowledgements

*   **ABIDE Consortium:** For collecting and openly sharing the invaluable ABIDE dataset.
*   **`neuroHarmonize` Library:** For providing robust tools for batch effect correction in neuroimaging data.
*   **PyTorch Community:** For the open-source deep learning framework.
