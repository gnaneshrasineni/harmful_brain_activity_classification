# Harmful Brain Activity Classification
CS 6190 Project - Probabilistic Machine Learning - u1470943 - Gnanesh Rasineni

## Project Overview
This project focuses on automating the classification of EEG spectrograms using machine learning techniques. Electroencephalograms (EEG) are crucial for diagnosing neurological disorders as they record brain's electrical activity. This project uses the HMS dataset, which includes categories such as seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), and "other".

## Methods
The project involves preprocessing and augmentation of the EEG data using techniques such as the Short-Time Fourier Transform and horizontal flips. For classification, an EfficientNet model optimized with KL-Divergence loss is employed. The model's performance is assessed through k-fold cross-validation, to enhance the precision and efficiency of EEG data analysis.

## Installation
Clone this repository and install the required packages:
```bash
git clone https://github.com/gnaneshrasineni/harmful-brain-activity-classification.git
cd harmful-brain-activity-classification
python main.py


