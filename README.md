# Predicting Inflammatory Bowel Disease (IBD) Using National Health Interview Survey (NHIS) Data (2023-2024)

## Project Overview

This project applies supervised machine learning techniques to predict **severe Inflammatory Bowel Disease (IBD)** using data from the **National Health Interview Survey (NHIS)**. The goal is to evaluate how different models and class-imbalance handling strategies perform on a rare-disease prediction task, with particular emphasis on minority-class performance.

The analysis was completed as a final project for **SI 670: Applied Machine Learning** and integrates concepts such as feature engineering, imbalanced classification, pipeline-based preprocessing, cross-validation, and model evaluation.


## Explore NHIS Data

This project uses the National Health Interview Survey (NHIS) Sample Adult datasets to explore demographic and health-related variables associated with Inflammatory Bowel Disease (IBD).

You can learn more about the NHIS and access the datasets here for [2024](https://www.cdc.gov/nchs/nhis/documentation/2024-nhis.html) and [2023](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html).

### Relevant Documentation

* [NHIS 2023 Sample Adult Codebook](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/adult-codebook.pdf)

* [NHIS 2024 Sample Adult Codebook](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2024/adult-codebook.pdf)

## Data Description

The project uses two NHIS Sample Adult datasets:

* `adult23.csv` (NHIS 2023)
* `adult24.csv` (NHIS 2024)

These datasets are vertically concatenated to form a single analytical dataset.

### Target Variable

A binary target variable, **`IBDSEV_A`**, is constructed:

* `1` (Severe IBD): Respondent reports either severe ulcerative colitis or severe Crohn’s disease
* `0` (Non-IBD): All other respondents

The original disease-specific severity variables are removed after target construction to avoid redundancy.

---

## Feature Engineering

* Only numeric variables with **binary-like coding** (NHIS-style responses such as 1, 2, 7, 8, 9, NaN) are considered.
* Features are initially filtered based on **absolute differences in “Yes” response rates** between IBD and non-IBD groups.
* Statistical significance is confirmed using **chi-squared tests**.
* All retained features are converted to strict binary indicators (1 = Yes, 0 = No).

### Demographic Features

The following demographics are explicitly included:

* Sex (binary encoded)
* Region, race/ethnicity, and education (one-hot encoded)
* Age (numeric, with non-substantive codes set to missing)
* Race/ethnicity (one-hot encoded with non-substantive codes set to missing)
* Education level (one-hot encoded with non-substantive codes set to missing)

---

## Modeling Approach

This is a **highly imbalanced binary classification problem**, with severe IBD cases representing a small minority of observations.

### Models Evaluated

* Dummy Classifiers (Prior and Stratified)
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Multilayer Perceptron (MLP)
* Random Forest
* Random Forest with Bagging
* XGBoost
* Stacking Classifier (ensemble of multiple base learners)

### Class Imbalance Strategies

Each learned model is evaluated under three regimes:

1. **No Resampling** (class weighting only)
2. **Random Undersampling**
3. **SMOTE Oversampling**

---

## Evaluation Strategy

* **5-fold TimeSeriesSplit cross-validation** is used to preserve chronological order between survey years.
* All preprocessing (imputation, scaling, resampling) is performed **within pipelines** to prevent data leakage.
* Performance metrics include:

  * **F1 Score**
  * **ROC–AUC**
  * **Minority-Class Recall**, to assess the ability to correctly identify severe IBD cases

---

## Structure (when run by team)

```
.
├── Final_Code–SI_670_Applied_Machine_Learning.ipynb
├── data/
│   └── adult23.csv
│   └── adult24.csv
└── README.md
```

Our data files were too large to upload, so you can access them [here](https://drive.google.com/drive/folders/1mudCpRoK4Rj4_PDrtO6kwWVumpfySgo7?usp=drive_link)

Or, the entire Google Colab environment [here](https://drive.google.com/drive/folders/1IXILImZvjzstYqMX7R30fuI1U_Uo0gv9?usp=drive_link)


## How to Run the Project

This project can be run either **locally using Jupyter Notebook** or **in Google Colab**. The notebook includes setup code for both environments.

---

## Required Libraries

The notebook uses the following Python libraries:

* numpy
* pandas
* scikit-learn
* imbalanced-learn
* xgboost
* umap-learn
* matplotlib
* seaborn

---

## Option A: Running in Google Colab 

### 1. Upload Files to Google Drive

Ensure the following directory structure in your Google Drive:

```
MyDrive/
└── 670_Project/
    ├── Final_Code–SI_670_Applied_Machine_Learning.ipynb
    └── data/
        ├── adult23.csv
        └── adult24.csv
```

### 2. Open the Notebook in Colab

Open `Final_Code–SI_670_Applied_Machine_Learning.ipynb` using Google Colab.

### 3. Install Dependencies

The first cell installs required packages:

```python
!pip install imblearn
!pip install xgboost
!pip install umap-learn
```

(Other dependencies are preinstalled in Colab.)

### 4. Mount Google Drive

The notebook includes the following code:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Authorize access when prompted.

### 5. Data Path Configuration

The data path is defined as:

```python
data_path = '/content/drive/MyDrive/670_Project/data/'
```

No changes are required **as long as the folder structure is preserved**. You may need to adjust `data_path` if your Google Drive structure differs (especially with shared files).

### 6. Run the Notebook

Run all cells sequentially in the notebook to execute the analysis.

---

## Option B: Running Locally (Jupyter Notebook)

### 1. Install Dependencies

Create a local environment and install required packages:

```bash
pip install numpy pandas scikit-learn imblearn xgboost umap-learn matplotlib seaborn
```

### 2. Directory Structure

Ensure the following local structure:

```
.
├── Final_Code–SI_670_Applied_Machine_Learning.ipynb
└── data/
    ├── adult23.csv
    └── adult24.csv
```

### 3. Modify Drive Mount Code

When running locally:

* **Comment out** the Google Drive mounting lines:

```python
# from google.colab import drive
# drive.mount('/content/drive')
```

### 4. Update Data Path

Change the data path to point to the local `data/` directory:

```python
data_path = './data/'
```

### 5. Launch Jupyter Notebook

From the project directory, run:

```bash
jupyter notebook
```

Open `Final_Code–SI_670_Applied_Machine_Learning.ipynb` and run all cells sequentially.

---
