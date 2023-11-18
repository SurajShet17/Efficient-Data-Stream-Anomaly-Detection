
# Efficient Data Stream Anomaly Detection - Credit Card Fraud Analysis

## Description

### Introduction
Anomaly detection identifies outliers in patterns, useful in various business contexts like intrusion detection and health monitoring. This repository contains Jupyter notebook which explores credit card fraud detection as a case study and a training.py file which is used to predict real time data stream for anomaly detection. A Dockerfile is also provided which sets up an environment for running a Python application for anomaly detection.

Anomalies are categorized into point (e.g., detecting fraud based on spent amount), contextual (context-specific abnormalities, common in time-series data), and collective anomalies (grouped instances, detecting potential cyber attacks). Anomaly detection differs from noise removal and novelty detection, the latter focusing on unobserved patterns in new data, and noise removal involves eliminating noise from a meaningful signal.

### About Dataset
The dataset, from September 2013, records credit card transactions by European cardholders over two days, comprising 492 frauds out of 284,807 transactions. Notably unbalanced, frauds make up only 0.172% of all transactions. The dataset includes numerical variables resulting from PCA transformation, excluding original features due to confidentiality. Features V1 to V28 are PCA-derived principal components, with 'Time' and 'Amount' being the only non-transformed ones. 'Time' indicates seconds since the first transaction, and 'Amount' represents the transaction amount, useful for cost-sensitive learning. The 'Class' feature, denoting fraud (1) or non-fraud (0), serves as the response variable. Given the class imbalance, accuracy is best evaluated using the Area Under the Precision-Recall Curve (AUPRC), as confusion matrix accuracy lacks meaning in unbalanced classification.

Dataset Link : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Model Selection
Model Prediction:

We'll use Isolation Forest Algorithm for anomaly detection:

Isolation Forest Algorithm:

Innovative method for rare anomaly detection, emphasizing isolation.
Low time complexity, minimal memory usage, and efficient model construction.
Ideal for imbalanced patterns where anomalies are scarce.
How It Works:

Randomly selects features and split values.
Anomalies require fewer conditions, determining the anomaly score.

## Requirements

- Python 3.x
- all libraries contained in requirements.txt file

## Installation

1. Clone this repository to your local machine.
2. Install all the required Python packages (present in requirements.txt file) by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the script using the following command:

```bash
python training.py
```

## Configuration
### Important 

The `configs` involves the following parameters that can be adjusted for training:

- `n_estimators`: This parameter represents the number of isolation trees in the forest.
- `max_samples`: This parameter determines the number of samples used to build each isolation tree.
- `contamination`: The contamination parameter sets the proportion of outliers in the dataset.

## Contact

For any issues or questions, please contact surajshet5555@gmail.com.

## References
1. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. https://www.analyticsvidhya.com/blog/2021/07/anomaly-detection-using-isolation-forest-a-complete-guide/#:~:text=In%20an%20Isolation%20Forest%2C%20randomly,more%20cuts%20to%20isolate%20them.

---
