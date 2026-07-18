# Breast Cancer Machine Learning Classifier 🩺💻

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow.svg)

## 1. Project Overview
This repository contains an end-to-end Machine Learning pipeline designed to classify breast cancer tumors as either **Malignant** or **Benign**. By leveraging the classic Breast Cancer Wisconsin (Diagnostic) dataset, this project demonstrates proficiency in data exploration, feature engineering, model training, and clinical evaluation metrics.

## 2. Biological & Clinical Context
*(To be expanded as the project progresses)*

Accurate classification of breast masses is a critical step in clinical oncology. This project analyzes mathematical features computed from digitized images of a fine needle aspirate (FNA) of breast masses. The algorithms deployed here aim to identify the specific cellular dimensions that most strongly correlate with malignancy, providing a computational "second opinion" for diagnostic pathology.

## 3. Dataset Description
*   **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset (via `sklearn.datasets`)
*   **Instances:** 569 patients
*   **Features:** 30 numerical attributes (e.g., radius, texture, perimeter, area, smoothness, concavity)
*   **Target Variable:**
    *   `0`: Malignant (212 cases)
    *   `1`: Benign (357 cases)

## 4. Project Structure
```text
breast-cancer-ml-classifier/
│
├── data/               # Raw and processed datasets (ignored by git)
├── notebooks/          # Jupyter notebooks for EDA and model prototyping
│   └── 01_data_exploration.ipynb
├── scripts/            # Standalone Python scripts for production
├── models/             # Saved serialized models (e.g., .pkl)
├── results/            # Exported figures, metrics, and reports
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 5. Technologies & Tools
*   **Language:** Python 3
*   **Data Manipulation:** `pandas`, `numpy`
*   **Machine Learning:** `scikit-learn`
*   **Data Visualization:** `matplotlib`, `seaborn`
*   **Environment:** WSL (Linux), Jupyter, VS Code

## 6. Methodology
1.  **Exploratory Data Analysis (EDA):** Class balance, feature distributions, and collinearity mapping. *[In Progress]*
2.  **Data Preprocessing:** Train/test splitting, standard scaling, and handling multicollinearity. *[Pending]*
3.  **Model Building:** Baseline Logistic Regression and advanced non-linear classifiers (e.g., Random Forest). *[Pending]*
4.  **Evaluation & Tuning:** Focus on clinical metrics like Recall (Sensitivity) to minimize false negatives. *[Pending]*

## 7. How to Run
To reproduce the findings in this repository, clone it and set up a local virtual environment:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/breast-cancer-ml-classifier.git
cd breast-cancer-ml-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 8. Results & Conclusion
*(To be filled upon completion of model evaluation)*
