# Breast cancer machine learning classifier 🩺💻

[![Open in Streamlit](https://img.shields.io/badge/Open_in_Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://breast-cancer-ml-classifier-mzyzgtde5r3fcurqv6rw9s.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 1. Project overview
This repository contains an end-to-end machine learning pipeline and an interactive web application designed to classify breast cancer tumors as either **Malignant** or **Benign**. By leveraging the classic Breast Cancer Wisconsin (Diagnostic) dataset, this project demonstrates proficiency in data exploration, feature engineering, model training, and the deployment of a clinical diagnostic tool using Streamlit.

## 2. Biological and clinical context
Accurate classification of breast masses is a critical step in clinical oncology. This project analyzes mathematical features computed from digitized images of a fine needle aspirate (FNA) of breast masses—a minimally invasive biopsy procedure.

In clinical histopathology, malignant (cancerous) cells typically exhibit distinct morphological abnormalities compared to benign (non-cancerous) cells. The dataset captures these differences through specific cellular characteristics:

*   **Size and shape (radius, area, perimeter):** Malignant cells often show significant variation in cell size and shape, a medical condition known as *pleomorphism*.
*   **Nuclear contours (smoothness, concavity, compactness):** Benign cell nuclei tend to have smooth, round, and regular boundaries. In contrast, malignant nuclei frequently exhibit irregular, jagged, or highly concave contours.
*   **Chromatin distribution (texture):** Variations in grayscale values reflect abnormalities in the cell's DNA distribution (*hyperchromasia*), which is strongly indicative of the rapid, uncontrolled cell division typical of cancer.

By leveraging machine learning, this project aims to quantify these visual abnormalities, identifying which specific cellular dimensions most strongly correlate with malignancy to provide a highly accurate, computational "second opinion" for diagnostic pathology.

## 3. Dataset description
*   **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset (via `sklearn.datasets`)
*   **Instances:** 569 patients
*   **Features:** 30 numerical attributes (computed from the characteristics described above)
*   **Target variable:**
    *   `0`: Malignant (212 cases)
    *   `1`: Benign (357 cases)

## 4. Project structure
```text
breast-cancer-ml-classifier/
│
├── models/             # Saved serialized models (deployment-ready)
│   ├── breast_cancer_log_reg_model.pkl
│   └── breast_cancer_scaler.pkl
├── scripts/            # Main codebase and notebooks
│   └── 01_data_exploration.ipynb
├── app.py              # Streamlit web application
├── .gitignore          # Ignored files (e.g., venv/)
├── LICENSE             # Project license
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies

```

## 5. Technologies and tools

* **Language:** Python 3
* **Data manipulation:** `pandas`, `numpy`
* **Machine learning:** `scikit-learn`, `joblib`
* **Web framework:** `streamlit`
* **Data visualization:** `matplotlib`, `seaborn`
* **Environment:** WSL (Linux), Jupyter, VS Code

## 6. Methodology

1. **Exploratory data analysis (EDA):** Analyzed class balance, feature distributions, and utilized correlation heatmaps to identify multicollinearity among cellular characteristics.
2. **Data preprocessing:** Performed strict train/test splitting (455 training / 114 testing) to prevent data leakage, and applied `StandardScaler` for distance-based and gradient-descent optimization.
3. **Model building:** Developed and trained a baseline logistic regression model alongside an advanced non-linear classifier (random forest).
4. **Evaluation and serialization:** Evaluated models based on accuracy and clinical reliability. Serialized the winning model and scaler (`.pkl`) for future inference.
5. **User interface (UI):** Developed an interactive Streamlit application to serve the serialized model, allowing users to adjust clinical parameters via sliders and receive real-time diagnostic predictions.

## 7. How to run

To reproduce the findings and run the web application locally, clone this repository and set up a virtual environment:

```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/breast-cancer-ml-classifier.git](https://github.com/YOUR_GITHUB_USERNAME/breast-cancer-ml-classifier.git)
cd breast-cancer-ml-classifier
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

To start the interactive web application, run:

```bash
streamlit run app.py

```

*(Note for WSL users: If you encounter network or connection issues, force the server to bind to a specific port and address by running `streamlit run app.py --server.port 8502 --server.address 0.0.0.0`)*

## 8. Results and conclusion

The project successfully evaluated multiple algorithms to find the most robust classifier for tumor diagnosis:

* **Logistic regression (winner):** Achieved an outstanding accuracy of **98.25%** on unseen test data.
* **Random forest classifier:** Achieved a highly competitive accuracy of **95.61%**.

Due to its superior performance, **logistic regression** was selected as the final production model. The complete pipeline is successfully integrated into a user-friendly web interface, demonstrating a full end-to-end machine learning lifecycle from raw data to a deployable medical application.
