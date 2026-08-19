# Credit Card Fraud Detection Web App

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff69b4.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-card-fraud-detection-mlp.streamlit.app/)

An interactive web application built with Streamlit to detect fraudulent credit card transactions in real-time. This project uses a pre-trained **LightGBM (Light Gradient Boosting Machine)** model to classify transactions based on user-provided features.

The application serves as a user-friendly interface for the machine learning model trained in the accompanying Jupyter Notebook, `credit_card_fraud_detection.ipynb`.

## 🚀 Live Demo
You can try the live application deployed on Streamlit Community Cloud:

**[👉 Access the Live App Here](https://credit-card-fraud-detection-mlp.streamlit.app/)**

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Methodology](#-methodology)
- [Setup and Installation](#-setup-and-installation)
- [How to Use](#-how-to-use)
- [File Structure](#-file-structure)
- [License](#-license)

## 📌 Project Overview
This project addresses the critical task of identifying fraudulent credit card transactions. Due to the highly imbalanced nature of fraud data (fraudulent transactions are rare), a specialized approach is required. This application encapsulates a trained LightGBM model, providing an intuitive UI for real-time predictions. Users can input transaction details, and the app will return a fraud probability score along with a clear classification: **Fraudulent** or **Legitimate**.

## ✨ Features
- **Real-Time Prediction**: Instantly classify transactions as fraudulent or legitimate.
- **Interactive UI**: An intuitive sidebar allows users to input all necessary transaction features.
- **Probability Score**: Displays the model's confidence in its prediction as a probability score.
- **Feature Engineering**: Automatically calculates derived features like `distance` (using the Haversine formula) and time-based features from user input.
- **Detailed Feedback**: Shows the engineered features that were sent to the model for prediction.
- **Easy Deployment**: Includes a Dev Container configuration for one-click setup in environments like GitHub Codespaces.

## 💻 Technology Stack
- **Backend & ML**: Python
- **ML Model**: LightGBM
- **Web Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Core ML Library**: Scikit-learn
- **Development Environment**: VS Code, GitHub Codespaces

## ⚙️ Methodology
The LightGBM model was trained and evaluated following a systematic machine learning workflow in the `credit_card_fraud_detection.ipynb` notebook. The key steps were:

1.  **Data Loading**: The model was trained on the `fraudTrain.csv` dataset.
2.  **Data Cleaning**: Dropped irrelevant columns (e.g., `cc_num`, `first`, `last`, `trans_num`) and handled missing values.
3.  **Feature Engineering**:
    - **Age Calculation**: Calculated the cardholder's age from their date of birth (`dob`).
    - **Transaction Distance**: Computed the geographical distance between the cardholder's location and the merchant's location using the Haversine formula.
    - **Time-Based Features**: Extracted `hour`, `day`, `month`, and `weekday` from the transaction timestamp to capture temporal patterns.
4.  **Data Preprocessing**:
    - **Label Encoding**: Converted categorical features (`category`, `gender`, `city`, `state`, `job`) into numerical representations.
5.  **Model Training**:
    - A LightGBM classifier was chosen for its high performance, speed, and ability to handle large datasets.
    - The model was trained with parameters optimized for the imbalanced dataset, including `is_unbalance=True`.
6.  **Model Evaluation**: The model achieved an **AUC of ~0.9995** on the validation set, demonstrating excellent performance in distinguishing between fraudulent and legitimate transactions.
7.  **Model Serialization**: The final trained model was saved as `lgb_model.pkl` for use in this Streamlit application.

## 🚀 Setup and Installation
Follow these steps to run the application on your local machine.

**Prerequisites:**
- Python 3.9+
- `git`

**Step-by-step instructions:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ashish-kharde1/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add Required Files:**
    > **IMPORTANT**: The following files are required to run the application but are not included in the repository due to their size or because they are generated artifacts.
    >
    > - `lgb_model.pkl`: The trained model file. You must generate this by running the `credit_card_fraud_detection.ipynb` notebook.
    > - `fraudTrain.csv`: The training dataset. Download it from the [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place it in the root directory of this project.
    >
    > Your project directory should look like this before running the app:
    > ```
    > ├── app.py
    > ├── fraudTrain.csv      <-- Add this file
    > ├── lgb_model.pkl       <-- Add this file
    > ├── requirements.txt
    > └── ... (other files)
    > ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    Your browser should automatically open a new tab with the application running.

## 📖 How to Use
1.  Navigate to the **[Live Demo](https://credit-card-fraud-detection-mlp.streamlit.app/)**.
2.  Use the sidebar on the left to enter the details of the transaction you want to evaluate.
3.  Fill in all the fields under "Transaction Details," "Cardholder Information," and "Location Details."
4.  Click the **Detect Fraud** button.
5.  The application will display the prediction result, including the classification and the fraud probability score.

## 📂 File Structure
```
credit-card-fraud-detection/
│
├── .devcontainer/
│   └── devcontainer.json   # Configuration for GitHub Codespaces
│
├── app.py                  # The main Streamlit application script
├── LICENSE                 # MIT License file
├── README.md               # This README file
└── requirements.txt        # Python package dependencies
```

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.