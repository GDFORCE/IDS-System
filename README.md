# 🚀 Network Intrusion Detection System (NIDS) - KDD Cup 1999

## 📌 Project Overview
This project focuses on developing a **Network Intrusion Detection System (NIDS)** using the **KDD Cup 1999** dataset. The goal is to preprocess the dataset, extract features, and train machine learning models to detect network intrusions.

---

## 📂 Dataset Information
- **Dataset Name:** KDD Cup 1999
- **Source:** UCI Machine Learning Repository
- **Total Records:** ~4.9 million
- **Classes:** Normal & 22 Attack Types
- **Features:** 41 attributes + 1 label

---

## ⚙️ Steps Performed

### 1️⃣ Data Preprocessing 🛠️
- Loaded the dataset and handled missing values.
- Converted categorical features into numerical form using **Label Encoding**.
- Normalized numerical features using **Min-Max Scaling**.
- Split the dataset into **training** and **testing** sets.

### 2️⃣ Feature Engineering 🔍
- Performed **feature selection** to identify the most relevant attributes.
- Used **Principal Component Analysis (PCA)** for dimensionality reduction.

### 3️⃣ Model Training 🤖
- Trained multiple machine learning models:
  - ✅ **Decision Tree**
  - ✅ **Random Forest**
  - ✅ **XGBoost**
  - ✅ **Support Vector Machine (SVM)**
  - ✅ **Artificial Neural Networks (ANN)**
- Evaluated models based on **accuracy, precision, recall, and F1-score**.

### 4️⃣ Anomaly Detection ⚠️
- Implemented **Isolation Forest** and **Autoencoders** to detect anomalies.
- Addressed **class imbalance** using **SMOTE (Synthetic Minority Over-sampling Technique)**.

### 5️⃣ Deployment 🌍
- Built a **Flask API** to serve the trained model.
- Integrated a **Streamlit web interface** for real-time intrusion detection.

---

## 📊 Results & Performance
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|----------|--------|----------|
| Decision Tree       | 95.3%    | 94.5%    | 95.1%  | 94.8%    |
| Random Forest      | 97.2%    | 96.8%    | 97.0%  | 96.9%    |
| XGBoost           | 98.1%    | 97.8%    | 98.0%  | 97.9%    |
| SVM               | 92.4%    | 91.5%    | 92.0%  | 91.7%    |
| ANN               | 96.5%    | 95.9%    | 96.2%  | 96.0%    |

---

## 🚀 Technologies Used
- 🐍 **Python** (pandas, NumPy, scikit-learn, TensorFlow, PyTorch)
- 📊 **Data Processing** (Pandas, NumPy, Scikit-learn)
- 🎯 **Machine Learning Models** (Decision Tree, Random Forest, XGBoost, SVM, ANN)
- 🌍 **Deployment** (Flask, Streamlit)
- 🔍 **Anomaly Detection** (Isolation Forest, Autoencoders)

---

## 📌 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/nids-kdd99.git
   cd nids-kdd99
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask API:
   ```bash
   python app.py
   ```
4. Run the Streamlit interface:
   ```bash
   streamlit run interface.py
   ```

---

## 📜 Acknowledgments
- **UCI Machine Learning Repository** for providing the dataset.
- Open-source contributors for the ML frameworks used.

---

## 📞 Contact
📧 Email: [Bhaodai2005@gmail.com](mailto:bhanodai2005@gmail.com)  

