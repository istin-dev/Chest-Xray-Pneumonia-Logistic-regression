# 🩺 Chest X-ray Pneumonia Detection App

A Machine Learning-based web application that detects **Pneumonia** from Chest X-ray images using **Logistic Regression + PCA**, deployed with **Streamlit**.

---

## 📌 Features

* 📤 Upload Chest X-ray images (JPG, PNG, JPEG)
* 🧠 Machine Learning prediction using Logistic Regression
* ⚡ Dimensionality reduction using PCA
* 📊 Displays prediction with confidence score
* 🌐 Simple and interactive web interface using Streamlit

---

## 🛠️ Tech Stack

* Python 🐍
* Streamlit 🌐
* Scikit-learn 🤖
* NumPy 📊
* Pillow 🖼️

---

## 📂 Project Structure

```
chest_xray/
│── app/
│   ├── app.py
│   ├── pneumonia_logistic_model_final_pca_30.pkl
│   ├── pca_model.pkl
│── requirements.txt
│── logistic-pneumonia.ipynb
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/istin-dev/Chest-Xray-Pneumonia-Logistic-regression.git
cd Chest-Xray-Pneumonia-Logistic-regression/chest_xray
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
cd app
streamlit run app.py
```

---

## 📸 How it Works

1. Upload a Chest X-ray image
2. Image is preprocessed (grayscale + resized)
3. PCA reduces dimensions
4. Logistic Regression predicts
5. Output shows:

   * ✅ Normal Lung
   * ❌ Pneumonia Detected
   * 📊 Confidence Score

---

## ⚠️ Notes

* Model trained using Logistic Regression with PCA
* Ensure model files are in the `app/` directory
* Compatible with Streamlit Cloud deployment

---

## 📈 Future Improvements

* 🔬 Use Deep Learning (CNN) for higher accuracy
* 📊 Add Grad-CAM visualization
* 📱 Mobile-friendly UI
* ☁️ Add API support

---

## 👨‍💻 Author

**Istin B**
🎓 Saveetha Engineering College
**Karan A**
🎓 Saveetha Engineering College


---

## ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 📢 Share with others

---

## 📜 License

This project is open-source and available under the MIT License.
