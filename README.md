
# 📰 Fake News Detection Using Random Forest

A machine learning-based project that classifies news articles as **Fake** or **Real** using the **Random Forest algorithm**. This project aims to assist in identifying and curbing the spread of misinformation by analyzing textual content.

---

## 📌 Project Overview

This project focuses on building a binary classification model using supervised machine learning to detect fake news. The model leverages textual features from articles, preprocesses them for optimal learning, and uses the **Random Forest** algorithm to determine authenticity.

---

## 📂 Dataset

- **Source:** Publicly available Fake News datasets (e.g., Kaggle).
- **Contents:**
  - News Article Text
  - Title
  - Author
  - Source
  - Label (Fake or Real)
- **Format:** CSV file

---

## 🔍 Steps Followed in the Project

### 1. Data Collection  
News datasets containing labeled samples as **Fake** or **Real** were gathered from open sources like Kaggle.

### 2. Data Preprocessing  
- Removed missing values or imputed them appropriately.  
- Cleaned textual content by removing stop words, punctuation, and converting text to lowercase.  
- Used **TF-IDF Vectorizer** to transform text into numerical features.

### 3. Feature Selection  
Selected relevant features such as article text and title for classification. Extraneous features were removed to improve performance.

### 4. Model Training  
- The dataset was split into training and testing subsets.
- **Random Forest Classifier** was applied using the `scikit-learn` library.

### 5. Prediction & Evaluation  
Model performance was evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

### 6. Feature Importance Analysis (Optional)  
Random Forest’s built-in feature importance function was used to identify the most influential terms or patterns that contributed to the fake/real classification.

---

## 🛠 Technologies Used

- **Python 3.x**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **Jupyter Notebook** or **VS Code**

---

## 📈 Results

- Achieved high accuracy and balanced precision-recall scores.
- The model effectively distinguished between fake and real news articles.
- Demonstrated robustness and interpretability of Random Forest in handling text-based classification.

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Fake-News-Detection-Using-Random-Forest.git
   cd Fake-News-Detection-Using-Random-Forest
   ```

2. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook Fake_News_Detection_RF.ipynb
   ```

   Or run the script:
   ```bash
   python fake_news_detector.py
   ```

---

## 📌 Project Status

✅ Completed - The Random Forest model has been implemented and tested.  
📈 Future Enhancements:
- Integration with a **web app** using **Flask** or **Streamlit**
- Testing with larger datasets or multilingual sources
- Comparison with other models like **Logistic Regression**, **LSTM**, or **BERT**

---

## 🤝 Contributing

Contributions are welcome!  
To contribute:
- Fork the repository
- Create a new branch (`git checkout -b feature-name`)
- Commit your changes (`git commit -m "Add feature"`)
- Push to the branch (`git push origin feature-name`)
- Create a Pull Request

---

## 📃 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share with proper attribution.

---

## ✍️ Author

**Bala Subramani**  
📧 bala026b@gmail.com  
🔗 [GitHub](https://github.com/bbalasubramani) | [LinkedIn](https://linkedin.com/in/balasubramani-dev)

---

## 📷 Screenshots

![Dashboard Screenshot](images/news1.jpg)
![Dashboard Screenshot](images/news2.jpg)
