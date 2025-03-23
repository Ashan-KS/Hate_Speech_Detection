# Hate Speech Detection using LSTM

## 📌 Project Overview
This project focuses on detecting hate speech in text using a Long Short-Term Memory (LSTM) model. The model is trained on a dataset containing labeled text data, classifying content as hate speech, offensive speech and neither. The primary goal was to build a deep learning-based NLP model that can effectively identify harmful content in social media posts and online discussions.

## 📂 Dataset
The dataset used for training the model is available at:
https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

## 🧠 Model Used
- Long Short-Term Memory (LSTM) Neural Network
- Preprocessing steps include removing unnecessary characters, removing spacing, lemmatizing, removing stop words, adding padding, and one hot representation.
- The model is trained using TensorFlow/Keras.

## 📊 Model Performance

Classification Report: 

Model Accuracy : 89.57574367523193

              precision    recall  f1-score   support

           0       0.91      0.93      0.92      3812
           1       0.92      0.91      0.91      3807
           2       0.75      0.71      0.73       890

    accuracy                           0.90      8509
   macro avg       0.86      0.85      0.85      8509
weighted avg       0.89      0.90      0.90      8509

## 📌 Libraries Used
- Python 3.x
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn
- Spacy
- Matplotlib
- Seaborn
  
## 🚀 How to Run the Project

### Open the Jupyter Notebook
Open `Hate_Speech_Detection_LSTM.ipynb`

### Add Dataset
Ensure the dataset is available in the specified directory or download it from the provided link.

### Train the Model
Follow the notebook instructions to preprocess data, train the LSTM model, and evaluate performance.
