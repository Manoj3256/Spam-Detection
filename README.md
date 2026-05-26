# Spam Detection using Machine Learning

A machine learning project that classifies emails/SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and multiple classification algorithms.

---

## Project Overview

Spam detection is one of the classic applications of Machine Learning. In this project, raw text messages are cleaned, transformed into numerical features using TF-IDF Vectorization, and classified using multiple ML models. The models are then compared based on Precision, Recall, and F1 Score to find the best performer.

---

## Project Structure

```
Spam Detection/
│
├── data/
│   └── spam.csv               # Dataset
│
├── model.ipynb                # Notebook
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation(current file)
```

---

## Tech Stack

|      Tool      |       Purpose          |
|----------------|------------------------|
| Python         | Programming Language   |
| Pandas & NumPy | Data manipulation      |
| NLTK           | Text preprocessing     |
| Scikit-learn   | ML models & evaluation |
| Matplotlib     | For visualization      |

---

## Project Pipeline

```
Raw Text Data
     ↓
Data Preprocessing (Lowercase, Tokenization, 
Remove Special Characters, Stop Words, Stemming)
     ↓
TF-IDF Vectorization
     ↓
Train / Test Split (80/20)
     ↓
Model Training & Comparison
     ↓
Evaluation (Precision, Recall, F1 Score)
```

---

## Text Preprocessing Steps

1. Lowercasing — Convert all text to lowercase
2. Remove Special Characters — Strip out symbols and punctuation
3. Tokenization — Split text into individual words
4. Remove Stop Words — Remove common words like "the", "is", "a"
5. Stemming — Reduce words to their root form (e.g. "running" to "run")

---

## Models Used

| Model                           | Description                                 |
|---------------------------------|---------------------------------------------|
| Logistic Regression             | Linear classifier for binary classification |
| Support Vector Classifier (SVC) | Finds optimal decision boundary             |
| Random Forest                   | Ensemble of decision trees                  |
| Decision Tree                   | Tree-based rule learning                    |
| K-Nearest Neighbors (KNN)       | Distance-based classification               |
| Bagging Classifier              | Reduces variance using bootstrap            |
| AdaBoost                        | Boosting weak learners                      |
| Dummy Classifier                | Baseline model for comparison               |

---

## Results

| Model               | Precision | Recall | F1 Score |
|---------------------|-----------|--------|----------|
| Random Forest       | 1.000     | 0.820  | 0.901    |
| KNN                 | 1.000     | 0.353  | 0.522    |
| SVC                 | 0.985     | 0.860  | 0.918    |
| Logistic Regression | 0.961     | 0.653  | 0.778    |
| Bagging             | 0.925     | 0.820  | 0.869    |
| AdaBoost            | 0.874     | 0.600  | 0.711    |
| Decision Tree       | 0.807     | 0.807  | 0.807    |
| Dummy               | 0.000     | 0.000  | 0.000    |

### Best Model: Support Vector Classifier (SVC)
SVC achieved the best F1 Score of 0.918, making it the most balanced model for spam detection in terms of both Precision and Recall.

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/Manoj3256/spam-detection.git
cd spam-detection
```
2. Install required libraries
```bash
pip install -r requirements.txt
```
3. Download NLTK data (run once)
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
4. Open and run the notebook
```bash
jupyter notebook model.ipynb
```
---

## Visualization

The project includes a grouped bar chart comparing Precision vs Recall across all models, making it easy to visually identify the best performing classifier.

---

## References

- Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — Aurelien Geron

---

## Author

Manoj  
Feel free to connect and give feedback

Thank You!
