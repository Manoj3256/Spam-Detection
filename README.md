# Spam Detection using Machine Learning (Dockerized)

A machine learning project that classifies emails/SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and multiple classification algorithms.
And creating a API using fastapi and dockerizing it for deploying.

---

## Project Overview

Spam detection is one of the classic applications of Machine Learning. In this project, raw text messages are cleaned, transformed into numerical features using TF-IDF Vectorization, and classified using multiple ML models. The models are then compared based on Precision, Recall, and F1 Score to find the best performer.
Used Fastapi for API and dockerized it with compose file for simple key to start.

---

## Project Structure

```
Spam Detection/
│
├── data/
│   └── spam.csv               # Dataset
├── app/
│   ├──appAPI.py               # API file
│   └──models/                 # models
│       ├── vectorizer_file.pkl
│       └── model_file.pkl
├── Dockerfile                 # docker file
├── compose.yaml               # Compose for running container
├── model.ipynb                # Notebook
├── requirements.txt           # Required libraries
├── .gitignore                 # To avoid pushing models and __pycache__ folder to git repository
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
|  uvicorn       | server to  run FastAPI |
|   joblib       | For downloading and loading models |
|   fastapi      |     creating API       |

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
| Support Vector Classifier(Final)| Finds optimal decision boundary             |
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


## References

- Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow
- https://docs.docker.com/get-started/workshop Dockers official website (tutorial)
- https://fastapi.tiangolo.com/tutorial/  FastAPI official website (tutorial)
---
