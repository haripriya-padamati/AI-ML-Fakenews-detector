# 🛡️ FakeShield — Fake News Detection using NLP & Machine Learning

> **Detecting misinformation before it spreads.**  
> FakeShield is an AI/ML-powered fake news detection system that classifies news content as **Fake** or **Real** using NLP, TF-IDF feature extraction, and machine learning models.

---

## 🚀 Project Highlights

✅ NLP-based text preprocessing  
✅ TF-IDF vectorization with unigrams and bigrams  
✅ Multiple ML models trained and compared  
✅ Final model: Random Forest Classifier  
✅ Live fake/real news prediction function  
✅ Confusion matrix and performance metrics  
✅ Power BI dashboard for visual insights  
✅ Built during an AI/Data Science Hackathon  

---

## 📌 Problem Statement

In today’s digital world, misinformation spreads rapidly through news websites, social media, and online platforms. Manually verifying every article is difficult and time-consuming.

The goal of this project is to build an AI-based system that can automatically classify news content as:


Fake News
Real News


FakeShield acts as a **first-level screening system** that flags suspicious news so it can be reviewed further by humans or fact-checking teams.

---

## 🧠 What This Project Does

FakeShield takes news text as input and predicts whether it is fake or real.


News Statement / Article
        ↓
Text Cleaning
        ↓
TF-IDF Feature Extraction
        ↓
Machine Learning Model
        ↓
Fake / Real Prediction

---

## 📂 Dataset Used

The project uses the **FakeNewsNet dataset**.

The dataset contains three CSV files:

fnn_train.csv
fnn_dev.csv
fnn_test.csv

### Dataset Columns

| Column                   | Description                             |
| ------------------------ | --------------------------------------- |
| `id`                     | Unique news record ID                   |
| `date`                   | Date related to the news                |
| `speaker`                | Person/source associated with the claim |
| `statement`              | Short news claim or statement           |
| `sources`                | Source information                      |
| `pragraph`               | Paragraph/context information           |
| `fullText_based_content` | Full article-based content              |
| `label_fnn`              | Target label: fake or real              |

### Target Column

label_fnn


### Input Text Used

To give the model more context, two text columns were combined:

statement + fullText_based_content

---

## 🛠️ Tech Stack

| Area                    | Tools Used                                      |
| ----------------------- | ----------------------------------------------- |
| Programming Language    | Python                                          |
| Data Handling           | Pandas, NumPy                                   |
| Visualization           | Matplotlib, Seaborn                             |
| NLP Feature Extraction  | TF-IDF Vectorizer                               |
| Machine Learning        | Scikit-learn                                    |
| Models                  | Logistic Regression, Naive Bayes, Random Forest |
| Dashboard               | Power BI                                        |
| Development Environment | VS Code / Jupyter Notebook                      |

---

## 📦 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
```

---

## 🔄 Project Workflow

1. Load train, dev, and test datasets
2. Understand columns and target labels
3. Combine statement and full article text
4. Clean the text using NLP preprocessing
5. Perform basic EDA
6. Convert text into numerical features using TF-IDF
7. Train multiple ML models
8. Compare model performance
9. Select the best model
10. Evaluate using test data
11. Save predictions and metrics
12. Build Power BI dashboard
13. Test unseen news samples

---

## 🧹 Text Preprocessing

Raw news text may contain URLs, symbols, punctuation, HTML tags, and inconsistent capitalization.
To reduce noise, the following preprocessing steps were applied:

* Converted text to lowercase
* Removed URLs
* Removed HTML tags
* Removed punctuation and special characters
* Removed extra spaces

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

### Why Text Cleaning?

Text cleaning helps the model focus on meaningful words instead of unnecessary noise.

Example:

Before: "BREAKING!!! Visit https://fake-news.com <br> Shocking claim!!!"
After:  "breaking visit shocking claim"

---

## 🔢 Feature Extraction using TF-IDF

Machine learning models cannot directly understand raw text.
So, the cleaned text was converted into numerical features using **TF-IDF Vectorization**.

```python
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)
```

### TF-IDF Settings

| Parameter              | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| `max_features=10000`   | Keeps top 10,000 important words/phrases |
| `stop_words="english"` | Removes common words like the, is, and   |
| `ngram_range=(1,2)`    | Uses single words and two-word phrases   |

### Why TF-IDF?

TF-IDF gives importance to useful words and phrases based on how frequently they appear in one article compared to the whole dataset.

Raw Text → Numerical Features → Machine Learning Model

---

## 🤖 Models Trained

Three models were trained and compared.

| Model               | Why Used                                         |
| ------------------- | ------------------------------------------------ |
| Logistic Regression | Fast and strong baseline for text classification |
| Naive Bayes         | Popular probabilistic model for NLP tasks        |
| Random Forest       | Ensemble model that captures complex patterns    |

---

## 🏆 Final Model

The best-performing model was:

Random Forest Classifier

Random Forest was selected because it gave the best performance during model comparison.

---

## 📊 Final Model Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **79.60%** |
| Precision | **86.40%** |
| Recall    | **79.60%** |
| F1 Score  | **79.62%** |

---

## 📈 Classification Report

`
              precision    recall  f1-score   support

        fake       0.66      1.00      0.80       418
        real       1.00      0.66      0.80       636

    accuracy                           0.80      1054
   macro avg       0.83      0.83      0.80      1054
weighted avg       0.86      0.80      0.80      1054


---

## 🔍 Confusion Matrix

|             | Predicted Fake | Predicted Real |
| ----------- | -------------: | -------------: |
| Actual Fake |        **417** |              1 |
| Actual Real |            214 |        **422** |

### Key Interpretation

The model correctly detected:


417 out of 418 fake news samples


This means FakeShield is very strong at catching fake news.

However, some real news samples were flagged as fake. This makes the system useful as a **first-level screening tool**, where suspicious content can be sent for human verification.

---

## 🧪 Live Prediction Function

FakeShield can classify unseen news samples using this function:

```python
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vectorized = tfidf.transform([cleaned])
    prediction = final_model.predict(vectorized)[0]
    
    if hasattr(final_model, "predict_proba"):
        confidence = final_model.predict_proba(vectorized).max()
    else:
        confidence = None
    
    return prediction, confidence
```

### Example

```python
sample_news = "Government announces new education policy for students."
predict_news(sample_news)
```

Example output:

('real', 0.73)

---

## 📁 Output Files Generated

| File                    | Purpose                                         |
| ----------------------- | ----------------------------------------------- |
| `model_results.csv`     | Comparison of trained models                    |
| `final_metrics.csv`     | Final accuracy, precision, recall, and F1-score |
| `prediction_output.csv` | Actual vs predicted results                     |
| `eda_output.csv`        | EDA data for dashboard                          |
| `confusion_matrix.csv`  | Confusion matrix values                         |

---

## 📊 Power BI Dashboard

A Power BI dashboard was created to visually present the system performance.

### Dashboard Includes

Fake vs Real News Distribution
Model Comparison
Final Metrics
Actual vs Predicted Labels
Correct vs Wrong Predictions
Confusion Matrix
Sample Prediction Results

### Dashboard Purpose

The dashboard helps evaluators quickly understand:

* How the dataset is distributed
* Which model performed best
* How accurate the final model is
* Where the model makes mistakes
* How predictions compare with actual labels

---

## 💡 Key Insights

* Random Forest achieved the best performance.
* The final model reached around **80% test accuracy**.
* Fake news recall was almost perfect.
* Only **1 fake news sample** was missed.
* Some real news samples were flagged as fake.
* The system is best used as a first-level screening tool.
* Human verification can be added for uncertain or flagged cases.

---

## 🏗️ Project Structure

FakeShield/
│
├── data/
│   ├── fnn_train.csv
│   ├── fnn_dev.csv
│   └── fnn_test.csv
│
├── notebooks/
│   └── fake_news_detection.ipynb
│
├── outputs/
│   ├── model_results.csv
│   ├── final_metrics.csv
│   ├── prediction_output.csv
│   ├── eda_output.csv
│   └── confusion_matrix.csv
│
├── dashboard/
│   └── powerbi_dashboard.pbix
│
└── README.md

---

## ⚙️ How to Run

### 1. Clone the Repository

git clone <repository-link>
cd FakeShield


### 2. Install Required Libraries

pip install pandas numpy matplotlib seaborn scikit-learn


### 3. Open the Notebook

jupyter notebook

Open:

fake_news_detection.ipynb

### 4. Run All Cells

The notebook will:

Load the dataset
Clean the text
Apply TF-IDF
Train ML models
Evaluate performance
Generate output CSV files

---

## 🧠 Why This Project Matters

Fake news can influence opinions, create panic, damage reputations, and spread misinformation at large scale.

FakeShield helps reduce this risk by automatically identifying suspicious news content and supporting faster human review.

---

## 🔮 Future Scope

FakeShield can be improved further by adding:

* BERT or RoBERTa-based deep learning models
* Explainable AI for word-level reasoning
* Source credibility scoring
* Fact-checking API integration
* Browser extension
* Social media post verification
* Multilingual fake news detection
* Human review dashboard
* Real-time fake news alert system
* Deployment as a web app

---

## 🏁 Conclusion

FakeShield successfully demonstrates a complete NLP-based machine learning pipeline for fake news detection.

The system performs:

Text preprocessing
TF-IDF feature extraction
Model training
Model comparison
Final evaluation
Prediction generation
Dashboard visualization


With around **79.6% accuracy** and extremely strong fake-news recall, FakeShield can be used as an effective first-level misinformation screening system.

---

## 👥 Team

Developed as part of an AI/Data Science Hackathon.

Team Name: <Team AARIV>
Members: <Haripriya Padamati>
Members: <Dushyant Vasupalli>


---

## ⭐ Final Note

> Fake news spreads fast.
> Detection should be faster.
> **FakeShield helps make information safer, one article at a time.**


