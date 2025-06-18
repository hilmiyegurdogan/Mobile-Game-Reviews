import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer #(metin verilerini sayısal verilere dönüştürüyor)
from sklearn.preprocessing import LabelEncoder #converts class labels to numeric form. Positive' → 1, 'Negative' → 0
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

import nltk #naturelanguage toolkit
from nltk.corpus import stopwords #noktalama işaretlerini, gereksiz kelimeleri "the" gibi siler.
from nltk.stem import WordNetLemmatizer
import re
import string
nltk.download('stopwords')
nltk.download('wordnet')


df = pd.read_csv('googleplaystore_user_reviews.csv')
print(df.head())
print(df.info())
print(df['Sentiment'].value_counts())


df = df.dropna(subset=['Translated_Review', 'Sentiment'])

df = df[df['Sentiment'].isin(['Negative', 'Positive'])].copy()


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split() #metni kelimelere böler
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)



df['cleaned_review'] = df['Translated_Review'].astype(str).apply(clean_text)


tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_review'])
y = df['Sentiment']

le = LabelEncoder()
y_enc = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)


param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [200]
}

log_reg = LogisticRegression(random_state=42)


grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"En iyi parametreler: {grid_search.best_params_}")
print(f"En iyi F1 skoru (CV): {grid_search.best_score_:.4f}")

best_log_reg = grid_search.best_estimator_


models = {
    'LogisticRegression': best_log_reg,
    'MultinomialNB': MultinomialNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'conf_matrix': confusion_matrix(y_test, preds),
        'roc_auc': auc(*roc_curve(y_test, probs)[:2])
    }


for name, res in results.items():
    print(f"--- {name} ---")
    print(f"Accuracy: {res['accuracy']:.3f}")
    print(f"Precision: {res['precision']:.3f}")
    print(f"Recall: {res['recall']:.3f}")
    print(f"F1-score: {res['f1']:.3f}")
    print(f"ROC AUC: {res['roc_auc']:.3f}")

    sns.heatmap(res['conf_matrix'], annot=True, fmt='d')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, models[name].predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {res["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.show()

best_model = max(results.items(), key=lambda x: x[1]['f1'])[0]
print(f"En iyi model: {best_model}")
print(classification_report(y_test, models[best_model].predict(X_test), target_names=le.classes_))


if 'Sentiment_Polarity' in df.columns and 'Sentiment_Subjectivity' in df.columns:
    corr = df[['Sentiment_Polarity', 'Sentiment_Subjectivity']].corr()
    print("Korelasyon Matrisi:")
    print(corr)
    sns.heatmap(corr, annot=True)
    plt.title('Sentiment Polarity & Subjectivity Correlation')
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.histplot(df['Sentiment_Polarity'], bins=30, kde=True)
    plt.title('Distribution of Sentiment Polarity')
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.histplot(df['Sentiment_Subjectivity'], bins=30, kde=True)
    plt.title('Distribution of Sentiment Subjectivity')
    plt.show()

    sns.jointplot(data=df, x='Sentiment_Polarity', y='Sentiment_Subjectivity', kind='scatter', hue='Sentiment')
    plt.show()
