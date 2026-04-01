import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# LOAD DATA
df = pd.read_csv("data/sellercaption.csv")

df = df[['caption','label']]
df = df.dropna()

print("Dataset Shape:", df.shape)

# Fix label spacing
df['label'] = df['label'].str.strip()

# Convert labels
df['label'] = df['label'].map({'Fake':0,'Real':1})

# TEXT CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"[^a-zA-Z ]","",text)
    return text

df['caption'] = df['caption'].apply(clean_text)

print("\nLabel Distribution:")
print(df['label'].value_counts())


# SPLIT DATA
X = df['caption']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# SHOW TRAIN / TEST SIZE
print("\nTrain-Test Split")
print("Training Data Size:", X_train.shape[0])
print("Testing Data Size:", X_test.shape[0])

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)



# MODEL
model = LogisticRegression(max_iter=200)

model.fit(X_train_vec,y_train)

print("\nModel Used:", model)

# PREDICTION
y_pred = model.predict(X_test_vec)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# SAVE MODEL
joblib.dump(model,"model/model.pkl")
joblib.dump(vectorizer,"model/vectorizer.pkl")

print("Model saved successfully!")