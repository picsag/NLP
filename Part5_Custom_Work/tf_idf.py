import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import time

# loading ecommerce data
df = pd.read_csv("ecommerce_data.csv", encoding='unicode_escape')
print(df.shape)
print(f"Columns: {df.columns}")
print(df.head())

print(df['label'].value_counts())
mapping = {
    'Household': 0,
    'Books': 1,
    'Electronics': 2,
    'Clothing & Accessories': 3
}

df['label_num'] = df['label'].map(mapping)

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    df['Text'],
    df['label_num'],
    test_size=0.2,
    random_state=2022,
    stratify=df['label_num']
)

print(f"Shape of X_train {X_train.shape}")
print(f"Shape of X_test {X_test.shape}")

clf = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer()),
    ('kNN', KNeighborsClassifier())
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


clf2 = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer()),
    ('random_forest', RandomForestClassifier(n_estimators=15))
])

clf2.fit(X_train, y_train)

y_pred = clf2.predict(X_test)

print(classification_report(y_test, y_pred))

nlp = spacy.load("en_core_web_sm")

# preprocess function
def preprocess(text):
    doc = nlp(text)

    filtered_tokens = []

    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)


start = time.process_time()
print("Preprocessing . . .")
df['preprocessed_text'] = df['Text'].apply(preprocess)

end = time.process_time()
print(f"Time to preprocess dataset: {end - start}")

X_train, X_test, y_train, y_test = train_test_split(
    df['preprocessed_text'],
    df['label_num'],
    test_size=0.2,
    random_state=2022,
    stratify=df['label_num']
)


clf3 = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer()),
    ('random_forest', RandomForestClassifier(n_estimators=15))
])

clf3.fit(X_train, y_train)

y_pred = clf3.predict(X_test)

print(classification_report(y_test, y_pred))
