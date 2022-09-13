import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import time

corpus = [
    "Batman drinks coffee",
    "Robin is short",
    "Robin is drinking coffee "
]

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


print(preprocess("Robin is drinking coffee"))

processed_corpus = [preprocess(text) for text in corpus]

# build vocabolary for unigram and bi-gram tokens
v = CountVectorizer(ngram_range=(1, 2))

v.fit(processed_corpus)

print(v.vocabulary_)

# In order to convert the text to numerical values (features), the transform function of the trained vectorizer is used
feat = v.transform(["Batman drinks coffee"])

print(f"Feature for 'Batman drinks coffee' = {feat.toarray()}")

# importing news category classification dataset from Kaggle
start = time.process_time()
df = pd.read_json("News_Category_Dataset_v2.json", lines=True)
end = time.process_time()
print(f"Time to read news dataset: {end - start}")

print(df.shape)

print(df.head(5))

print(df.columns)

# an imbalanced dataset:
print(df["category"].value_counts())

# to tackle imbalanced dataset:
min_samples = 1509
df_fifty = df[df["category"] == "FIFTY"].sample(n=min_samples, random_state=2022, replace=True)
df_good_news = df[df["category"] == "GOOD NEWS"].sample(n=min_samples, random_state=2022, replace=True)
df_arts_culture = df[df["category"] == "ARTS & CULTURE"].sample(n=min_samples, random_state=2022, replace=True)
df_environment= df[df["category"] == "ENVIRONMENT"].sample(n=min_samples, random_state=2022, replace=True)
df_college = df[df["category"] == "COLLEGE"].sample(n=min_samples, random_state=2022, replace=True)
df_latino_voices = df[df["category"] == "LATINO VOICES"].sample(n=min_samples, random_state=2022, replace=True)
df_culture_arts = df[df["category"] == "CULTURE & ARTS"].sample(n=min_samples, random_state=2022, replace=True)
df_education = df[df["category"] == "EDUCATION"].sample(n=min_samples, random_state=2022, replace=True)

df_balanced = pd.concat([df_fifty, df_good_news, df_arts_culture, df_environment, df_college, df_latino_voices,
                         df_culture_arts, df_education])

print(df_balanced["category"].value_counts())

target = {'FIFTY':0, 'GOOD NEWS': 1, 'ARTS & CULTURE': 2, 'ENVIRONMENT': 3, 'COLLEGE': 4, 'LATINO VOICES': 5,
          'CULTURE & ARTS': 6, 'EDUCATION': 7}

df_balanced['category_num'] = df_balanced['category'].map(target)

X_train, X_test, y_train, y_test = train_test_split(df_balanced["short_description"], df_balanced["category_num"],
                                                    test_size=0.2, random_state=2022,
                                                    stratify=df_balanced['category_num'])  #eq no of samples for classes

print(f"Size of train data: {X_train.shape[0]}")
print(X_train.head())

print(y_train.value_counts())

# using 1-gram Approach
clf = Pipeline([
    ('vectorizer_bow', CountVectorizer()),
    ('Multi NB', MultinomialNB())
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# using 2-gram BOW approach
clf2 = Pipeline([
    ('vectorizer_bow', CountVectorizer(ngram_range = (1, 2))),
    ('Multi NB', MultinomialNB())
])

clf2.fit(X_train, y_train)

y_pred = clf2.predict(X_test)

print(classification_report(y_test, y_pred))

# using 2-gram BOW approach
clf3 = Pipeline([
    ('vectorizer_bow', CountVectorizer(ngram_range=(1, 3))),
    ('Multi NB', MultinomialNB())
])

clf3.fit(X_train, y_train)

y_pred = clf3.predict(X_test)

print(classification_report(y_test, y_pred))

start = time.process_time()
df_balanced['preprocessed_text'] = df_balanced['short_description'].apply(preprocess)
end = time.process_time()
print(f"Time to preprocess dataset: {end - start}")

X_train, X_test, y_train, y_test = train_test_split(df_balanced["preprocessed_text"], df_balanced["category_num"],
                                                    test_size=0.2, random_state=2022,
                                                    stratify=df_balanced['category_num'])  #eq no of samples for classes

clf4 = Pipeline([
    ('vectorizer_bow', CountVectorizer(ngram_range=(1, 3))),
    ('Multi NB', MultinomialNB())
])

clf4.fit(X_train, y_train)

y_pred = clf4.predict(X_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.show()
