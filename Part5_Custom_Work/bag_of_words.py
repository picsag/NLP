import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

df = pd.read_csv("spam.csv")
print(df.head())

print(df['Category'].value_counts())

df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'], test_size=0.2)

print(X_train.shape)
print(X_test.shape)

print(X_train[:4])

v = CountVectorizer()

X_train_cv = v.fit_transform(X_train.values)

vectorized = X_train_cv.toarray()

print(vectorized)

shape = X_train_cv.shape

print(f"Vocabulary has size of {shape[1]} words")

print(v.get_feature_names_out()[1100:1200])

# for a list of methods of a class, the 'dir' command can be used:
print(dir(v))

print(v.vocabulary_)

# convert features to numpy array
X_train_np = X_train_cv.toarray()

# the first 4 emails:
print(X_train_np[0:4])

# get the words indexes which are found in the first email
print(f"Indexes of words found in the first email: {np.where(X_train_np[0] != 0)}")

indexes = np.where(X_train_np[0] != 0)

print(f"Words in the first email: {v.get_feature_names_out()[indexes]}")

model = MultinomialNB()
model.fit(X_train_cv, y_train)

X_test_cv = v.transform(X_test)

y_pred = model.predict(X_test_cv)

print(classification_report(y_test, y_pred))

# build a pipeline (sequence of operations)
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X=X_train, y=y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
