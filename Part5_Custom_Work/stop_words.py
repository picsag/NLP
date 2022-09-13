import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


nlp = spacy.load("en_core_web_sm")


def remove_stop_words(text):
    doc = nlp(text)
    no_stop_words = [token.text for token in doc if not token.is_stop]
    return no_stop_words


print(remove_stop_words("Elon flew to Mars yesterday. He carried mashed potato with him"))

# reading the

df = pd.read_json("combined.json", lines=True)

print(df.shape)

print(df.head(5))

df = df[df["topics"].str.len()!=0]

print(df.shape)

df = df.head(100)

# removing the stop words
df["contents_new"] = df["contents"].apply(remove_stop_words)

print(df["contents"].iloc[4])
print("=====================")
print(df["contents_new"].iloc[4])
