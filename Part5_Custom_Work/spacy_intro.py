import spacy
from spacy.symbols import ORTH

# Spacy is an OOP NLP package, unlike NLTK, which is more traditional function-based NLP package

nlp = spacy.blank("en")  # an NLP "Object" for English language

corpus = "Tesla first built an electric sports car, the Roadster, in 2008. With sales of about 2,500 vehicles, " \
         "it was the first serial production all-electric car to use lithium-ion battery cells."

doc = nlp(corpus)

for token in doc:
    print(token)

with open("emails.txt") as f:
    text = f.readlines()
print(text)

text = ' '.join(text)
print(text)

doc2 = nlp(text=text)

emails = []
for token in doc2:
    if token.like_email:
        emails.append(token)

print(emails)

# Customize a jargon:
jargon = "Please gimme a slice of healthy pizza"

nlp.tokenizer.add_special_case("gimme", [
    {ORTH: "gim"},
    {ORTH: "me"}
])

doc3 = nlp(jargon)
tokens = [token for token in doc3]
print(tokens)


