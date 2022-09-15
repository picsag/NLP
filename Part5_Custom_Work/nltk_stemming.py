# Lematization is more performant than stemming
# as Spacy does not support stemming, only lemmatization, NLTK has to be used for stemming
import spacy
from nltk.stem import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS

stemmer = PorterStemmer()

words = ['eating', 'eats', 'eat', 'ate', 'adjustable', 'rafting', 'ability', 'meeting']

for word in words:
    print(word, " | ", stemmer.stem(word))

nlp = spacy.load("en_core_web_sm")

doc = nlp("eating eats eat ate adjustable rafting ability meeting")

for token in doc:
    print(token, " | ", token.lemma_)

print(nlp.pipe_names)

ar = nlp.get_pipe("attribute_ruler")
ar.add([[{"TEXT": "Bro"}], [{"TEXT": "Brod"}]], {"LEMMA": "Brother"})

doc2 = nlp("Hey Bro, how are you? Very well Brod")

for token in doc2:
    print(token, " | ", token.lemma_)

doc3 = nlp("Elon flew to Mars yesterday. He carried mashed potato with him")
for token in doc3:
    print(token, " | ", token.pos_, " | ", token.tag_, " | ", spacy.explain(token.tag_))

# print by excluding spaces and punctuation
for token in doc3:
    if token.pos_ in ["SPACE", "X", "PUNCT"]:
        print(token, " | ", token.pos_, " | ", token.tag_, " | ", spacy.explain(token.tag_))

count = doc.count_by(spacy.attrs.POS)

for k, v in count.items():
    print(doc.vocab[k].text, " | ", v)

# print by excluding spaces and punctuation
for ent in doc3.ents:
    print(ent, " | ", ent.text, " | ", spacy.explain(ent.label_))

print(f"Number of stop words in spacy: {STOP_WORDS}")


