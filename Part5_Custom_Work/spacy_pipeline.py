import spacy
from spacy import displacy

nlp0 = spacy.blank("en")  # a blank NLP "Object" for English language, where blank = only tokenization

corpus = "Tesla first built an electric sports car, the Roadster, in 2008. With sales of about 2,500 vehicles, " \
         "it was the first serial production all-electric car to use lithium-ion battery cells."

doc = nlp0(corpus)

for token in doc:
    print(token)

#  The pipe_names will be an empty list
print(nlp0.pipe_names)

#  In order to apply other operations apart from tokenization, a Spacy object has to be created from additional resource
#  RUN python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")

print(nlp.pipeline)

doc2 = nlp(corpus)

print("============PARTS OF SPEECH===============================")

# print parts of speech
for token in doc2:
    print(token, " | ", token.pos_, " | ", token.lemma)

print("============NAMED ENTITIES================================")

# print named entities:
for ent in doc2.ents:
    print(ent.text, " | ", ent.label_)

displacy.serve(doc2, style="ent", host="127.0.0.1")  # use displacy.render for Jupyter Notebooks

#  it is also possible to add a part of a pipeline to a blank pipe:
nlp0.add_pipe("ner", source=nlp)
print(nlp.pipe_names)

