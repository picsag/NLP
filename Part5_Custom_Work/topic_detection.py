import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")


def remove_stop_words(text):
    doc = nlp(text)
    no_stop_words = [token.text for token in doc if not token.is_stop]
    return no_stop_words


print(remove_stop_words("Elon flew to Mars yesterday. He carried mashed potato with him"))



