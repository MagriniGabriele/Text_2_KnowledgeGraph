import spacy
from spacy.lang.en import English
from spacy.lang.it import Italian

from Summarizer import Summarizer

if __name__ == '__main__':

    # Caricamento modello inglese web per spacy
    nlp = Italian()
    sm = Summarizer(nlp, "./documents")
    sm.print()

