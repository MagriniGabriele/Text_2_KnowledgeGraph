class Document:

    def __init__(self, nlp, text):

        self.text = text
        self.TFIDF = dict()
        self.terms = self.TFIDF.keys()