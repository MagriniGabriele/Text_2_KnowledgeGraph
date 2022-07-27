from math import log10


class Summarizer:

    def __init__(self):
        self.IDF = dict()  # term -> term idf score
        self.terms = self.IDF.keys()
        self.documents = list()

    def get_keywords(self):
        for doc in self.documents:
            self.terms |= self.documents.terms

    def _get_term_frequency(self, term: str):
        frequency = 0
        for doc in self.documents:
            frequency += int(doc.contains(term))
        return frequency

    def compute_scores(self):
        # step 1: compute idf
        for term in self.terms:
            self.IDF[term] = log10(len(self.documents) / self._get_term_frequency(term))
        # step 2: computer tf-idf
        for doc in self.documents:
            doc.compute_TF_IDF(self.IDF)
