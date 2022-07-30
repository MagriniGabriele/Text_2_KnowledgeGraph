from math import log10
from Document import Document
from os import listdir, sep
import fitz


class SummarizerTest:

    @staticmethod
    def _get_text(path: str) -> (str, str):
        text = ""
        title = ""
        extension = path.split('.')[-1]
        if extension == "txt":
            document = open(path, "r")
            # in text files the title is assumed to be the file name
            title = path.split('/')[-1].split('.')[-1].replace("_", " ")
            for line in document:
                text += line
        elif extension == "pdf":
            document = fitz.open(path)
            title = document.title
            for page in document:
                text += page.get_text()
        else:
            raise Exception("Cannot determine the file type")
        return text

    def __init__(self, nlp, document_folder):
        self.IDF = dict()  # term -> term idf score
        self.terms = self.IDF.keys()
        self.nlp = nlp
        self.documents: list[Document] = list()
        for file in listdir(document_folder):
            text = ""
            title = ""
            try:
                text, title = self._get_text(document_folder + sep + file)
            except Exception as error:
                print(file, error)
            self.documents.append(Document(nlp, text, file, title))
        self.get_terms()
        self.compute_scores()

    def get_terms(self):
        for doc in self.documents:
            self.terms |= doc.terms

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
            doc.compute_tfidf(self.IDF)

    def print(self):
        for i in range(len(self.documents)):
            print("\n\nDocument\t", self.documents[i].name)
            print("Term\tTerm tfidf score")
            self.documents[i].print_keywords()

