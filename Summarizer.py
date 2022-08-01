import os
from abc import ABC

import pandas as pd
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from typing import List

from Document import Document
import os
import numpy as np
import networkx as nx


class Summarizer(ABC):
    """
    Classe base per il summarizer, espone i metodi per effettuare la summarization e
    per mostrare i risultati con la parte estratta evidenziata.
    Non implementa la logica vera e proprio, quindi usare una delle sue implementazioni
    """

    def __init__(self, document_dir: str, top_n: int):
        """
        Costruttore, carica i testi in memoria, ma non effettua la summarization
        :param document_dir: i testi sono estratti dalla directory indicata, letti da un file di testo
        :param top_n: numero di frasi significative da estrarre: se il documento presenta un numero i
        nferiore di frasi, ne vengono estratte la metà
        """
        def _load_text(path: str):
            texts = list()
            for file in os.listdir(path):
                if not file.endswith(".txt"):
                    # ignora file non testuali
                    continue
                document = open(path + os.sep + file, "r")
                texts.append("".join(document.readlines()))
            return texts

        self.texts = _load_text(document_dir)
        self.summaries = list()
        self.top_n = top_n

    def summarize(self):
        raise NotImplementedError

    def __str__(self) -> str:
        if len(self.texts) == 0:
            return "There are no text to be summarized"
        if len(self.texts) != len(self.summaries):
            return "The summarization has not been performed yet"
        text = ""
        for i in range(len(self.texts)):
            original_sents = self.read_document(i)
            extracted_sents = self.read_document(i, self.summaries)
            text += "Text #" + str(i + 1) + ":\n"
            text = ""
            for sentence in original_sents:
                if sentence in extracted_sents:
                    text += "\033[1;31;49m" + " ".join(sentence) + ". "
                else:
                    text += "\033[1;38;49m" + " ".join(sentence) + ". "
            return text
        # for i in range(len(self.texts)):
        #     print(
        #         "Text #", i + 1, ":\n",
        #         self.texts[i], "\n"
        #                        "Summary #", i + 1, ":\n",
        #         self.texts[i], "\n\n"
        #     )

    def read_document(self, file_index, text_list: List[str] = None):
        """
        effettua il parsing dei testi o dei riassunti caricati
        :param file_index: indice del testo all'interno del vettore dei testi
        :param text_list: vettore dei testi. se non specificato è il vettore dei testi originali
        :return:
        """
        if text_list is None:
            text_list = self.texts
        sentences = list()
        text = text_list[file_index]
        split = text.split(".")
        for sentence in split:
            sentences.append(sentence.strip().replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop()

        return sentences


class PageRankSummarizer(Summarizer):

    def __init__(self, document_dir: str, top_n: int = 5):
        super().__init__(document_dir, top_n)

    def summarize(self):
        self.summaries = [""] * len(self.texts)
        for i in range(len(self.texts)):
            self.summaries[i] = self.generate_summary(i, self.top_n)

    @staticmethod
    def __sentence_similarity(sentence1: List[str], sentence2: List[str], stop_words=stopwords):
        """
        compara due frasi dopo averle proiettate in uno spazio vettoriale
        :param sentence1: una lista di stringhe, ciascuna rappresenta una parola della frase
        :param sentence2: una lista di stringhe, ciascuna rappresenta una parola della frase
        :param stop_words: porle comuni da ignorare
        :return: la similarità delle due frasi [0,1]
        """

        def to_vector_space(sentence: List[str], word_set: List[str], stop_word_list) -> List[int]:
            """
            mappa una frase in uno spazio vettoriale, modellato come un beg of words + cardinalità
            :param sentence: una lista di stringhe, ciascuna rappresenta una parola della frase
            :param word_set: l'insieme di parole dello spazio vettoriale
            :param stop_word_list: porle comuni da ignorare
            :return: il vettore corrispondente alla frase
            """
            vector = [0] * len(word_set)
            for word in sentence:
                if word in stop_word_list:
                    continue
                vector[word_set.index(word)] += 1
            return vector

        # make sentences lowercase
        sentence1 = [word.lower() for word in sentence1]
        sentence2 = [word.lower() for word in sentence2]

        # the words outside the union set can be ignored, each one of the sentences has a 0 value there
        all_words = list(set(sentence1 + sentence2))

        # allocate the sentences in a vector space

        sentence1_vec = to_vector_space(sentence1, all_words, stop_words)
        sentence2_vec = to_vector_space(sentence2, all_words, stop_words)

        return 1 - cosine_distance(sentence1_vec, sentence2_vec)

    @staticmethod
    def __similarity_matrix(sentences: List[List[str]], stop_words):
        """
        genera una mtrice quadrata dove l'elemento ij è il fattore di similarità della frase i-esima con la
        frase j-esima
        :param sentences: lista delle frasi, ciascuna rappresentanta come la lista delle parole della frase stessa
        :param stop_words: parole comuni da ignorare
        :return:
        """
        sm = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i == j:
                    continue
                sm[i][j] = PageRankSummarizer.__sentence_similarity(sentences[i], sentences[j], stop_words)
        return sm

    def generate_summary(self, file_index: int, top_n: int = 5):
        """
        crea il riassunto del testo usando l'algoritmo di page rank
        :param file_index: indice del testo da riassumere
        :param top_n: numero di frasi da estrarre
        :return:
        """
        summarized_sentences = list()
        stop_words = stopwords.words("english")

        sentences = self.read_document(file_index)
        sentence_similarity_matrix = PageRankSummarizer.__similarity_matrix(sentences, stop_words)
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
        scores = nx.pagerank(sentence_similarity_graph)

        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        print("Ranked sentences: ", ranked_sentences)
        for i in range(top_n if top_n < len(sentences) else len(sentences)/2 + 1):
            summarized_sentences.append(" ".join(ranked_sentences[i][1]))
        return ". ".join(summarized_sentences)
#
#
# class DocWrapper:
#
#     def __init__(self, text: str, title: str):
#         self.title = title
#         self.text = text.lower()
#         for stopword in stopwords.word("english"):
#             self.text = self.text.replace(stopword, "")
#         self.sentences = text.split(".")
#
#
# class SuperviseSummarizer:
#
#     def __init__(self):
#         self.documents = list()
#
#     def load_docs(self, document_folder: str = "./documents"):
#         for file in listdir(document_folder):
#             text = ""
#             title = ""
#             try:
#                 text, title = self._get_text(document_folder + sep + file)
#             except Exception as error:
#                 print(file, error)
#             self.documents.append(Document(self.nlp, text, file, title))
#         self.get_terms()
#         self.compute_scores()
