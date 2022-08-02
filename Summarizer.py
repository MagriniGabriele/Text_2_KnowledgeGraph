import math
import os
from abc import ABC
from builtins import staticmethod

import pandas as pd
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.cluster import kmeans
from gensim.models import Word2Vec

from typing import List, Tuple

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

    def __init__(self, document_dir: str, top_n: int, nlp_pipe = None):
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
                text = "".join(document.readlines())
                if nlp_pipe is not None:
                    try:
                        doc = nlp_pipe(text)
                        text = doc._.resolved_coref
                    except Exception as ex:
                        print(f"Coreference failed with error: \n{ex}\nUsing original text")
                texts.append(text)
            return texts

        self.texts = _load_text(document_dir)
        self.summaries = list()
        self.top_n = top_n

    def summarize(self):
        raise NotImplementedError

    def __str__(self) -> str:
        if len(self.texts) == 0:
            return "There are no text to be summarized"
        # if len(self.texts) != len(self.summaries):
        # return "The summarization has not been performed yet"
        text = ""
        for i in range(len(self.texts)):
            original_sents = self.read_document(i)
            extracted_sents = self.read_document(i, self.summaries)
            text += "\n\n\033[1;38;49mText #" + str(i + 1) + ":\n"
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

    def read_document(self, file_index, text_list: List[str] = None) -> List[List[str]]:
        """
        effettua il parsing dei testi o dei riassunti caricati
        :param file_index: indice del testo all'interno del vettore dei testi
        :param text_list: vettore dei testi. se non specificato è il vettore dei testi originali
        :return lista delle frasi, rappresentate come lista di parole:
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

    def __init__(self, document_dir: str, top_n: int = 5, nlp_pipe = None):
        super().__init__(document_dir, top_n, nlp_pipe)

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

        def to_vector_space(sentence: List[List[str]], word_set: List[str], stop_word_list) -> List[int]:
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
        for i in range(top_n if top_n < len(sentences) else len(sentences) / 2 + 1):
            summarized_sentences.append(" ".join(ranked_sentences[i][1]))
        return ". ".join(summarized_sentences)


class ClusterSummarizer(Summarizer):
    """
    uso del k-mean clustering per ottenere i k cluster (topic) più significativi del documento
    se un cluster viene sceleto la frase usata sarò quelle più simile alle altre
    mentre all'inserimento nel cluster, si cerca il centroide più viciono
    le frasi sono proiettate nello spazione vettoriale bag of words con cardinalità
    """

    def __init__(self, document_dir: str, top_n: 5, nlp_pipe = None):
        super().__init__(document_dir, top_n, nlp_pipe)
        self.sentences = []  # List[Tuple[List[str], List[float]]]
        self.word_set = set()
        for i in range(len(self.texts)):
            sentences = self.read_document(i)
            for sent in sentences:
                self.sentences.append((sent, []))
                self.word_set = self.word_set | set(sent)
        self.clusters = []

    @staticmethod
    def to_vector_space(sentence: List[List[str]], word_set: List[str], stop_word_list) -> List[int]:
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

    def create_clusters(self):
        """
        il kmeans è delegeto ad un implementazione già pronta,
        qui viene creata l'associazione frase-vettore che permette in seguito di risalire alla frase
        di partenza
        :return:
        """
        for i in range(len(self.sentences)):
            self.sentences[i] = (self.sentences[i][0], self.to_vector_space(self.sentences[i][0], list(self.word_set),
                                                                            stopwords.words("english")))

    def summarize(self):
        """
        per effettuare il riassunto si procede come segue
        - si crea lo spazioe vettoriale formato da tutte le parole presenti in tutti i documenti
        - si proiettano le frasi su tale spazio
        - ne si esegue il cluster con k-means, dove k-means è il numero di topic cercato
        - si cerca per ogni centroide la proiezione più vicina e si prende la frase associata
        :return:
        """

        def __closest_vector(target, candidates, distance_function):
            """

            :param target: vettore numpy bersaglio
            :param candidates: lista di vettorei numpy nella quale cercare il più viciono
            :param distance_function: funzione da usare per calcolare la metrica
            :return: il vettore più vicino
            """
            distance = math.inf
            result = None
            for candidate in candidates:
                d = distance_function(target, candidate)
                if d < distance:
                    distance = d
                    result = candidate
            return result

        self.create_clusters()

        clusterer = kmeans.KMeansClusterer(num_means=self.top_n, distance=cosine_distance, )
        vectors = []
        for sentences in self.sentences:
            vectors.append(np.array(sentences[1]))
        self.clusters = clusterer.cluster(vectors)
        self.summaries = []
        for mean in clusterer.means():
            sent_vector = __closest_vector(mean, vectors, cosine_distance)
            for sentence in self.sentences:
                if np.array_equal(np.array(sentence[1]), sent_vector):
                    sent = sentence[0]
                    self.summaries.append(" ".join(sent))
        self.summaries = ". ".join(self.summaries)

    def __str__(self) -> str:
        return self.summaries


class KnowledgeBaseSummarizer(Summarizer):
    """
    riassunto eseguito dopo aver estratto una knowledge base dal testo:
    l'idea è quella di scegliere il numero minimo di frasi in modo da coprire la maggiore quantità
    di nodi del knowledge graph
    """

    def __init__(self, documents_path: str, coverage: float, triples: List[Tuple[str, str, str]], nlp_pipe = None):
        super().__init__(documents_path, 0, nlp_pipe)  # top_n is not used
        self.coverage = coverage if 0 < coverage <= 1 else 0.5
        self.triples = triples

    def __score_sentence(self, sentence: List[str]):
        def __triple_match(sent: List[str], triple: Tuple[str, str, str]) -> int:
            """
            controlla quanto la tripla è appartentene alla frase, controllando quanti dei suoi elementi sono contenuti
            dentro la frase; dato che il processo di estrazione dei dati può modificare la stringa, non viene eseguito
            il match esatto, ma un .contains()
            :param sent:
            :param triple:
            :return numero di elementi della tripla presenti nella frase:
            """

            score = 0
            for word in sent:
                if word in triple[0]:
                    score += 1
                    continue
                if word in triple[1]:
                    score += 1
                    continue
                if word in triple[2]:
                    score += 1
                    continue
            return score

        sentence_score = 0
        for triple_to_match in self.triples:
            sentence_score += __triple_match(sentence, triple_to_match) / 3
        return sentence_score

    def __generate_summary(self, index) -> List[str]:
        sentences = self.read_document(index)
        scores = []
        result = list(list(str))
        for sent in sentences:
            scores.append(self.__score_sentence(sent))
        for r in sorted((scores, sentences), reverse=True):
            result.append(" ".join(r[1]))
        return result

    def summarize(self):
        self.summaries = [""] * len(self.texts)
        for i in range(len(self.texts)):
            sorted_sents = self.__generate_summary(i)
            self.summaries[i] = ". ".join(sorted_sents[0: self.top_n if self.top_n <= len(sorted_sents) else
                (len(sorted_sents) / 2) + 1])
