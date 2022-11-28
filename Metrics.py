from typing import List


def information_density(triples, doc) -> float:
    """
    una metrica comparativa, ritorna la percentuale di triple trovate in relazione alla lunghezza (numero di parole)
    del documento
    :param triples:
    :param text:
    :return:
    """
    return len(triples[0]) / len(doc)


def usable_triples_density(triples, doc) -> float:
    count = 0
    for i in range(len(triples[0])):
        try:
            if triples[0][i] != "" and triples[1][i] != "" and triples[2][i] != "":
                count += 1
        except IndexError:
            pass
    return count / len(triples[0]) if count != 0 else 0


def usable_information_density(triples, doc):
    return usable_triples_density(triples, doc) * information_density(triples, doc)


def compression_ratio(original_text: str, summarized_text: str, nlp) -> float:
    """
    rende il rapporto tra la lunghezza del riassunto e la lunghezza del testo originale, ovviamente dipende
    dal numero di frasi da estrarre
    :param original_text:
    :param summarized_text:
    :param nlp:
    :return:
    """
    original_doc = nlp(original_text)
    summarized_doc = nlp(summarized_text)
    return 1 - len(summarized_doc) / len(original_doc)


def data_loss(original_text: str, summarized_text: str, nlp) -> float:
    """
    la percenuale di dati (entità) andate perse nel riassunto
    :param original_text:
    :param summarized_text:
    :param nlp:
    :return:
    """
    original_doc = nlp(original_text)
    summarized_doc = nlp(summarized_text)
    return 1 - len(summarized_doc.ents) / len(original_doc.ents)


def synthesis_score(original_text: str, summarized_text: str, nlp) -> float:
    """
    combinazione dei due score precedenti
    :param original_text:
    :param summarized_text:
    :param nlp:
    :return:
    """
    return compression_ratio(original_text, summarized_text, nlp) * (1 - data_loss(original_text, summarized_text, nlp))


def extractive_comparison(original_text: str, summarized_text: str, extractor):
    triples = extractor.parse(original_text)
    print("Original document usable information density is ",
          usable_information_density(triples, extractor.nlp(original_text)))
    triples = extractor.parse(summarized_text)
    print("Summarized document usable information density is ",
          usable_information_density(triples, extractor.nlp(summarized_text)))


def summarization_information_loss(original_text: str, summarized_text: str, extractor) -> float:
    original_text_n_triples = len(extractor.parse(original_text))
    summarized_text_n_triples = len(extractor.parse(summarized_text))
    return summarized_text_n_triples / original_text_n_triples


def summarization_compression(original_text: List[str], summarized_text: List[str]) -> float:
    return len(summarized_text) / len(original_text)


def mixed_metric(original_text: List[str], summarized_text: List[str], extractor) -> float:
    return summarization_information_loss(" ".join(original_text), " ".join(summarized_text), extractor) * \
        summarization_compression(original_text, summarized_text)


def gt_metric(gt_relations: int, found_relations: int) -> float:
    return found_relations/gt_relations
