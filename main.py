from Summarizer import PageRankSummarizer, ClusterSummarizer, KnowledgeBaseSummarizer
import spacy
import neuralcoref
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy

from spacy.matcher import Matcher
from spacy.tokens import Span

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "..." in text:
        text = text.replace("...", "<prd><prd><prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    print(text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(",", ".<stop>")  # AGGIUNTA VIRGOLA COME PUNTO STOP
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    print(text)
    text = text.replace("<prd>", ".")
    print(text)
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]


def get_relation(sent):
    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    # define the pattern
    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': "?"},
               {'DEP': 'agent', 'OP': "?"},
               {'POS': 'ADJ', 'OP': "?"}]

    matcher.add("matching_1", None, pattern)

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]

    return (span.text)


def draw_kg(pairs):
    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',
                                      create_using=nx.MultiDiGraph())
    node_deg = nx.degree(k_graph)
    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(120, 90), dpi=80)
    nx.draw_networkx(
        k_graph,
        node_size=[int(deg[1]) * 500 for deg in node_deg],
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='red',
        edgecolors='black',
        node_color='white',
    )
    labels = dict(zip(list(zip(pairs.subject, pairs.object)),
                      pairs['relation'].tolist()))
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,
                                 font_color='red')
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    # Caricamento modello inglese web per spacy
    nlp = spacy.load('en_core_web_sm')

    # Settaggio NeuralCoref
    neuralcoref.add_to_pipe(nlp)

    # Testo di esempio
    text = "Dr.Emanuele hates Lollo.He also sucks dicks.Lollo eats strawberries."
    doct = nlp(text)

    # Risoluzione coreferenza
    resolved_doc = doct._.coref_resolved
    print(resolved_doc + "\n")
    text1 = nlp(resolved_doc)

    # Named entity recognition COREFERENZA NON RISOLTA
    # for ent in doct.ents:
    # print(f"Named Entity '{ent.text}' with label '{ent.label_}'")

    # Named entity recognition COREFERENZA RISOLTA
    # for word in text1.ents:
    # print(f"Named Entity '{word.text}' with label '{word.label_}'")

    # for sent in text1.sents:
    # print(sent)

    entity_pairs = []
    new_text = str(text1)
    sentences = split_into_sentences(new_text)

    print("Sentenze:")
    print(sentences)
    print("\n")

    for sent in sentences:
        entity_pairs.append(get_entities(sent))

    print("Entità:")
    print(entity_pairs)
    print("\n")

    relations = [get_relation(i) for i in tqdm(sentences)]
    # extract subject
    source = [i[0] for i in entity_pairs]

    # extract object
    target = [i[1] for i in entity_pairs]

    kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})

    # create a directed-graph from a dataframe
    G = nx.from_pandas_edgelist(kg_df, "source", "target",
                                edge_attr=True, create_using=nx.MultiDiGraph())

    # G = nx.from_pandas_edgelist(kg_df[kg_df['edge'] == "won"], "source", "target",
    #                            edge_attr=True, create_using=nx.MultiDiGraph())

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5)  # k regulates the distance between nodes
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=pos)
    plt.show()


    print("PageRank Summarization - Carbon Lang ---------------------------------\n")
    pr_summarizer = PageRankSummarizer("./documents/carbon", 5, nlp)
    pr_summarizer.summarize()

    print(pr_summarizer)

    print("\n\nCluster Summarization - Mario Draghi ------------------------------\n")
    c_summarizer = ClusterSummarizer("./documents/mario draghi", 5)
    c_summarizer.summarize()
    print(c_summarizer)

    print("\n\nKnowledge Base Summarization - Mario Draghi ------------------------------\n")
    kb_summarizer = KnowledgeBaseSummarizer("./documents/mario draghi", 5,
                                           [])
    kb_summarizer.summarize()
    print(kb_summarizer)
