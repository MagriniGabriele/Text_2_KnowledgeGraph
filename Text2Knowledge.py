import os

from KnowledgeExtractor import MatcherExtractor, AlternativeMatcherExtractor
from TripleExporter import data_to_n3, data_to_graph
from Summarizer import PageRankSummarizer, ClusterSummarizer, KnowledgeBaseSummarizer

import argparse
from os.path import isdir, isfile

#
# alphabets = "([A-Za-z])"
# prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
# suffixes = "(Inc|Ltd|Jr|Sr|Co)"
# starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
# acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
# websites = "[.](com|net|org|io|gov)"
# digits = "([0-9])"
#
#
# def split_into_sentences(text):
#     text = " " + text + "  "
#     text = text.replace("\n", " ")
#     text = re.sub(prefixes, "\\1<prd>", text)
#     text = re.sub(websites, "<prd>\\1", text)
#     text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
#     if "..." in text:
#         text = text.replace("...", "<prd><prd><prd>")
#     text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
#     text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
#     text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
#     text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
#     text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
#     text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
#     text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
#     print(text)
#     if "”" in text:
#         text = text.replace(".”", "”.")
#     if "\"" in text:
#         text = text.replace(".\"", "\".")
#     if "!" in text:
#         text = text.replace("!\"", "\"!")
#     if "?" in text:
#         text = text.replace("?\"", "\"?")
#     text = text.replace(",", ".<stop>")  # AGGIUNTA VIRGOLA COME PUNTO STOP
#     text = text.replace(".", ".<stop>")
#     text = text.replace("?", "?<stop>")
#     text = text.replace("!", "!<stop>")
#     print(text)
#     text = text.replace("<prd>", ".")
#     print(text)
#     sentences = text.split("<stop>")
#     sentences = sentences[:-1]
#     sentences = [s.strip() for s in sentences]
#     return sentences
#
#
# def get_entities(sent):
#     ## chunk 1
#     ent1 = ""
#     ent2 = ""
#
#     prv_tok_dep = ""  # dependency tag of previous token in the sentence
#     prv_tok_text = ""  # previous token in the sentence
#
#     prefix = ""
#     modifier = ""
#
#     #############################################################
#
#     for tok in nlp(sent):
#         ## chunk 2
#         # if token is a punctuation mark then move on to the next token
#         if tok.dep_ != "punct":
#             # check: token is a compound word or not
#             if tok.dep_ == "compound":
#                 prefix = tok.text
#                 # if the previous word was also a 'compound' then add the current word to it
#                 if prv_tok_dep == "compound":
#                     prefix = prv_tok_text + " " + tok.text
#
#             # check: token is a modifier or not
#             if tok.dep_.endswith("mod") == True:
#                 modifier = tok.text
#                 # if the previous word was also a 'compound' then add the current word to it
#                 if prv_tok_dep == "compound":
#                     modifier = prv_tok_text + " " + tok.text
#
#             ## chunk 3
#             if tok.dep_.find("subj") == True:
#                 ent1 = modifier + " " + prefix + " " + tok.text
#                 prefix = ""
#                 modifier = ""
#                 prv_tok_dep = ""
#                 prv_tok_text = ""
#
#                 ## chunk 4
#             if tok.dep_.find("obj") == True:
#                 ent2 = modifier + " " + prefix + " " + tok.text
#
#             ## chunk 5
#             # update variables
#             prv_tok_dep = tok.dep_
#             prv_tok_text = tok.text
#     #############################################################
#
#     return [ent1.strip(), ent2.strip()]
#
#
# def get_relation(sent):
#     doc = nlp(sent)
#
#     # Matcher class object
#     matcher = Matcher(nlp.vocab)
#
#     # define the pattern
#     pattern = [{'DEP': 'ROOT'},
#                {'DEP': 'prep', 'OP': "?"},
#                {'DEP': 'agent', 'OP': "?"},
#                {'POS': 'ADJ', 'OP': "?"}]
#
#     matcher.add("matching_1", None, pattern)
#
#     matches = matcher(doc)
#     k = len(matches) - 1
#     # ema: se non trova match qua muore
#     if k >= 0:
#         span = doc[matches[k][1]:matches[k][2]]
#         return (span.text)
#     else:
#         return ""
#
#
# def draw_kg(pairs):
#     k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',
#                                       create_using=nx.MultiDiGraph())
#     node_deg = nx.degree(k_graph)
#     layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
#     plt.figure(num=None, figsize=(120, 90), dpi=80)
#     nx.draw_networkx(
#         k_graph,
#         node_size=[int(deg[1]) * 500 for deg in node_deg],
#         arrowsize=20,
#         linewidths=1.5,
#         pos=layout,
#         edge_color='red',
#         edgecolors='black',
#         node_color='white',
#     )
#     labels = dict(zip(list(zip(pairs.subject, pairs.object)),
#                       pairs['relation'].tolist()))
#     nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,
#                                  font_color='red')
#     plt.axis('off')
#     plt.show()
#


if __name__ == '__main__':
    """
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

    tekken_document = open("./documents/tekken/tekken.txt", "r")
    text = ""
    for line in tekken_document.readlines():
        text += line
    doc = nlp(text)
    text = doc._.coref_resolved
    doc = nlp(text)
    entity_pairs = []
    new_text = str(doc)
    sentences = split_into_sentences(new_text)

    for sent in sentences:
        entity_pairs.append(get_entities(sent))

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
    # plt.show()
    plt.savefig(fname="./documents/tekken/tekken.png", format="png")
    if not len(entity_pairs) == len(source) == len(target):
        print("Unconsistent triple informations")
    else:
        triples = []
        for i in range(len(relations)):
            triples.append((
                source[i],
                relations[i],
                target[i]
            ))
        kb_summarizer = KnowledgeBaseSummarizer("./documents/tekken", 5, triples)
        kb_summarizer.summarize()
        print(kb_summarizer)
    """

    parser = argparse.ArgumentParser(description='Read the text from a given text file (if the input is a folder all '
                                                 'files within are processed), extracting information and providing a '
                                                 'summary when requested. The output files will have the original name'
                                                 'and will be located inside PATH (or the folder containing it)')
    parser.add_argument("-v", "--verbose", type=bool, default=False,
                        help="Print also the intermediate step of the process")
    parser.add_argument("target", nargs="+", type=str,
                        help="A path to a .txt file ora a folder", metavar="PATH")
    parser.add_argument("--base-uri", nargs=1, type=str, default="https://www.disit.dinfo.it",
                        help="Base URI used during the triple creation, default 'https://www.disit.dinfo.it'",
                        metavar="BASE_URI")
    parser.add_argument("--separator", nargs=1, type=str, default="/",
                        help="Separator between base URI and element name, default '/'", metavar="SEP")
    parser.add_argument("--extraction-method", nargs=1, type=int, default=0, choices=[0, 1],
                        help="The method used during information extraction:\n 0 -> Matcher\n | 1 -> Other",
                        metavar="EXTRACTION_MODE")
    parser.add_argument("--show-plot", type=bool, default=False,
                        help="If specified show the plot of the Knowledge Graph")
    parser.add_argument("--save-plot", type=bool, default=False,
                        help="If specified save the plot of the Knowledge Graph in PNG format")
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--no-summarize', dest='summarize', action='store_false')
    parser.set_defaults(summarize=True)
    '''parser.add_argument("--summarize", type=bool, default=True,
                        help="Supply this argument will cause the program to print the summary in the standard output."
                             " Using this flag with the verbose flag will mix log and results, use with caution")
    '''
    parser.add_argument("--number-of-sentences", nargs=1, default=5, type=int,
                        help="Number of sentences t be extracted, if the number is greater then the number of sentences"
                             "inside the original text half of the total sentences are extracted", metavar="N")
    parser.add_argument("--summarization-method", nargs=1, type=int, default=0, choices=[0, 1, 2],
                        help="The method used for creating the summary:\n 0 -> PageRank\n | 1 -> Clustering\n | 2 ->"
                             "Knowledge Base data parsing", metavar="SUMMARIZATION_MODE")
    parser.add_argument("--score", "-s", type=bool, default=False, help="Perform a benchmark using a sample text")

    args = parser.parse_args()

    if not args.score:
        files = []
        extractor = MatcherExtractor(verbose=args.verbose) if args.extraction_method == 0 \
            else AlternativeMatcherExtractor(verbose=args.verbose)
        summarizer = None
        last_text = ""
        for entry in args.target:
            # ogni caretella dovrebbe avere documenti affini

            # triplificazione
            triples = [[], [], []]
            if isdir(entry):
                for file in os.listdir(entry):
                    if file.endswith(".txt"):
                        temp_triples = extractor.parse_from_file(entry + os.sep + file, verbose=args.verbose)
                        triples[0] += temp_triples[0]
                        triples[1] += temp_triples[1]
                        triples[2] += temp_triples[2]
            elif isfile(entry) and entry.endswith(".txt"):
                temp_triples = extractor.parse_from_file(entry, verbose=args.verbose)
                triples[0] += temp_triples[0]
                triples[1] += temp_triples[1]
                triples[2] += temp_triples[2]
            else:
                print("Error: Cannot find a text file")
                exit(-1)
            if args.show_plot or args.save_plot:
                data_to_graph(triples, show=args.show_plot, save=args.save_plot, file_name=entry)
            data_to_n3(triples, entry, uri_prefix=args.base_uri, uri_separator=args.separator)
            if args.summarize:
                if args.summarization_method == 0:
                    summarizer = PageRankSummarizer(entry, args.number_of_sentences)
                elif args.summarization_method == 1:
                    summarizer = ClusterSummarizer(entry, args.number_of_sentences)
                else:
                    summarizer = KnowledgeBaseSummarizer(entry, args.number_of_sentences, triples)

                summarizer.summarize()
                print(summarizer)
    else:

        # extraction test battery:
        print("Begin Extraction Benchmark")

        text_path = "documents" + os.sep + "mario draghi"
        matcher = MatcherExtractor()
        matcher_alt = AlternativeMatcherExtractor()
        print("Standard Matcher")
        for file in os.listdir(text_path):
            if not file.endswith(".txt"):
                continue
            print("Parsing File ", file)
            matcher.report_from_file(text_path + os.sep + file)
        print("Alternative Matcher")
        for file in os.listdir(text_path):
            if not file.endswith(".txt"):
                continue
            print("Parsing File ", file)
            matcher_alt.report_from_file(text_path + os.sep + file)

        # summarization test battery:

        print("Begin Summarization benchmark")

        print("PageRank Summarizer")
        pagerank = PageRankSummarizer(text_path, top_n=5, nlp_pipe=None)
        pagerank_coref = PageRankSummarizer(text_path, top_n=5, nlp_pipe=matcher.nlp)

        pagerank.summarize()
        pagerank_coref.summarize()

        pagerank.report(matcher)
        pagerank_coref.report(matcher)
        # pagerank.report(matcher_alt)
        # pagerank_coref.report(matcher_alt)

        print("Cluster Summarizer")
        cluster = ClusterSummarizer(text_path, top_n=10)  # leggo più documenti, cerco più frasi
        cluster_coref = ClusterSummarizer(text_path, top_n=10,
                                          nlp_pipe=matcher.nlp)  # leggo più documenti, cerco più frasi
        cluster.summarize()
        cluster_coref.summarize()

        cluster.report(matcher)
        cluster_coref.report(matcher)
        # cluster.report(matcher_alt)
        # cluster_coref.report(matcher_alt)

        print("KnowledgeBase Summarizer")
        for file in os.listdir(text_path):
            if not file.endswith(".txt"):
                continue
            print("Summarizing ", file)
            triples = matcher.parse_from_file(text_path + os.sep + file)
            triples_alt = matcher_alt.parse_from_file(text_path + os.sep + file)

            kb = KnowledgeBaseSummarizer(text_path, triples=triples, top_n=5, nlp_pipe=matcher.nlp)
            kb_alt = KnowledgeBaseSummarizer(text_path, triples=triples_alt, top_n=5, nlp_pipe=matcher_alt.nlp)

            kb.summarize()
            kb_alt.summarize()

            kb.report(matcher)
            # kb_alt.report(matcher_alt)
    #
    # me = MatcherExtractor(verbose=args.verbose)
    # for file in files:
    #     triples = me.parse_from_file(file, verbose=args.verbose)
    #     if args.show_plot or args.save_plot:
    #         data_to_graph(triples, show=args.show_plot, save=args.save_plot, file_name=file)
    #     data_to_n3(triples, file, uri_prefix=args.uri_prefix, uri_separator=args.uri_separator)
    #     if args.summarize:
    #         strategies = {
    #             0: PageRankSummarizer(file, args.number_of_sentences),
    #             1: ClusterSummarizer(file, args.number_of_sentences),
    #             2: KnowledgeBaseSummarizer(file, args.number_of_sentences)
    #         }
