import os

from KnowledgeExtractor import MatcherExtractor, AlternativeMatcherExtractor
from TripleExporter import data_to_n3, data_to_graph
from Summarizer import PageRankSummarizer, ClusterSummarizer, KnowledgeBaseSummarizer

import argparse
from os.path import isdir, isfile

if __name__ == '__main__':
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
        matcher = MatcherExtractor(verbose=True)
        matcher_alt = AlternativeMatcherExtractor(verbose=True)
        print("Standard Matcher")
        for file in os.listdir(text_path):
            if not file.endswith(".txt"):
                continue
            triples = [[], [], []]
            print("Parsing File ", file)
            matcher.report_from_file(text_path + os.sep + file)
            temp_triples = matcher.parse_from_file(text_path + os.sep + file)
            triples[0] += temp_triples[0]
            triples[1] += temp_triples[1]
            triples[2] += temp_triples[2]
            data_to_n3(temp_triples, text_path + os.sep + "output" + os.sep + file.replace(".txt", "_matcher.n3"))
            data_to_graph(temp_triples, text_path + os.sep + "output" + os.sep + file.replace(".txt", "_matcher_plot.png"), show=False, save=True)
        print("Alternative Matcher")
        for file in os.listdir(text_path):
            if not file.endswith(".txt"):
                continue
            triples = [[], [], []]
            print("Parsing File ", file)
            matcher_alt.report_from_file(text_path + os.sep + file)
            temp_triples = matcher_alt.parse_from_file(text_path + os.sep + file)
            triples[0] += temp_triples[0]
            triples[1] += temp_triples[1]
            triples[2] += temp_triples[2]
            data_to_n3(temp_triples, text_path + os.sep + "output" + os.sep + file.replace(".txt", "_matcher_alt.n3"))
            data_to_graph(temp_triples, text_path + os.sep + "output" + os.sep + file.replace(".txt", "_matcher_alt_plot.png"), show=False, save=True)

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
