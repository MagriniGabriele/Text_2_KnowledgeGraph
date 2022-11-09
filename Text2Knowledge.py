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
    parser.add_argument("-v", "--verbose", default=False, action='store_true',
                        help="Print also the intermediate step of the process")
    parser.add_argument("target", nargs="+", type=str,
                        help="A path to a .txt file ora a folder", metavar="PATH")
    parser.add_argument("--output", type=str, default="./output",
                        help="Where dump the output",
                        metavar="OUTPUT_PATH")
    parser.add_argument("--base-uri", nargs=1, type=str, default="https://www.disit.dinfo.it",
                        help="Base URI used during the triple creation, default 'https://www.disit.dinfo.it'",
                        metavar="BASE_URI")
    parser.add_argument("--separator", nargs=1, type=str, default="/",
                        help="Separator between base URI and element name, default '/'", metavar="SEP")
    parser.add_argument("--extraction-method", nargs=1, type=int, default=0, choices=[0, 1, 2],
                        help="The method used during information extraction:\n 0 -> Matcher\n | 1 -> Other",
                        metavar="EXTRACTION_MODE")
    parser.add_argument("--show-plot", default=False, action='store_true',
                        help="If specified show the plot of the Knowledge Graph")
    parser.add_argument("--save-plot", default=False, action='store_true',
                        help="If specified save the plot of the Knowledge Graph in PNG format")
    parser.add_argument('--summarize', action='store_true', default=False)
    parser.add_argument("--number-of-sentences", nargs=1, default=5, type=int,
                        help="Number of sentences t be extracted, if the number is greater then the number of sentences"
                             "inside the original text half of the total sentences are extracted", metavar="N")
    parser.add_argument("--summarization-method", nargs=1, type=int, default=0, choices=[0, 1, 2],
                        help="The method used for creating the summary:\n 0 -> PageRank\n | 1 -> Clustering\n | 2 ->"
                             "Knowledge Base data parsing", metavar="SUMMARIZATION_MODE")
    parser.add_argument("--score", "-s", default=False, help="Perform a benchmark using a sample text",
                        action='store_true')
    parser.add_argument("--second-extraction", default=False, help="Perform a second extraction based on the previous one",
                        action='store_true')


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
                data_to_graph(triples, show=args.show_plot, save=args.save_plot, output_folder=args.output)
            data_to_n3(triples, output_folder=args.output, uri_prefix=args.base_uri, uri_separator=args.separator)
            if args.summarize:
                if args.summarization_method == 0:
                    summarizer = PageRankSummarizer(entry, args.number_of_sentences, nlp_pipe=extractor.nlp)
                elif args.summarization_method == 1:
                    summarizer = ClusterSummarizer(entry, args.number_of_sentences, nlp_pipe=extractor.nlp)
                else:
                    summarizer = KnowledgeBaseSummarizer(entry, args.number_of_sentences, triples)

                summarizer.summarize(output_folder=args.output)
                print(summarizer)
        if args.second_extraction and args.summarize:
            last_text = ""
            entries = [args.output + os.sep + "summary"]
            for entry in entries:
                # ogni caretella dovrebbe avere documenti affini

                # triplificazione
                triples = [[], [], []]
                if isdir(entry):
                    for file in os.listdir(entry):
                        if file.endswith(".txt"):
                            temp_triples = extractor.parse_from_file(entry + os.sep + file,
                                                                     verbose=args.verbose)
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
                if args.second_extraction and (args.show_plot or args.save_plot):
                    data_to_graph(triples, show=args.show_plot, save=args.save_plot, output_folder=args.output,
                                  prefix="second_")
                data_to_n3(triples, output_folder=args.output, uri_prefix=args.base_uri, uri_separator=args.separator,
                           prefix="second_")

    else:

        for entry in args.target:


            # extraction test battery:
            print("Begin Extraction Benchmark")

            matcher = MatcherExtractor(verbose=True)
            matcher_alt = AlternativeMatcherExtractor(verbose=True)
            print("Standard Matcher")
            for file in os.listdir(entry):
                if not file.endswith(".txt"):
                    continue
                triples = [[], [], []]
                print("Parsing File ", entry + os.sep + file)
                matcher.report_from_file(entry + os.sep + file)
                temp_triples = matcher.parse_from_file(entry + os.sep + file)
                triples[0] += temp_triples[0]
                triples[1] += temp_triples[1]
                triples[2] += temp_triples[2]
                data_to_n3(temp_triples, output_folder=args.output, prefix="standard_matcher_")
                data_to_graph(temp_triples, output_folder=args.output, prefix="standard_matcher_", show=False, save=True)
            print("Alternative Matcher")
            for file in os.listdir(entry):
                if not file.endswith(".txt"):
                    continue
                triples = [[], [], []]
                print("Parsing File ", file)
                matcher_alt.report_from_file(entry + os.sep + file)
                temp_triples = matcher_alt.parse_from_file(entry + os.sep + file)
                triples[0] += temp_triples[0]
                triples[1] += temp_triples[1]
                triples[2] += temp_triples[2]
                data_to_n3(temp_triples, output_folder=args.output, prefix="alternative_matcher_")
                data_to_graph(temp_triples, output_folder=args.output, prefix="alternative_matcher_", show=False, save=True)

            # summarization test battery:

            print("Begin Summarization benchmark")

            print("PageRank Summarizer")
            pagerank = PageRankSummarizer(entry, top_n=5, nlp_pipe=None)
            pagerank_coref = PageRankSummarizer(entry, top_n=5, nlp_pipe=matcher.nlp)

            pagerank.summarize(output_folder=args.output, prefix="pagerank_")
            pagerank_coref.summarize(output_folder=args.output, prefix="coref_pagerank_")

            pagerank.report(matcher)
            pagerank_coref.report(matcher)
            # pagerank.report(matcher_alt)
            # pagerank_coref.report(matcher_alt)

            print("Cluster Summarizer")
            cluster = ClusterSummarizer(entry, top_n=10)  # leggo più documenti, cerco più frasi
            cluster_coref = ClusterSummarizer(entry, top_n=10,
                                              nlp_pipe=matcher.nlp)  # leggo più documenti, cerco più frasi
            cluster.summarize(output_folder=args.output, prefix="cluster_")
            cluster_coref.summarize(output_folder=args.output, prefix="cluster_")

            cluster.report(matcher)
            cluster_coref.report(matcher)
            # cluster.report(matcher_alt)
            # cluster_coref.report(matcher_alt)

            print("KnowledgeBase Summarizer")
            for file in os.listdir(entry):
                if not file.endswith(".txt"):
                    continue
                print("Summarizing ", file)
                triples = matcher.parse_from_file(entry + os.sep + file)
                triples_alt = matcher_alt.parse_from_file(entry + os.sep + file)

                kb = KnowledgeBaseSummarizer(entry, triples=triples, top_n=5, nlp_pipe=matcher.nlp)
                kb_alt = KnowledgeBaseSummarizer(entry, triples=triples_alt, top_n=5, nlp_pipe=matcher_alt.nlp)

                kb.summarize(output_folder=args.output, prefix="kb_")
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
