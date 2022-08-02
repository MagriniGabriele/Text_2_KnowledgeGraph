from Summarizer import PageRankSummarizer, ClusterSummarizer, KnowledgeBaseSummarizer

if __name__ == '__main__':

    print("PageRank Summarization - Carbon Lang ---------------------------------\n")
    pr_summarizer = PageRankSummarizer("./documents/carbon", 5)
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
