from Summarizer import PageRankSummarizer

if __name__ == '__main__':
    pr_summarizer = PageRankSummarizer("./documents", )
    pr_summarizer.summarize()
    print(pr_summarizer)