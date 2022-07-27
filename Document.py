class Document:

    def __init__(self, nlp, text, name, title):
        self.name = name
        self.title = title
        self.doc = nlp(text)
        self.text = text
        self.TFIDF = dict()
        self.terms = set([token.text for token in self.doc])
        self.keywords = set()

    def compute_tfidf(self, idf):
        for term in self.terms:
            count = 0
            for token in self.doc:
                count += int(term == token.text)
            self.TFIDF[term] = (count / len(self.doc)) * idf[term]
        keywords = {k: v for k, v in sorted(self.TFIDF.items(), key=lambda item: item[1], reverse=True)}
        counter = 0
        for kw in keywords:
            self.keywords |= {kw}
            if counter == 5:
                break
            counter = counter + 1

    def contains(self, term: str):
        return term in self.terms

    def print_keywords(self):
        keywords = {k: v for k, v in sorted(self.TFIDF.items(), key=lambda item: item[1], reverse=True)}
        counter = 0
        for kw in keywords:
            print(kw, keywords[kw])
            if counter == 5:
                break
            counter = counter + 1

    def title_resemblance(self, sentence):
        intersection = 0
        union = 0
        for kw in self.keywords:
            if kw in self.title and kw in sentence.text:
                intersection += 1
            if kw in self.title or kw in sentence.text:
                union += 1
        # cover the case of a title with no keyword
        union = max(1, union)
        return intersection / union

    def coverage(self, sentence):
        intersection = 0
        for kw in self.keywords:
            if kw in sentence.text:
                intersection += 1
        return intersection / len(self.keywords)

    @staticmethod
    def proper_entities(sentence):
        return len(sentence.ents) / len(sentence)

    @staticmethod
    def numerical_words(sentence):
        numbers = 0
        for token in sentence:
            try:
                float(token.text)
                numbers += 1
            except ValueError:
                pass
        return numbers / len(sentence)
    
    def relative_length(self, sentence):
        # return (len(sentence) * len(self.doc.sents)) / len(self.doc) i don't get whu the constant multiplier
        return len(sentence) / len(self.doc)
