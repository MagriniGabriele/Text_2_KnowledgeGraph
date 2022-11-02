from abc import ABC

import re
import neuralcoref
import spacy
import pandas as pd
from spacy.matcher import Matcher
from tqdm import tqdm

from Metrics import information_density, usable_triples_density, usable_information_density

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"


class KnowledgeExtractor(ABC):

    def __init__(self, language_model: str = None, verbose: bool = False):
        if language_model is None:
            # il modell di default è quello inglese
            self.nlp = spacy.load("en_core_web_sm")
        else:
            try:
                self.nlp = spacy.load(language_model)
            except Exception as e:
                print("Could not found language model, check for typos or if installed: using English model")
                self.nlp = spacy.load("en_core_web_sm")
        self.verbose = verbose
        neuralcoref.add_to_pipe(self.nlp)

    @staticmethod
    def text_from_file(path_to_file: str):
        file = open(path_to_file, "r")
        text = ""
        for line in file.readlines():
            text += line
        return text

    def parse_from_file(self, path_to_file: str, verbose: bool = False):
        text = self.text_from_file(path_to_file)
        return self.parse(text, verbose)

    def report_from_file(self, path_to_file: str):
        text = self.text_from_file(path_to_file)
        self.report(text)

    def report(self, text):
        triples = self.parse(text, verbose=False)
        print("Information density: ", information_density(triples, self.nlp(text)))
        print("Usable triples density: ", usable_triples_density(triples, self.nlp(text)))
        print("Usable information density: ", usable_information_density(triples, self.nlp(text)))

    def parse(self, text, verbose: bool = False):
        raise NotImplementedError


class MatcherExtractor(KnowledgeExtractor):
    @staticmethod
    def split_into_sentences(text, verbose: bool):
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
        if verbose:
            print(text)
        if "”" in text:
            text = text.replace(".”", "”.")
        if "\"" in text:
            text = text.replace(".\"", "\".")
        if "!" in text:
            text = text.replace("!\"", "\"!")
        if "?" in text:
            text = text.replace("?\"", "\"?")
        text = text.replace(";", ";<stop>")
        text = text.replace(",", ",<stop>")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        if verbose:
            print(text)
        text = text.replace("<prd>", ".")
        if verbose:
            print(text)
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        # sentences = [self.nlp(s)._.coref_resolved for s in sentences]
        return sentences

    def get_entities(self, sent):
        ## chunk 1
        ent1 = ""
        ent2 = ""

        prv_tok_dep = ""  # dependency tag of previous token in the sentence
        prv_tok_text = ""  # previous token in the sentence

        prefix = ""
        modifier = ""

        #############################################################

        for tok in self.nlp(sent):
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

    def get_relation(self, sent):
        doc = self.nlp(sent)

        # Matcher class object
        matcher = Matcher(self.nlp.vocab)

        # define the pattern
        pattern = [{'DEP': 'ROOT'},
                   {'DEP': 'prep', 'OP': "?"},
                   {'DEP': 'agent', 'OP': "?"},
                   {'POS': 'ADJ', 'OP': "?"}]

        matcher.add("matching_1", None, pattern)

        matches = matcher(doc)
        k = len(matches) - 1
        # ema: se non trova match qua muore
        if k >= 0:
            span = doc[matches[k][1]:matches[k][2]]
            return (span.text)
        else:
            return ""

    def parse(self, text, verbose: bool = False):
        # prima iterazione di spacy, risolvo le coreferenze
        doc = self.nlp(text)
        coref_solved_text = doc._.coref_resolved
        if self.verbose:
            print("Original text:\n", text)
            print("Cooreference resolved text:\n", coref_solved_text)

        # seconda iterazione di spacy, estraggo i dati
        entity_pairs = []
        new_text = str(coref_solved_text)
        sentences = self.split_into_sentences(new_text, self.verbose)

        if self.verbose:
            print("Sentences:\n", sentences)

        for sent in sentences:
            entity_pairs.append(self.get_entities(sent))

        if self.verbose:
            print("Entities:\n", entity_pairs)

        relations = [self.get_relation(i) for i in tqdm(sentences)]
        # extract subject
        source = [i[0] for i in entity_pairs]

        # extract object
        target = [i[1] for i in entity_pairs]

        # purge malformed triples
        s = []
        r = []
        t = []
        for i in range(len(source)):
            if source[i] != "" and relations[i] != "" and target[i] != "":
                s.append(source[i])
                r.append(relations[i])
                t.append(target[i])
        return [s, r, t]


class AlternativeMatcherExtractor(KnowledgeExtractor):

    @staticmethod
    def filter_spans(spans):
        # Filter a sequence of spans so they don't contain overlaps
        # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
        get_sort_key = lambda span: (span.end - span.start, -span.start)
        sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
        result = []
        seen_tokens = set()
        for span in sorted_spans:
            # Check for end - 1 here because boundaries are inclusive
            if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
                result.append(span)
            seen_tokens.update(range(span.start, span.end))
        result = sorted(result, key=lambda span: span.start)
        return result

    def get_entity_pairs(self, text, coref=True):
        # preprocess text
        text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
        text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
        text = self.nlp(text)
        if coref:
            text = self.nlp(text._.coref_resolved)  # resolve coreference clusters

        def refine_ent(ent, sent):
            unwanted_tokens = (
                'PRON',  # pronouns
                'PART',  # particle
                'DET',  # determiner
                'SCONJ',  # subordinating conjunction
                'PUNCT',  # punctuation
                'SYM',  # symbol
                'X',  # other
            )
            ent_type = ent.ent_type_  # get entity type
            if ent_type == '':
                ent_type = 'NOUN_CHUNK'
                ent = ' '.join(str(t.text) for t in
                               self.nlp(str(ent)) if t.pos_
                               not in unwanted_tokens and t.is_stop is False)
            elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
                refined = ''
                for i in range(len(sent) - ent.i):
                    if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                        refined += ' ' + str(ent.nbor(i))
                    else:
                        ent = refined.strip()
                        break

            return ent, ent_type

        sentences = [sent.string.strip() for sent in text.sents]  # split text into sentences
        ent_pairs = []
        for sent in sentences:
            sent = self.nlp(sent)
            spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
            spans = self.filter_spans(spans)
            with sent.retokenize() as retokenizer:
                [retokenizer.merge(span, attrs={'tag': span.root.tag,
                                                'dep': span.root.dep}) for span in spans]
            deps = [token.dep_ for token in sent]

            # limit our example to simple sentences with one subject and object
            if (deps.count('obj') + deps.count('dobj')) != 1 \
                    or (deps.count('subj') + deps.count('nsubj')) != 1:
                continue

            for token in sent:
                if token.dep_ not in ('obj', 'dobj'):  # identify object nodes
                    continue
                subject = [w for w in token.head.lefts if w.dep_
                           in ('subj', 'nsubj')]  # identify subject nodes
                if subject:
                    subject = subject[0]
                    # identify relationship by root dependency
                    relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                    if relation:
                        relation = relation[0]
                        # add adposition or particle to relationship
                        if relation.nbor(1).pos_ in ('ADP', 'PART'):
                            relation = ' '.join((str(relation), str(relation.nbor(1))))
                    else:
                        relation = 'unknown'

                    subject, subject_type = refine_ent(subject, sent)
                    token, object_type = refine_ent(token, sent)

                    ent_pairs.append([str(subject), str(relation), str(token),
                                      str(subject_type), str(object_type)])

        ent_pairs = [sublist for sublist in ent_pairs
                     if not any(str(ent) == '' for ent in sublist)]
        pairs = pd.DataFrame(ent_pairs, columns=['subject', 'relation', 'object',
                                                 'subject_type', 'object_type'])
        if self.verbose:
            print('Entity pairs extracted:', str(len(ent_pairs)))

        return pairs

    def parse(self, text, verbose: bool = False):
        triples = [[], [], []]
        pairs = self.get_entity_pairs(text)
        for i in pairs:
            if verbose:
                print(pairs[i])
            try:
                t_0 = pairs[i][0]
                t_1 = pairs[i][1]
                t_2 = pairs[i][2]
                triples[0].append(t_0)
                triples[1].append(t_1)
                triples[2].append(t_2)
            except KeyError:
                pass

        return triples
        # draw_kg(pairs)
