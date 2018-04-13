import matplotlib
import nltk
import spacy
import string
import pandas as pd

from nltk.draw.tree import TreeView
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.parse.stanford import StanfordParser
from nltk.tree import ParentedTree
from nltk.parse.corenlp import CoreNLPServer
from nltk.parse.corenlp import CoreNLPParser
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from word_mod import zipf_frequency
import language_check
from fuzzywuzzy import fuzz

from anigen_tools.parsing import parse_description
from anigen_tools.parsing import sanitize_text
from anigen_tools.parsing import sentence_splitter

import enchant

edict = enchant.Dict("en_US")
anglo_edict = enchant.Dict("en_UK")

punct_set = set(string.punctuation)
punct_set.remove('.')
cached_sw = stopwords.words("english") + list(string.punctuation)

colors = list(matplotlib.colors.cnames.keys())

color_set = set(colors)

core_parser = CoreNLPParser(url='http://localhost:9000')

gram_checker = language_check.LanguageTool('en-US')
gram_checker.disabled = set(['UPPERCASE_SENTENCE_START'])

free_words_less = ['a',
 'is',
 'of',
 'the',
 'and',
 'it',
 'are',
 'an']

nuisance_phrases = ['a picture of ', 'this is ', 'there is ', 'shown ', 'an image of ', 'view of a ', 'close up of ']

verb_tags = set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])


def least_common_word(sent):
    wlfs = [zipf_frequency(w, 'en') for w in sent.split()]
    lowest_freq = min(wlfs)
    return lowest_freq > 4


def grammar_check_phrases(dataset):
    corrected = {}
    passed = []
    for sentence in dataset:
        corrections = gram_checker.check(sentence)
        if corrections:
            corrected[sentence] = corrections
        else:
            passed.append(sentence)
    return corrected, passed


def get_pos_tags(sentence):
    constituent_parse = [list(i)[0]
                              for i in core_parser.raw_parse_sents([sentence])][0]
    pos_tags = constituent_parse.pos()
    return pos_tags


def filter_pos_tags(tagged_sent):
    include_as_article = []
    for word, tag in tagged_sent:
        if tag == 'DT':
            include_as_article.append('True')
        else:
            include_as_article.append('False')
    return include_as_article


def least_common_word(sent, freq_thresh=4.7):
    wlfs = [zipf_frequency(w, 'en') for w in sent.split()]
    lowest_freq = min(wlfs)
    return lowest_freq > freq_thresh





def remove_blacklisted(sentences):
    word_blacklist = set(['uses', 'using', 'people', 'helped', 'measures', 'records', 'data', 'works',
                         'teach', 'body', 'with', 'training', 'waste', 'plastic', 'gravity', 'commission', 'space'] + colors)

    blacklist_removed = [s for s in sentences if not set(
        [w.lower() for w in s.split()]).intersection(word_blacklist)]

    return blacklist_removed


def filter_on_freq(sent, higher_freq, lower_freq):
    wlfs = [zipf_frequency(w, 'en') for w in sent.split()]
    lowest_freq = min(wlfs)
    return higher_freq > lowest_freq > lower_freq


def extract_np(psent):
    for subtree in psent.subtrees():
        if subtree.label() == 'NP':
            subprod = subtree.productions()[0].unicode_repr()
            if 'NN' in subprod or 'NNP' in subprod:
                yield ' '.join(word for word in subtree.leaves())


def np_chunker(doc, parsed_sents):
    noun_phrases = [list(extract_np(sent)) for sent in parsed_sents]
    token_spans = [list(compute_token_spans(sent, doc))
                        for sent in parsed_sents]
    noun_phrase_spans = [assign_word_spans(
        noun_phrases[n], doc, token_spans[n]) for n in range(len(parsed_sents))]
    return {'chunks': noun_phrase_spans, 'named_chunks': noun_phrases}



def check_mispelled(word):
    return word and word.isalpha() and not (edict.check(word))


def normalize_sent(sent):
    nsent = sent.lower()
    for np in nuisance_phrases:
        nsent = nsent.replace(np, '')
        nsent = nsent.replace('that is ', 'is ').replace('who is ', 'is ')
    if nsent[-1] not in ['.', '!']:
        nsent += '.'

    return nsent


def spellcheck_phrase(phrase):
    misspellings = [check_mispelled(word) for word in word_tokenize(phrase)]
    return sum(misspellings)


def filter_pos_tags(tagged_sent):
    include_as_article = []
    for word, tag in tagged_sent:
        if tag == 'DT' or word.lower() in free_words_less:
            include_as_article.append('True')
        else:
            include_as_article.append('False')
    return include_as_article


def least_common_word(sent, freq_thresh=4.7):
    wlfs = [zipf_frequency(w, 'en') for w in sent.split()]
    lowest_freq = min(wlfs)
    return lowest_freq > freq_thresh


def form_strings(tagged_sentence):
    words = [w[0] for w in tagged_sentence]
    include_word = filter_pos_tags(tagged_sentence)
    return f"{' '.join(words)} # {' '.join(include_word)}"


def form_strings_w_pos(tagged_sentence):
    words = [w[0] for w in tagged_sentence]
    include_word = [w[1] for w in tagged_sentence]
    return f"{' '.join(words)} # {' '.join(include_word)}"


def output_sentences(tagged_sents):
    build_output = pd.Series([form_strings(s) for s in tagged_sents])
    build_output.to_csv('simple_coco.txt', index=False, header=None)
    build_output = pd.Series([form_strings_w_pos(s) for s in tagged_sents])
    build_output.to_csv('simple_coco_tagged.txt', index=False, header=None)


def get_word_sisters(word):
    sister_words_syn = []
    sister_words = []
    for syn in wordnet.synsets(word):
        if syn.pos() != 'n':
            continue
        for hyper in syn.hypernyms():
            for hypo in hyper.hyponyms():
                sister_words_syn.append(hypo)
                sister_words.extend(hypo.lemma_names())
        break
    return sister_words, sister_words_syn


