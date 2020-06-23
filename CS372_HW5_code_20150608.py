from collections import defaultdict
import csv
import re
from nltk import *
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
import wikipediaapi
import pandas as pd

FILES_INPUT_TRAIN = ['gap-development.tsv', 'gap-validation.tsv']
FILE_INPUT_TEST = 'gap-test.tsv'
FILE_OUTPUT_SNIP = 'CS372_HW5_snippet_output_20150608.tsv'
FILE_OUTPUT_PAGE = 'CS372_HW5_page_output_20150608.tsv'

def idx_sent(offset, text_raw):
    '''
    args
    - offset: int (>=0)
    - text_raw: raw text
    returns
    - index of the sentence that contains text_raw[offset]
    - new offset that is relevant to the beginning of the above sentence
    '''
    sents_raw = sent_tokenize(text_raw)
    num_sents = len(sents_raw)
    for i in range(num_sents):
        num_ws = len(re.compile('\s*').match(text_raw).group())
        len_sent = len(sents_raw[i])
        if(offset < num_ws + len_sent):
            return (i, offset - num_ws)
        else:
            offset -= num_ws + len_sent
            text_raw = text_raw[num_ws + len_sent:]
    

def idx_token(offset, sent_raw, sent_tk):
    '''
    args
    - offset: int number (>=0)
    - sent_raw: raw sentence (str)
    returns
    - index of the token in the tokenized sentence that includes sent_raw[offset]
    '''
    sent_tk = word_tokenize(sent_raw)
    
    # number of whitespaces at the start of sent_raw
    num_ws = len(re.compile('\s*').match(sent_raw).group())
    # length of the first token in the sentence
    len_tk0 = len(sent_tk[0])
    
    if (offset <= num_ws + len_tk0): return 0
    else:
        return 1 + idx_token(offset - num_ws - len_tk0,
                             sent_raw[num_ws + len_tk0:],
                             sent_tk[1:])
    
def idx_leaf(idx_token, sent_tree):
    '''
    args
    - idx_token: index of a token in the tokenized sentence (>=0)
    - sent_tree: parsed sentence
    returns
    - returns the index of the leaf(=chunk/token) that contains the idx_token -th token of the tokenized sentence
    '''
    if(idx_token == 0): return 0
    if(isinstance(sent_tree[0], str)):
        # sent_tree[0] is a token
        return 1 + idx_leaf(idx_token-1, sent_tree[1:])
    else:
        # sent_tree[0] is a chunk
        len_leaf0 = len(sent_tree[0])
        if (idx_token < len_leaf0): return 0
        else: return 1 + idx_leaf(idx_token - len_leaf0, sent_tree[1:])


def get_feature(dic, use_url):
    '''
    args
    - dic: dictionary with GOLD_FIELDNAMES except A-coref and B-coref as necessary keys
    (actually in this code we use pandas dataframe to read data,
    so dic will not be exactly a 'dictionary' type object,
    but the method works fine as pandas.core.series.Series implements important dictionary features)
    - use_url: if True, use text from the Wikiepedia page on the URL to extract additional feature(s)
    returns
    - feature: dictionary with the following keys/values
      - pronoun 
      - ratio_distance
      - pre_A_tag, pre_B_tag
      - post_A_tag, post_B_tag
      - ratio_occ
    '''


if __name__ == '__main__':
    for file_input_train in FILES_INPUT_TRAIN:
        #feature_snip = get_feature(file, False)
        #classifier_snip = NaiveBayesClassifier.train(feature_snip)
        
        
    