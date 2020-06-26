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
    

def idx_token(offset, sent_raw):
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
    
def idx_leaf_sent(idx_token, sent_tree):
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

def get_tag(sent_tree, idx_leaf):
    '''
    args
    - sent_tree: parsed sentence
    - idx_leaf: idx of 'leaf' (token/chunk) in sent_tree
    returns
    - the tag of the leaf if idx_leaf is in range(len(sent_tree))
    - if idx_leaf is out of range, returns 'N/A'
    '''
    if idx_leaf in range(len(sent_tree)):
        if isinstance(sent_tree[idx_leaf], tuple):
            # sent_tree[idx_leaf] is a token
            return sent_tree[idx_leaf][1]
        else:
            # sent_tree[idx_leaf] is a chunk
            return 'NP'
    else:
        return 'N/A'


def get_feature(dic, use_url):
    '''
    args
    - dic: dictionary with GOLD_FIELDNAMES except A-coref and B-coref as necessary keys
    (actually in this code we use pandas dataframe to read data,
    so dic will not be exactly a 'dictionary' type object,
    but the method works fine as pandas.core.series.Series implements important dictionary features)
    - use_url: if True, use text from the Wikiepedia page on the URL to extract additional feature(s)
    returns
    - a dictionary with the following keys/values
      - pronoun : the pronoun itself
      - ratio_distance : with every token/chunks in the text indexed,
            measure the 'distances' between the pronoun and A/B i.e. how many token/chunks apart they are
            ratio_distance is the (distance between A and pronoun) / (distance betwene B and pronoun)
      - tag_pre_A, tag_pre_B : the tags of the token/chunk that come before/after the NP chunk that contains A
            if there is no such token/chunk, (i.e. A is at the start/end of a sentence) set to 'N/A'
      - tag_post_A, tag_post_B : same as above, but applied to B
      ### only contains below key/value(s) when use_url is True
      - ratio_occ : search for every occurence of A and B in the wikipedia page
            if A or B is a multi-word phrase, search its first word, its last word, and entire phrase separately
            and calculate total occurence as (occurence of first word) + (occ. of last word) - (occ. of entire phrase)
            this is to find reference to A/B by their given names AND surnames, without duplicating references by their full names
            ratio_occ is the (number of occurences of A in the page)^2 / (number of occurences of B in the page)^2
            since A and B is collected from the snippet, which is a part of the page, divide-by-zero error cannot happen
    '''
    feature = dict()
    
    snip_raw = dic['Text']
    idx_sent_A, offset_sent_A = idx_sent(dic['A-offset'], snip_raw)
    idx_sent_B, offset_sent_B = idx_sent(dic['B-offset'], snip_raw)
    idx_token_A = idx_token(offset_sent_A, sent_tokenize(snip_raw)[idx_sent_A])
    idx_token_B = idx_token(offset_sent_B, sent_tokenize(snip_raw)[idx_sent_B])
    
    ### Sentence-tokenize, word-tokenize, tag and parse the snip text
    
    sents_raw = sent_tokenize(snip_raw)
    sents_tk = [word_tokenize(sent_raw) for sent_raw in sents_raw]
    sents_tag = [pos_tag(sent_tk) for sent_tk in sents_tk]
    
    # Make sure that [A] and [B] are correctly tagged as NNP
    for i in len(word_tokenize(dic['A'])):
        sents_tag[idx_sent_A][idx_token_A + i] = 'NNP'
    
    for i in len(word_tokenize(dic['B'])):
        sents_tag[idx_sent_B][idx_token_B + i] = 'NNP'
    
    grammar = r"""
      NP: {<DT|CD|JJ|PRP.+>*<PRP|NN.*>+}          # Chunk sequences of DT, JJ, NN as NP
      """
    cp = RegexpParser(grammar, loop=100)
    
    sents_tree = [cp.parse(sent_tag) for sent_tag in sents_tag]
    
    idx_leaf_A_sent = idx_leaf_sent(idx_token_A, sents_tree[idx_sent_A])
    idx_leaf_B_sent = idx_leaf_sent(idx_token_B, sents_tree[idx_sent_B])
    idx_leaf_A_snip = sum([len(sent_tree) for sent_tree in sents_tree[:idx_sent_A]]) + idx_leaf_A_sent
    idx_leaf_B_snip = sum([len(sent_tree) for sent_tree in sents_tree[:idx_sent_B]]) + idx_leaf_B_sent
    
    idx_sent_pronoun, offset_sent_pronoun = idx_sent(dic['Pronoun-offset'], snip_raw)
    idx_token_pronoun = idx_token(offset_sent_pronoun, sent_tokenize(snip_raw)[idx_sent_pronoun])
    idx_leaf_pronoun_sent = idx_leaf_sent(idx_token_pronoun, sents_tree[idx_sent_pronoun])
    idx_leaf_pronoun_snip = sum([len(sent_tree) for sent_tree in sents_tree[:idx_sent_pronoun]]) + idx_leaf_pronoun_sent
    
    distance_A_pronoun = abs(idx_leaf_A_snip - idx_leaf_pronoun_snip)
    distance_B_pronoun = abs(idx_leaf_B_snip - idx_leaf_pronoun_snip)
    feature['ratio_distance'] = distance_A_pronoun / distance_B_pronoun
    
    feature['tag_pre_A'] = get_tag(sents_tree[idx_sent_A], idx_leaf_A_sent-1)
    feature['tag_post_A'] = get_tag(sents_tree[idx_sent_A], idx_leaf_A_sent+1)
    feature['tag_pre_B'] = get_tag(sents_tree[idx_sent_B], idx_leaf_B_sent-1)
    feature['tag_post_B'] = get_tag(sents_tree[idx_sent_B], idx_leaf_B_sent+1)
    
    feature['pronoun'] = dic['Pronoun']
    
    if(use_url):
        wiki = wikipediaapi.Wikipedia('en')
        
        # extract entry string from the URL
        regex_entry = re.compile(r'.+/(.*)$')
        entry = regex_entry.search(dic['URL']).group(1)
        
        page_wiki = wiki.page(entry)
        
        if(not page_wiki.exists()):
            # safety measure to avoid crash
            feature['ratio_occ'] = -1
            return feature
        
        page_raw = page_wiki.text
        #page_tk = word_tokenize(page_raw)
        
        words_A = [word for word in word_tokenize(dic['A']) if word.isalnum()]
        words_B = [word for word in word_tokenize(dic['A']) if word.isalnum()]
        
        if(len(words_A)==1):
            occ_A = page_raw.count(words_A[0])
        else:
            occ_A = page_raw.count(words_A[0])\
                 + page_raw.count(words_A[len(words_A)])\
                 - page_raw.count(dic['A'])
        if(occ_A==0): occ_A = 1 #shouldn't happen
        
        if(len(words_B)==1):
            occ_B = page_raw.count(words_B[0])
        else:
            occ_B = page_raw.count(words_B[0])\
                 + page_raw.count(words_B[len(words_B)])\
                 - page_raw.count(dic['B'])
        if(occ_B==0): occ_B = 1 #shouldn't happen
        
        feature[ratio_occ] = occ_A**2 / occ_B**2
    
    return feature



if __name__ == '__main__':
    features_snip = [] # features extracted from only the snippet
    features_page = [] # features extracted with wiki page
    
    for file_input_train in FILES_INPUT_TRAIN:
        data_input_train = pd.read_csv(file_input_train, sep='\t', header=0)
        for dic in data_input_train.to_dict(orient='records'):
            feature_snip = get_feature(dic, False)
            feature_page = get_feature(dic, True)
            # Ok, this is probably not the world's most efficient way to do this..
            
            features_snip.append(feature_snip, dic['A-coref'])
            features_page.append(feature_page, dic['A-coref']) 
    
    classifier_snip = NaiveBayesClassifier.train(features_snip)
    classifier_page = NaiveBayesClassifier.train(features_page)
    
    data_output_snip = []
    data_output_page = []
    
    data_input_test = pd.read_csv(FILE_INPUT_TEST, sep='\t', header=0)
    for dic in data_input_test.to_dict(orient='records'):
        id_test = dic['ID']
        
        feature_snip = get_feature(dic, False)
        feature_page = get_feature(dic, True)
        
        A_coref_snip = classifier_snip.classify(feature_snip)
        A_coref_page = classifier_page.classify(feature_page)
        
        data_output_snip.append({'ID':id_test, 'A-coref':A_coref_snip, 'B-coref':(not A_coref_snip)})
        data_output_page.append({'ID':id_test, 'A-coref':A_coref_page, 'B-coref':(not A_coref_page)})
    
    pd.Dataframe(data_output_snip, columns = ['ID','A-coref','B-coref']).to_csv(FILE_OUTPUT_SNIP, index=False, header=False)
    pd.Dataframe(data_output_page, columns = ['ID','A-coref','B-coref']).to_csv(FILE_OUTPUT_PAGE, index=False, header=False)
        
    