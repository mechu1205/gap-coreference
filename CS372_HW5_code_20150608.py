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



if __name__ == '__main__':
    for file_input_train in FILES_INPUT_TRAIN:
        feature_snip = get_feature(file, False)
        classifier_snip = NaiveBayesClassifier.train(feature_snip)
        
        
    