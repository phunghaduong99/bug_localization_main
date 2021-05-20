import json
import pickle
import nltk
import re
import string
import inflection
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv


def process_text(text):
    # tokenize raw_text
    tokens_text = nltk.wordpunct_tokenize(text)
    # split_camel
    returning_tokens = tokens_text[:]
    for token in tokens_text:
        split_tokens = re.split(fr'[{string.punctuation}]+', token)
        # if token is split into some other tokens
        if len(split_tokens) > 1:
            returning_tokens.remove(token)
            # camel case detection for new tokens
            for st in split_tokens:
                camel_split = inflection.underscore(st).split('_')
                if len(camel_split) > 1:
                    returning_tokens.append(st)
                    returning_tokens = returning_tokens + camel_split
                else:
                    returning_tokens.append(st)
        else:
            camel_split = inflection.underscore(token).split('_')
            if len(camel_split) > 1:
                returning_tokens = returning_tokens + camel_split
    # normalize
    # build a translate table for punctuation and number removal
    punctual_table = str.maketrans({c: None for c in string.punctuation + string.digits})
    text_punctual = [token.translate(punctual_table) for token in returning_tokens]
    text_lower = [token.lower() for token in text_punctual if token]
    # remove stop_word
    text_rm_stopwords = [token for token in text_lower if token not in stop_words]
    # remove_java_keyword
    text_rm_javakeyword = [token for token in text_rm_stopwords if token not in java_keywords]
    # stem
    stemmer = PorterStemmer()
    processed_text = [stemmer.stem(token) for token in text_rm_javakeyword]
    listToStr = ' '.join([str(elem) for elem in processed_text])
    return listToStr


def cosine_sim(text1, text2):
    """ Cosine similarity with tfidf

    Arguments:
        text1 {string} -- first text
        text2 {string} -- second text
    """
    vectorizer = TfidfVectorizer(min_df=1)
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = (tfidf * tfidf.T).A[0, 1]
    return sim


def class_name_similarity(src_files, bug_reports):
    """[summary]

    Arguments:
        raw_text {string} -- raw text of the bug report
        source_code {string} -- java source code
    """
    reports_strings = [' '.join(report.summary['stemmed'] + report.description['stemmed'])
                       for report in bug_reports.values()]
    class_name_strings = [src.exact_file_name
                          for src in src_files.values()]

    simis = [[0] * len(src_files) for i in range((len(bug_reports)))]
    idx_report = 0
    for report in reports_strings:
        idx_src = 0

        for class_name_string in class_name_strings:
            if class_name_string in report:
                simis[idx_report][idx_src] = len(class_name_string)

            idx_src = idx_src + 1
        idx_report = idx_report + 1
    return simis


def new_class_name_similarity(src_files, bug_reports):
    """[summary]

    Arguments:
        raw_text {string} -- raw text of the bug report
        source_code {string} -- java source code
    """
    reports_strings = [' '.join(report.summary['stemmed'] + report.description['stemmed'])
                       for report in bug_reports.values()]
    class_name_strings = [' '.join(src.file_name['stemmed'])
                          for src in src_files.values()]

    simis = [[0] * len(src_files) for i in range((len(bug_reports)))]
    idx_report = 0
    for report in reports_strings:
        idx_src = 0

        for class_name_string in class_name_strings:
            simis[idx_report][idx_src] = cosine_sim(class_name_string, report)
            idx_src = idx_src + 1
        idx_report = idx_report + 1
        print(idx_report)

    return simis


def main():
    with open('data_output/preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)
    with open('data_output/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    all_scores = new_class_name_similarity(src_files, bug_reports)

    with open('data_output/class_similarity_main.json', 'w') as file:
        json.dump(all_scores, file)


if __name__ == '__main__':
    main()
