import json
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Similarity:
    __slots__ = ['src_files', 'src_strings']

    def __init__(self, src_files):
        self.src_files = src_files
        self.src_strings = [''.join(' '.join(src_api["stemmed"])
                                    for src_api in src.methods_api)
                            for src in self.src_files.values()]

    def calculate_similarity(self, src_tfidf, reports_tfidf):
        """Calculatnig cosine similarity between source files and bug reports"""

        # Normalizing the length of source files
        src_lenghts = np.array([float(len(src_str.split()))
                                for src_str in self.src_strings]).reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_src_len = min_max_scaler.fit_transform(src_lenghts)

        # Applying logistic length function
        src_len_score = 1 / (1 + np.exp(-12 * normalized_src_len))

        simis = []
        for report in reports_tfidf:
            s = cosine_similarity(src_tfidf, report)

            # revised VSM score calculation
            rvsm_score = s * src_len_score

            normalized_score = np.concatenate(
                min_max_scaler.fit_transform(rvsm_score)
            )

            simis.append(normalized_score.tolist())

        return simis

    def find_similars(self, bug_reports):
        """Calculating tf-idf vectors for source and report sets
        to find similar source files for each bug report.
        """

        reports_strings = [' '.join(report.summary['stemmed'] + report.description['stemmed'])
                           for report in bug_reports.values()]

        tfidf = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
        src_tfidf = tfidf.fit_transform(self.src_strings)

        reports_tfidf = tfidf.transform(reports_strings)

        simis = self.calculate_similarity(src_tfidf, reports_tfidf)

        return simis

    def cosine_sim(self, text1, text2):
        """Calculating tf-idf vectors for source and report sets
        to find similar source files for each bug report.
        """
        vectorizer = TfidfVectorizer(min_df=1)
        tfidf = vectorizer.fit_transform([text1, text2])
        sim = (tfidf * tfidf.T).A[0, 1]
        return sim



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


def api_similarity(src_files, bug_reports):
    """[summary]

    Arguments:
        raw_text {string} -- raw text of the bug report
        source_code {string} -- java source code
    """
    reports_strings = [' '.join(report.summary['stemmed'] + report.description['stemmed'])
                       for report in bug_reports.values()]
    api_methods_sim = [[0] * len(src_files) for i in range((len(bug_reports)))]

    for idx_report, report_strings in enumerate(reports_strings):
        idx_src = 0
        for idx, src in src_files.items():
            simis_methods = []
            api_source_strings = ""

            if len(src.methods_api) > 0:
                for idx_api_method, api_method in enumerate(src.methods_api):
                    if len(api_method['stemmed']) > 0:
                        src_api_method_strings = ' '.join(api_method['stemmed'])
                        api_source_strings = api_source_strings + src_api_method_strings + " "
                        sim_api_method_report = cosine_sim(src_api_method_strings, report_strings)
                        simis_methods.append(sim_api_method_report)

                if len(api_source_strings) > 0:
                    sim_src_api = cosine_sim(api_source_strings, report_strings)
                    simis_methods.append(sim_src_api)
                    api_methods_sim[idx_report][idx_src] = max(simis_methods)
            idx_src = idx_src + 1
    return api_methods_sim


def main():
    with open('data_output/preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)
    with open('data_output/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)
    sm = Similarity(src_files)
    api_similarity_score = sm.find_similars(bug_reports)

    with open('data_output/api_similarity.json', 'w') as file:
        json.dump(api_similarity_score, file)


if __name__ == '__main__':
    main()
