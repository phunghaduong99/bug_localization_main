import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def most_recent_report(reports):
    """ Returns the most recently submitted previous report that shares a filename with the given bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        current_date {datetime} -- until date
        bug_reports {list of dictionaries} -- list of all bug reports
    """

    if len(reports) > 0:
        return max(reports, key=lambda x: x.fixdate)

    return None


def get_months_between(d1, d2):
    """ Calculates the number of months between two date strings

    Arguments:
        d1 {datetime} -- date 1
        d2 {datetime} -- date 2
    """

    diff_in_months = abs((d1.year - d2.year) * 12 + d1.month - d2.month)

    return diff_in_months


def bug_fixing_recency(br, prev_reports):
    """ Calculates the Bug Fixing Recency as defined by Lam et al.

    Arguments:
        report1 {dictionary} -- current bug report
        report2 {dictionary} -- most recent bug report
    """
    most_rr = most_recent_report(prev_reports)

    if br and most_rr:
        return 1 / float(
            get_months_between(br.fixdate, most_rr.fixdate) + 1
        )

    return 0


def previous_reports(filename, until, bug_reports):
    """ Returns a list of previously filed bug reports that share a file with the current bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        until {datetime} -- until date
        bug_reports {list of dictionaries} -- list of all bug reports
    """
    return [
        br
        for idx, br in bug_reports.items()
        if (filename in br.fixed_files and br.fixdate < until)
    ]


def cosine_sim(text1, text2):
    """
        Calculating tf-idf vectors for source and report sets
        to find similar source files for each bug report.
        """
    vectorizer = TfidfVectorizer(min_df=1)
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = (tfidf * tfidf.T).A[0, 1]
    return sim


def bug_fixing_frequency(src_files, bug_reports):
    bug_fix_fr = [[0] * len(src_files) for i in range((len(bug_reports)))]
    bug_fix_re = [[0] * len(src_files) for i in range((len(bug_reports)))]
    cola_filter_score = [[0] * len(src_files) for i in range((len(bug_reports)))]
    idx_report = 0
    for idx_br, br in bug_reports.items():
        idx_src = 0

        for filename, src in src_files.items():
            prev_reports = previous_reports(filename, br.fixdate, bug_reports)

            if len(prev_reports) > 0:
                prev_report_string = ""
                for report in prev_reports:
                    prev_report_string = prev_report_string + ' '.join(report.summary['stemmed']
                                                                       + report.description['stemmed']) + ' '
                report_string = ' '.join(br.summary['stemmed'] + br.description['stemmed'])
                cola_filter_score[idx_report][idx_src] = cosine_sim(report_string, prev_report_string)
                bug_fix_fr[idx_report][idx_src] = len(prev_reports)
                bug_fix_re[idx_report][idx_src] = bug_fixing_recency(br, prev_reports)

            idx_src = idx_src + 1
        idx_report = idx_report + 1
        print(idx_report)
    return [bug_fix_fr, bug_fix_re, cola_filter_score]


def main():
    with open('data_output/preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)
    with open('data_output/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    scores = bug_fixing_frequency(src_files, bug_reports)

    with open('data_output/bug_fixing_frequency.json', 'w') as file:
        json.dump(scores[0], file)

    with open('data_output/bug_fixing_recency.json', 'w') as file:
        json.dump(scores[1], file)

    with open('data_output/collaborative_filtering_score.json', 'w') as file:
        json.dump(scores[2], file)


if __name__ == '__main__':
    main()
