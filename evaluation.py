import json
import pickle
from sklearn.svm import SVC
import numpy as np
import os
from statistics import mean
import time

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def prepare_data(src_files, bug_reports, *rank_scores):
    k = 5
    num_fold = 10
    file_names = [filename for filename, src in src_files.items()]
    vsm_simi = rank_scores[0][0]
    i = 0
    avg_fold = int(len(bug_reports) / num_fold) + 1

    samples_test = [[], [], [], [], [], [], [], [], [], []]
    samples_train = [[], [], [], [], [], [], [], [], [], []]
    labels = [[], [], [], [], [], [], [], [], [], []]
    fold10_train = []
    labels10 = []
    fold10_test = []
    remains_number = len(bug_reports) - avg_fold * 9

    for bug_id, bug_report in bug_reports.items():
        idx_fold = int(i/avg_fold)

        # data for training
        k_least_vsm_simi = np.array(vsm_simi[i]).argsort()[::-1][-k:]

        # data training label 0
        for j in k_least_vsm_simi:
            vecto = [rank_scores[0][0][i][j],
                     rank_scores[0][1][i][j], rank_scores[0][2][i][j],
                     rank_scores[0][3][i][j], rank_scores[0][4][i][j],
                     rank_scores[0][5][i][j]]
            samples_train[idx_fold].append(vecto)
            labels[idx_fold].append(0)

        # data testing label 1
        fixed_files = bug_report.fixed_files
        idx_right_files = [file_names.index(file) for file in fixed_files
                           if file in file_names]

        for j in idx_right_files:
            vecto = [rank_scores[0][0][i][j],
                     rank_scores[0][1][i][j], rank_scores[0][2][i][j],
                     rank_scores[0][3][i][j], rank_scores[0][4][i][j],
                     rank_scores[0][5][i][j]]
            samples_train[idx_fold].append(vecto)
            labels[idx_fold].append(1)

        # data for testing
        for j in range(0, len(src_files)):
            sample = {
                'report_id': bug_id,
                'file': file_names[j],
                'rVSM_similarity': rank_scores[0][0][i][j],
                'api_similarity': rank_scores[0][1][i][j],
                'class_similarity': rank_scores[0][2][i][j],
                'collab_filter': rank_scores[0][3][i][j],
                'bug_recency': rank_scores[0][4][i][j],
                'bug_frequency': rank_scores[0][5][i][j],
            }
            samples_test[idx_fold].append(sample)

        # get oldest fold
        if idx_fold == 9:
            idx_train = 9*avg_fold + int(remains_number*0.4)

            # training data for fold 10
            if i <= idx_train:
                # data training label 0 for 40% bug reports
                for j in k_least_vsm_simi:
                    vecto = [rank_scores[0][0][i][j],
                             rank_scores[0][1][i][j], rank_scores[0][2][i][j],
                             rank_scores[0][3][i][j], rank_scores[0][4][i][j],
                             rank_scores[0][5][i][j]]
                    fold10_train.append(vecto)
                    labels10.append(0)

                # data training label 1 for 40% bug reports
                for j in idx_right_files:
                    vecto = [rank_scores[0][0][i][j],
                             rank_scores[0][1][i][j], rank_scores[0][2][i][j],
                             rank_scores[0][3][i][j], rank_scores[0][4][i][j],
                             rank_scores[0][5][i][j]]
                    fold10_train.append(vecto)
                    labels10.append(1)

            else:
                # data testing label 1 for 40% bug reports
                for j in range(0, len(src_files)):
                    sample = {
                        'report_id': bug_id,
                        'file': file_names[j],
                        'rVSM_similarity': rank_scores[0][0][i][j],
                        'api_similarity': rank_scores[0][1][i][j],
                        'class_similarity': rank_scores[0][2][i][j],
                        'collab_filter': rank_scores[0][3][i][j],
                        'bug_recency': rank_scores[0][4][i][j],
                        'bug_frequency': rank_scores[0][5][i][j],
                    }
                    fold10_test.append(sample)
        i = i + 1

    return samples_test, samples_train, labels, [fold10_test, fold10_train, labels10]


def helper_collections(samples, bug_reports, only_rvsm=False):
    """ Generates helper function for calculations

    Arguments:
        samples {list} -- samples from features.csv

    Keyword Arguments:
        only_rvsm {bool} -- If True only 'rvsm' features are added to 'sample_dict'. (default: {False})
    """
    sample_dict = {}
    for s in samples:
        sample_dict[s["report_id"]] = []

    for s in samples:
        temp_dict = {}

        values = [
                    float(s["rVSM_similarity"]),
                  float(s["api_similarity"]),
                  float(s["class_similarity"]),
                  float(s["collab_filter"]),
                  float(s["bug_recency"]),
                  float(s["bug_frequency"]),
                  float(s["semantic_similarity"]),
                  ]

        temp_dict[os.path.normpath(s["file"])] = values

        sample_dict[s["report_id"]].append(temp_dict)

    br2files_dict = {}

    for sample in sample_dict:
        br2files_dict[sample] = bug_reports[sample].fixed_files

    return sample_dict, br2files_dict


def topk_accuarcy(src_files, sample_dict, br2files_dict, clf=None):
    """ Calculates top-k accuracies

    Arguments:
        test_bug_reports {list of dictionaries} -- list of all bug reports
        sample_dict {dictionary of dictionaries} -- a helper collection for fast accuracy calculation
        br2files_dict {dictionary} -- dictionary for "bug report id - list of all related files in features.csv" pairs

    Keyword Arguments:
        clf {object} -- A classifier with 'predict()' function. If None, rvsm relevancy is used. (default: {None})
    """
    topk_counters = [0] * 20
    negative_total = 0
    mrr = []
    mean_avgp = []
    file_names = [filename for filename, src in src_files.items()]

    for bug_id in sample_dict:
        dnn_input = []
        corresponding_files = []

        try:
            for temp_dict in sample_dict[bug_id]:
                java_file = list(temp_dict.keys())[0]
                features_for_java_file = list(temp_dict.values())[0]
                dnn_input.append(features_for_java_file)
                corresponding_files.append(java_file)

        except:
            negative_total += 1
            continue

        # Calculate relevancy for all files related to the bug report in features.csv
        # Remember that, in features.csv, there are 50 wrong(randomly chosen) files for each right(buggy)
        relevancy_list = []
        if clf:  # dnn classifier
            relevancy_list = clf.predict_proba(dnn_input)
            relevancy_list = np.array([float(proba[1]) for proba in relevancy_list])

        else:  # rvsm
            relevancy_list = np.array(dnn_input).ravel()

        x = list(np.argsort(relevancy_list))
        x.reverse()
        # print(x)

        temp = []
        # print(br2files_dict[bug_id])
        for y in x:
            temp.append(corresponding_files[y])
        # print(temp)
        relevant_ranks = sorted(temp.index(fixed) + 1
                                for fixed in br2files_dict[bug_id] if fixed in temp)

        if (len(relevant_ranks) == 0):
            continue
        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)

        # MAP
        mean_avgp.append(np.mean([len(relevant_ranks[:j + 1]) / rank
                                  for j, rank in enumerate(relevant_ranks)]))

        # Top-1, top-2 ... top-20 accuracy
        for i in range(1, 21):
            max_indices = np.argpartition(relevancy_list, -i)[-i:]
            # print(max_indices)
            for corresponding_file in np.array(corresponding_files)[max_indices]:
                if str(corresponding_file) in br2files_dict[bug_id]:
                    topk_counters[i - 1] += 1
                    break

    acc_dict = {}
    # print(negative_total)
    # print("MRR", np.mean(mrr))
    # print("MAP", np.mean(mean_avgp))
    for i, counter in enumerate(topk_counters):
        acc = counter / (len(sample_dict) - negative_total)
        acc_dict[i + 1] = round(acc, 3)

    # for k, value in acc_dict.items():
    #     print(str(k) + ": " + str(value))

    return acc_dict, np.mean(mrr), np.mean(mean_avgp)


def validate_param_C(fold10, bug_reports, src_files, *rank_scores):
    fold10_test, fold10_train, labels10 = fold10

    list_mean_avgp = []
    c = [5000000 + i*200000 for i in range(1, 15)]
    for i in range(10):
        start = time.time()
        clf = SVC(kernel='linear', C=c[i], probability=1)  # just a big number
        clf.fit(fold10_train, labels10)

        # These collections are speed up the process while calculating top-k accuracy
        sample_dict, br2files_dict = helper_collections(fold10_test, bug_reports, True)

        acc_dict, mrr, mean_avgp = topk_accuarcy(src_files, sample_dict, br2files_dict, clf)

        end = time.time()

        list_mean_avgp.append([mean_avgp, c[i], end - start])

    return list_mean_avgp


def evaluate(src_files, bug_reports):
    """Estimating linear combination parameters"""
    acc_dict = [0] * 10
    mrr = [0] * 10
    mean_avgp = [0] * 10

    with open('data_output/prepare_data.json', 'r') as file:
        prepare_data = json.load(file)
    samples_test, samples_train, labels, fold10 = prepare_data

    # samples_test_main = []
    # for i in range(10):
    #     samples_test_main.extend(samples_test[i])
    #
    # sample_dict, br2files_dict = helper_collections(samples_test_main, bug_reports, True)
    #
    # acc_dict[0], mrr[0], mean_avgp[0] = topk_accuarcy(src_files, sample_dict, br2files_dict, None)

    # find suitable suboptimalC for training
    # sample_dict = validate_param_C(fold10, bug_reports, src_files, rank_scores)

    for i in range(10):

        # clf = SVC(kernel='linear', C=0.05, probability=1)

        svm = LinearSVC(random_state=0, tol=1e-5, C=.01)
        clf = CalibratedClassifierCV(svm)

        sample = []
        label = []

        for j in range(10):
            if j != i:
                sample.extend(samples_train[j])
                label.extend(labels[j])

        clf.fit(sample, label)

        # These collections are speed up the process while calculating top-k accuracy
        sample_dict, br2files_dict = helper_collections(samples_test[i], bug_reports, True)

        acc_dict[i], mrr[i], mean_avgp[i] = topk_accuarcy(src_files, sample_dict, br2files_dict, clf)

    acc_dict = [list(acc.values())for acc in acc_dict]
    acc_dict_transposed = np.array(acc_dict).T
    acc_dict = list(acc_dict_transposed)
    acc_dict = [mean(acc) for acc in acc_dict]

    return acc_dict, mean(mrr), mean(mean_avgp)


def main():
    with open('data_output/preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)
    with open('data_output/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    acc_dict, mrr, mean_avgp = evaluate(src_files, bug_reports)

    print("MRR", mrr)
    print("MAP", mean_avgp)
    for index, value in enumerate(acc_dict):
        print(str(index) + ": " + str(value))


if __name__ == '__main__':
    main()
