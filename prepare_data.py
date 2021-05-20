import json
import pickle
from sklearn.svm import SVC
import numpy as np


def testing_sample(i, j, features, rank_scores):
    sample = {}
    for o in range(len(features)):
        if features[o] == 1:
            sample['rVSM_similarity'] = rank_scores[0][i][j]
        if features[o] == 2:
            sample['api_similarity'] = rank_scores[1][i][j]
        if features[o] == 3:
            sample['class_similarity'] = rank_scores[2][i][j]
        if features[o] == 4:
            sample['collab_filter'] = rank_scores[3][i][j]
        if features[o] == 5:
            sample['bug_recency'] = rank_scores[4][i][j]
        if features[o] == 6:
            sample['bug_frequency'] = rank_scores[5][i][j]
        if features[o] == 7:
            sample['semantic_similarity'] = rank_scores[6][i][j]

    return sample


def training_sample(i, j, features, rank_scores):
    sample = []
    for o in range(len(features)):
        sample.append(rank_scores[features[o] - 1][i][j])
    return sample


def prepare_data(src_files, bug_reports, *rank_scores):
    k = 300
    num_fold = 10
    file_names = [filename for filename, src in src_files.items()]
    vsm_simi = rank_scores[0]
    i = 0
    avg_fold = int(len(bug_reports) / num_fold) + 1

    samples_test = [[], [], [], [], [], [], [], [], [], []]
    samples_train = [[], [], [], [], [], [], [], [], [], []]
    labels = [[], [], [], [], [], [], [], [], [], []]
    fold10_train = []
    labels10 = []
    fold10_test = []
    remains_number = len(bug_reports) - avg_fold * 9
    features = [1, 2, 3, 4, 5, 6, 7]

    for bug_id, bug_report in bug_reports.items():
        idx_fold = int(i/avg_fold)

        # data for training
        k_least_vsm_simi = np.array(vsm_simi[i]).argsort()[::-1][-k:]

        # data training label 0
        for j in k_least_vsm_simi:
            vecto = training_sample(i, j, features, rank_scores)
            samples_train[idx_fold].append(vecto)
            labels[idx_fold].append(0)

        # data testing label 1
        fixed_files = bug_report.fixed_files
        idx_right_files = [file_names.index(file) for file in fixed_files
                           if file in file_names]

        for j in idx_right_files:
            vecto = training_sample(i, j, features, rank_scores)
            samples_train[idx_fold].append(vecto)
            labels[idx_fold].append(1)

        # data for testing
        for j in range(0, len(src_files)):
            sample = testing_sample(i, j, features, rank_scores)
            sample['report_id'] = bug_id
            sample['file'] = file_names[j]
            samples_test[idx_fold].append(sample)

        # get oldest fold
        if idx_fold == 9:
            idx_train = 9*avg_fold + int(remains_number*0.4)

            # training data for fold 10
            if i <= idx_train:
                # data training label 0 for 40% bug reports
                for j in k_least_vsm_simi:
                    vecto = training_sample(i, j, features, rank_scores)
                    fold10_train.append(vecto)
                    labels10.append(0)

                # data training label 1 for 40% bug reports
                for j in idx_right_files:
                    vecto = training_sample(i, j, features, rank_scores)
                    fold10_train.append(vecto)
                    labels10.append(1)

            else:
                # data testing label 1 for 40% bug reports
                for j in range(0, len(src_files)):
                    sample = testing_sample(i, j, features, rank_scores)
                    sample['report_id'] = bug_id
                    sample['file'] = file_names[j]

                    fold10_test.append(sample)
        i = i + 1

    return samples_test, samples_train, labels, [fold10_test, fold10_train, labels10]


def main():
    with open('data_output/preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)
    with open('data_output/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)
    with open('data_output/vsm_similarity_score_scaled.json', 'r') as file:
        vsm_similarity_score = json.load(file)
    with open('data_output/api_similarity_score_scaled.json', 'r') as file:
        api_similarity_score = json.load(file)
    with open('data_output/class_similarity_score_scaled.json', 'r') as file:
        class_similarity_score = json.load(file)
    with open('data_output/collaborative_filtering_score_scaled.json', 'r') as file:
        collaborative_filtering_score = json.load(file)
    with open('data_output/bug_fixing_recency_score_scaled.json', 'r') as file:
        bug_fixing_recency_score = json.load(file)
    with open('data_output/bug_fixing_frequency_score_scaled.json', 'r') as file:
        bug_fixing_frequency_score = json.load(file)
    with open('data_output/semantic_simi_score_scaled.json', 'r') as file:
        semantic_simi_score = json.load(file)

    samples_test, samples_train, labels, fold10 = prepare_data(src_files, bug_reports, vsm_similarity_score,
                                        api_similarity_score, class_similarity_score,
                                        collaborative_filtering_score, bug_fixing_recency_score,
                                        bug_fixing_frequency_score, semantic_simi_score)

    with open('data_output/prepare_data.json', 'w') as file:
        json.dump([samples_test, samples_train, labels, fold10], file)

    return 0


if __name__ == '__main__':
    main()
