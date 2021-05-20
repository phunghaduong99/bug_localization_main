import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json


def scaling_features(data, scaler):
    data_T = np.array(data).T
    scaler.fit(data_T)
    return scaler.transform(data_T).T.tolist()


def main():
    with open('data_output/vsm_similarity.json', 'r') as file:
        vsm_similarity_score = json.load(file)
    with open('data_output/api_similarity.json', 'r') as file:
        api_similarity_score = json.load(file)
    with open('data_output/class_similarity_main.json', 'r') as file:
        class_similarity_score = json.load(file)
    with open('data_output/collaborative_filtering_score.json', 'r') as file:
        collaborative_filtering_score = json.load(file)
    with open('data_output/bug_fixing_recency.json', 'r') as file:
        bug_fixing_recency_score = json.load(file)
    with open('data_output/bug_fixing_frequency.json', 'r') as file:
        bug_fixing_frequency_score = json.load(file)
    with open('data_output/semantic_similarity.json', 'r') as file:
        semantic_simi_score = json.load(file)

    scaler = MinMaxScaler()

    vsm_similarity_score_scaled = scaling_features(vsm_similarity_score, scaler)
    api_similarity_score_scaled = scaling_features(api_similarity_score, scaler)
    class_similarity_score_scaled = scaling_features(class_similarity_score, scaler)
    collaborative_filtering_score_scaled = scaling_features(collaborative_filtering_score, scaler)
    bug_fixing_recency_score_scaled = scaling_features(bug_fixing_recency_score, scaler)
    bug_fixing_frequency_score_scaled = scaling_features(bug_fixing_frequency_score, scaler)
    semantic_simi_score_scaled = scaling_features(semantic_simi_score, scaler)

    with open('data_output/vsm_similarity_score_scaled.json', 'w') as file:
        json.dump(vsm_similarity_score_scaled, file)
    with open('data_output/api_similarity_score_scaled.json', 'w') as file:
        json.dump(api_similarity_score_scaled, file)
    with open('data_output/class_similarity_score_scaled.json', 'w') as file:
        json.dump(class_similarity_score_scaled, file)
    with open('data_output/collaborative_filtering_score_scaled.json', 'w') as file:
        json.dump(collaborative_filtering_score_scaled, file)
    with open('data_output/bug_fixing_recency_score_scaled.json', 'w') as file:
        json.dump(bug_fixing_recency_score_scaled, file)
    with open('data_output/bug_fixing_frequency_score_scaled.json', 'w') as file:
        json.dump(bug_fixing_frequency_score_scaled, file)
    with open('data_output/semantic_simi_score_scaled.json', 'w') as file:
        json.dump(semantic_simi_score_scaled, file)
    return 0


if __name__ == '__main__':
    main()
