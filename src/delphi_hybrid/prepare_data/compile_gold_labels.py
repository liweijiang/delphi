import os
import sys
from collections import Counter

sys.path.append(os.getcwd())

from scripts.utils.utils import *

def get_maj_vote_class_label(raw_class_labels):
    class_label_counts = Counter(raw_class_labels)

    if len(set(raw_class_labels)) == 1:
        return raw_class_labels[0]
    elif len(set(raw_class_labels)) == 2:
        if -1 not in raw_class_labels:
            count_0 = class_label_counts[0]
            count_1 = class_label_counts[1]

            if count_0 > count_1:
                return 0
            else:
                return 1
        else:
            count_minus_1 = class_label_counts[-1]
            if 0 in class_label_counts:
                count_POS = class_label_counts[0]
            else:
                count_POS = class_label_counts[1]

            if count_minus_1 > count_POS:
                return -1
            else:
                if 0 in class_label_counts:
                    return 0
                else:
                    return 1
    else:
        count_minus_1 = class_label_counts[-1]
        count_0 = class_label_counts[0]
        count_1 = class_label_counts[1]

        if count_minus_1 == 3:
            return -1
        elif count_minus_1 == 2:
            if count_0 == 2:
                return 0
            elif count_1 == 2:
                return 1
        else:
            if count_0 == 3:
                return 0
            elif count_1 == 3:
                return 1
            else:
                return 0


def get_agreement_rate(raw_class_labels):
    class_label_counts = Counter(raw_class_labels)

    if -1 in class_label_counts:
        count_minus_1 = class_label_counts[-1]
        if count_minus_1 in [2, 3]:
            return 3/5
        elif count_minus_1 in [5]:
            return 1
        else:
            return 4/5
    else:
        return 1


def compile_gold_labels_v5():
    data_path = "data/demo/mturk/result/v5.csv"
    df_data = pd.read_csv(data_path)

    # print(df_data)

    event_to_raw_labels = {}
    all_evants = []
    for i, row in df_data.iterrows():
        for j in range(4):
            event = row[f"Input.action{j+1}"]
            label = row[f"Answer.judgment_{j+1}"]

            all_evants.append(event)

            # if label not in [0, 1, -1]:
            #     print(label)
            if event not in event_to_raw_labels:
                event_to_raw_labels[event] = [label]
            else:
                event_to_raw_labels[event].append(label)

    all_evants = list(set(all_evants))
    print(len(event_to_raw_labels))


    all_class_labels = []
    all_agreement_rates = []
    all_raw_class_labels = []
    for event in all_evants:
        class_label = get_maj_vote_class_label(event_to_raw_labels[event])
        agreement_rate = get_agreement_rate(event_to_raw_labels[event])

        all_class_labels.append(class_label)
        all_agreement_rates.append(agreement_rate)
        all_raw_class_labels.append(event_to_raw_labels[event])


    df_data_to_save = pd.DataFrame()
    df_data_to_save["event"] = all_evants
    df_data_to_save["raw_class_labels"] = all_raw_class_labels
    df_data_to_save["agreement_rate"] = all_agreement_rates
    df_data_to_save["class_label"] = all_class_labels

    print(df_data_to_save)

    df_data_to_save.to_csv("data/demo/mturk/result/v5_clean.csv", index=False)


if __name__ == "__main__":
    compile_gold_labels_v5()

