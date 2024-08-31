import os
import sys
import random
import pandas as pd

# sys.path.append(os.getcwd())

sys.path.append("/Users/liweijiang/Desktop/delphi_algo/scripts/utils")
# print(os.getcwd())

from main_utils import *

random.seed(10)


def get_event_len(e):
    return len(e)

def filter_by_keywords_text_label(e):
    kw_to_exclude = ["yes,", "no,"]
    return not any(kw in e.lower() for kw in kw_to_exclude)

def filter_by_keywords_event(e):
    kw_to_exclude = ["?", "\"", "yeri", "izone", "aaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                     "delphi", "is a ", "hhhhhooooooo", "poiuytpoiuytrpoiuytrezapoiuytrezaqwxcvbnracist",
                     "rhum", "..", "ben"]
    return not any(kw in e.lower() for kw in kw_to_exclude)

def filter_by_startswith_keywords_event(e):
    kw_to_exclude = ["can", "is ", "should", "how", "what", "when",
                     "are ", "why"]
    return not any(e.lower().startswith(kw) for kw in kw_to_exclude)

def is_in_english(e):
    try:
        e.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def get_upper_letter_count(e):
    return sum(1 for c in (e[0].lower() + e[1:]) if c.isupper())

def main():
    df_data_done = pd.read_csv("/data/demo/mturk/result/v2_v3_comprehensible_clean.csv", sep=",")
    event_done = df_data_done["event"].tolist()

    df_data = pd.read_csv("/data/demo/demo_examples_102721_delphi.csv", sep="\t")
    df_data["action1_len"] = df_data["action1"].apply(get_event_len)
    df_data = df_data.dropna(subset=["action1", "text_label"])

    df_data = df_data[~df_data["action1"].isin(event_done)]
    df_data = df_data[df_data["action1_len"] > 25]
    df_data = df_data[df_data["action1_len"] < 200]
    df_data = df_data[df_data["text_label"].apply(filter_by_keywords_text_label)]
    df_data = df_data[df_data["action1"].apply(filter_by_keywords_event)]
    df_data = df_data[df_data["action1"].apply(filter_by_startswith_keywords_event)]
    df_data = df_data[df_data["action1"].apply(is_in_english)]

    df_data["clean_event"] = df_data["action1"].apply(normalize_event)

    # print(df_data) any(kw in e.lower() for kw in remove_keywords_list)

    events_to_annotate = df_data["clean_event"].value_counts()[:10000].keys().tolist()
    # print(events_to_annotate)


    df_data["question1"] = df_data["clean_event"]

    df_data_selected = df_data[df_data["clean_event"].isin(events_to_annotate)]
    df_data_selected = df_data_selected[["action1", "class_label", "text_label", "clean_event", "question1"]]

    df_data_selected = df_data_selected.drop_duplicates(subset=["clean_event"])
    df_data_selected.to_csv("/Users/liweijiang/Desktop/delphi_algo/data/demo/mturk/input/demo_events_to_annotate_v4.csv", index=False)

    # df_data = pd.read_csv("/Users/liweijiang/Desktop/delphi_algo/data/demo/mturk/input/demo_events_to_annotate_v4.csv", sep=",")
    # df_data = df_data.rename(columns={"action1": "raw_event"})
    # df_data.to_csv("/Users/liweijiang/Desktop/delphi_algo/data/demo/mturk/input/demo_events_to_annotate_v4.csv", index=False)

def compile_annotated_data():
    df_data = pd.read_csv("/data/demo/mturk/result/demo_events_to_annotate_v4.csv",
                          sep=",")

    df_data_done = pd.read_csv(
        "/data/demo/mturk/result/v2_v3_comprehensible_clean.csv", sep=",")
    event_done = df_data_done["event"].tolist()

    headers = ["Input.action1", "Input.class_label",
               "Input.text_label", "Input.clean_event",
               "Answer.controversial_1", "Answer.lewd_1",
               "Answer.feedback", "Answer.is_action_1",
               "Answer.make_sense_1", "Answer.privacy_1"]

    rename_map = {"Input.action1": "event",
                  "Input.class_label": "class_label",
                  "Input.text_label": "text_label",
                  "Input.clean_event": "clean_event",
                  "Answer.controversial_1": "is_controversial",
                  "Answer.lewd_1": "is_lewd",
                  "Answer.feedback": "feedback",
                  "Answer.is_action_1": "is_action",
                  "Answer.make_sense_1": "is_make_sense",
                  "Answer.privacy_1": "is_privacy"}

    df_data = df_data[headers]
    df_data = df_data.rename(columns=rename_map)
    df_data = df_data[~df_data["event"].isin(event_done)]
    df_data = df_data[df_data["feedback"] == "{}"]
    df_data = df_data[df_data["is_make_sense"] == 1.0]
    df_data = df_data[df_data["is_lewd"] == -1.0]
    df_data = df_data[df_data["is_action"] == 1.0]
    df_data = df_data[df_data["is_privacy"] == -1.0]
    df_data = df_data[df_data["event"].apply(filter_by_keywords_event)]

    df_data["event_len"] = df_data["event"].apply(get_event_len)
    df_data = df_data[df_data["event_len"] < 120]
    df_data = df_data[df_data["text_label"].apply(filter_by_keywords_text_label)]

    df_data["upper_letter_count"] = df_data["event"].apply(get_upper_letter_count)
    df_data = df_data[df_data["upper_letter_count"] < 2]

    df_data = df_data.sort_values(by=["event_len"])
    print(df_data["event_len"].mean())

    df_data_non_controversial = df_data[df_data["is_controversial"] == -1.0]
    df_data_non_controversial = df_data_non_controversial.sample(n=2500, random_state=0)
    df_data_controversial = df_data[df_data["is_controversial"] == 1.0]
    df_data_controversial = df_data_controversial.sample(n=4000 - df_data_non_controversial.shape[0], random_state=0)

    df_data_selected = pd.concat([df_data_non_controversial, df_data_controversial], ignore_index=True, sort=False)
    df_data_selected.to_csv("/Users/liweijiang/Desktop/delphi_algo/data/demo/mturk/input/demo_events_to_annotate_v4_selected.csv", index=False)
    input_events = df_data_selected["event"].tolist()

    df_data_input = pd.DataFrame()
    for i in range(4):
        df_data_input[f"action{i+1}"] = input_events[i * 1000: (i+1) * 1000]
    df_data_input.to_csv("/Users/liweijiang/Desktop/delphi_algo/data/demo/mturk/input/demo_events_to_annotate_v4_selected_input.csv",
        index=False)

    # print(len(event_done), len(set(event_done)))
    # print(len(input_events), len(set(input_events)))
    # print(len(set(event_done + input_events)))


def compile_v234_data():
    df_data_v4 = pd.read_csv(
        "/data/demo/mturk/input/demo_events_to_annotate_v4_selected.csv", sep=",")
    event_v4 = df_data_v4["event"].tolist()

    df_data_v2_v3 = pd.read_csv(
        "/data/demo/mturk/result/v2_v3_comprehensible_clean.csv", sep=",")
    event_v2_v3 = df_data_v2_v3["event"].tolist()
    # print(len(event_v4))
    # print(len(event_v2_v3))

    all_events = event_v2_v3 + event_v4
    random.shuffle(all_events)

    # train_event = all_events[0: int(0.4 * len(all_events))]
    # dev_event = all_events[int(0.4 * len(all_events)): int(0.7 * len(all_events))]
    # test_event = all_events[int(0.7 * len(all_events)):]

    train_split_labels = ["train" for _ in range(int(0.5 * len(all_events)) + 2)] \
                         + ["dev" for _ in range(int(0.25 * len(all_events)))] \
                         + ["test" for _ in range(int(0.25 * len(all_events)))]

    df_data = pd.DataFrame()
    df_data["event"] = all_events
    df_data["split"] = train_split_labels

    df_data["source"] = "v23"
    df_data.loc[df_data["event"].isin(event_v4), "source"] = "v4"

    # df_data["source"] = df_data["source"]
    df_data["clean_event"] = df_data["event"].apply(normalize_event)

    # print(df_data["source"].value_counts())
    df_data.to_csv("/Users/liweijiang/Desktop/delphi_algo/data/demo/mturk/split/event_only_v234.csv", index=False, sep="\t")
    # print(len(train_event))
    # print(len(dev_event))
    # print(len(test_event))


def compile_v5_data():
    """
    remove some controversial data from v4, and combine with v23 data to form v5
    """
    df_data_v4 = pd.read_csv(
        "/data/demo/mturk/input/demo_events_to_annotate_v4_selected.csv", sep=",")
    df_data_v4 = df_data_v4[df_data_v4["is_controversial"] == -1.0]
    event_v4 = df_data_v4["event"].tolist()

    df_data_v2_v3 = pd.read_csv(
        "/data/demo/mturk/result/v2_v3_comprehensible_clean.csv", sep=",")
    event_v2_v3 = df_data_v2_v3["event"].tolist()
    # print(len(event_v4))
    # print(len(event_v2_v3))

    all_events = event_v2_v3 + event_v4
    random.shuffle(all_events)

    # train_event = all_events[0: int(0.4 * len(all_events))]
    # dev_event = all_events[int(0.4 * len(all_events)): int(0.7 * len(all_events))]
    # test_event = all_events[int(0.7 * len(all_events)):]

    train_split_labels = ["train" for _ in range(int(0.5 * len(all_events)) + 2)] \
                         + ["dev" for _ in range(int(0.25 * len(all_events)))] \
                         + ["test" for _ in range(int(0.25 * len(all_events)))]

    df_data = pd.DataFrame()
    df_data["event"] = all_events
    df_data["split"] = train_split_labels

    df_data["source"] = "v23"
    df_data.loc[df_data["event"].isin(event_v4), "source"] = "v4"

    df_data["clean_event"] = df_data["event"].apply(normalize_event)

    # df_data["source"] = df_data["source"]

    # print(df_data["source"].value_counts())
    # print(df_data["split"].value_counts())
    df_data.to_csv("/Users/liweijiang/Desktop/delphi_algo/data/demo/mturk/split/event_only_v5.csv", index=False, sep="\t")
    # print(len(train_event))
    # print(len(dev_event))
    # print(len(test_event))


if __name__ == "__main__":
    # compile_v234_data()
    compile_v5_data()

