import os
import sys
import random
from tqdm import tqdm
import argparse
import pandas as pd
from collections import Counter

sys.path.append(os.getcwd())

from scripts.utils.utils import *
from scripts.utils.main_utils import *
from scripts.utils.CompositionalityParser import *

# "data/demo/mturk/result/v5_clean.csv"

random.seed(10)

def filter_paraphrase_by_nli():
    paraphrases_cache = read_json(data_base_path + "cache/nov2022/paraphrases.json")
    print("total events with paraphrases:", len(paraphrases_cache))
    # nli_map = read_json(data_base_path + "cache/nli.json")
    nli_map = read_json(data_base_path + "cache/nov2022/nli.json")
    print("total event paraphrase nli", len(nli_map))

    event_paraphrase_pair_to_include = []
    event_paraphrase_pair_to_dump = []
    for event in paraphrases_cache:
        for paraphrase in paraphrases_cache[event]:
            key_0 = event + " | " + paraphrase
            key_1 = paraphrase + " | " + event

            if key_0 in nli_map and key_1 in nli_map:
                score_0 = nli_map[key_0]
                score_1 = nli_map[key_1]

                entail_0 = score_0["entailment"]
                entail_1 = score_1["entailment"]

                if entail_0 > 0.5 and entail_1 > 0.5:
                    event_paraphrase_pair_to_include.append((event, paraphrase))
                else:
                    event_paraphrase_pair_to_dump.append((event, paraphrase))
                #     print("-" * 20)
                #     print(key_0, score_0)
                #     print(key_1, score_1)

    print("all event paraphrase pair to include:", len(event_paraphrase_pair_to_include))
    print("all event paraphrase pair to dump:", len(event_paraphrase_pair_to_dump))
    return event_paraphrase_pair_to_include


def compile_gold_labels():
    df_data_gold_v5 = pd.read_csv("data/demo/mturk/result/v5_clean.csv")
    df_data_gold_v5["event"] = df_data_gold_v5["event"].apply(normalize_event)

    df_data_gold_v23 = pd.read_csv(data_base_path + "demo/new_test_set/v5.csv")

    paraphrases_cache = read_json(data_base_path + "cache/nov2022/paraphrases.json")

    # print(df_data_gold_v5)
    # print(df_data_gold_v23)

    ############### compile gold events with paraphrases ###############
    events_without_paraphrases = []
    # map_raw_class_labels = {}
    # map_class_label = {}
    # map_agreement_rate = {}

    all_events = []
    all_raw_class_labels = []
    all_class_label = []
    all_agreement_rate = []
    for i, row in df_data_gold_v23.iterrows():
        event = row["event"]
        raw_class_labels = row["raw_class_labels"]
        class_label = row["class_label"]
        agreement_rate = row["agreement_rate"]

        if event not in paraphrases_cache:
            events_without_paraphrases.append(event)
        else:
            all_events.append(event)
            all_raw_class_labels.append(raw_class_labels)
            all_class_label.append(class_label)
            all_agreement_rate.append(agreement_rate)

            # map_raw_class_labels[event] = raw_class_labels
            # map_class_label[event] = class_label
            # map_agreement_rate[event] = agreement_rate

    for i, row in df_data_gold_v5.iterrows():
        event = row["event"]
        raw_class_labels = row["raw_class_labels"]
        class_label = row["class_label"]
        agreement_rate = row["agreement_rate"]

        if event not in paraphrases_cache:
            events_without_paraphrases.append(event)
        else:
            all_events.append(event)
            all_raw_class_labels.append(raw_class_labels)
            all_class_label.append(class_label)
            all_agreement_rate.append(agreement_rate)

            # map_raw_class_labels[event] = raw_class_labels
            # map_class_label[event] = class_label
            # map_agreement_rate[event] = agreement_rate

    print("events_without_paraphrases:", len(events_without_paraphrases))
    print("all_events:", len(all_events))


    df_data_to_save = pd.DataFrame()
    df_data_to_save["event"] = all_events
    df_data_to_save["raw_class_labels"] = all_raw_class_labels
    df_data_to_save["class_label"] = all_class_label
    df_data_to_save["agreement_rate"] = all_agreement_rate
    df_data_to_save.to_csv(data_base_path + "cache/data_gold_labels.csv", index=False)

    return all_events


def compile_paraphrase(all_events):
    paraphrases_cache = read_json(data_base_path + "cache/nov2022/paraphrases.json")

    event_paraphrase_pair_to_include = filter_paraphrase_by_nli()

    ############### compile events + filtered paraphrases ###############
    map_paraphrases = {event: [] for event in all_events}
    for event in all_events:
        for paraphrase in paraphrases_cache[event]:
            if (event, paraphrase) in event_paraphrase_pair_to_include:
                map_paraphrases[event].append(paraphrase)

    total_paraphrases = 0
    for event in map_paraphrases:
        total_paraphrases += len(map_paraphrases[event])
        print(len(map_paraphrases[event]))

    print("average paraphrase per instance:", total_paraphrases / len(map_paraphrases))

    save_json(data_base_path + "cache/paraphrases_filtered_by_nli.json", map_paraphrases)
    return map_paraphrases


def compile_all_sequences():
    # df_data = pd.read_csv(data_base_path + "cache/all_sequences.csv")
    # print(df_data)

    compose_parser = CompositionalityParser()
    paraphrases_cache = read_json(data_base_path + "cache/paraphrases_filtered_by_nli.json")

    all_seqs = []
    for event in paraphrases_cache:
        all_seqs += [event]
        all_seqs += paraphrases_cache[event]
    all_compositions = []
    for seq in tqdm(all_seqs):
        parsed_events = compose_parser.get_parsed_event(seq)

        main_event = parsed_events["main_event"]
        main_action = parsed_events["main_event_main_clause"]
        relative_clause = parsed_events["main_event_relative_clause"]
        adjunct_event = parsed_events["adjunct_event"]

        # print("-" * 20)
        # print(main_action)
        # print(relative_clause)
        # print(adjunct_event)

        # if main_action != None and main_action not in events:
        #     all_compositions.append(main_action)

        # if relative_clause != None:
        #     all_compositions.append(main_event)
        #     all_compositions.append(main_action)
        #     all_compositions.append(relative_clause)
        #
        # if adjunct_event != None:
        #     all_compositions.append(main_event)
        #     all_compositions.append(adjunct_event)
            # print(main_action)
            # print(adjunct_event)

        if main_event != None:
            all_compositions.append(main_event)

        if main_action != None:
            all_compositions.append(main_action)

        if relative_clause != None:
            all_compositions.append(relative_clause)

        if adjunct_event != None:
            all_compositions.append(adjunct_event)

    all_seqs += all_compositions
    all_seqs = list(set(all_seqs))

    df_data_to_save = pd.DataFrame()
    df_data_to_save["seq"] = all_seqs
    # print(df_data_to_save)
    df_data_to_save.to_csv(data_base_path + "cache/all_sequences.csv", index=False)


def compile_delphi_remained_to_generate():
    all_seqs = pd.read_csv(data_base_path + "cache/all_sequences.csv")["seq"].tolist()
    delphi_cache = read_json(data_base_path + "cache/nov2022/delphi_scores.json")

    seqs_remained = []
    for seq in tqdm(all_seqs):
        if seq not in delphi_cache:
            seqs_remained.append(seq)
            # print(seq)
        elif "prob_1" not in delphi_cache[seq]:
            seqs_remained.append(seq)

    print(len(seqs_remained))

    df_data_to_save = pd.DataFrame()
    df_data_to_save["seq"] = seqs_remained
    df_data_to_save.to_csv(data_base_path + "cache/delphi_sequences_remained.csv", index=False)
    print(df_data_to_save)

def compile_comet_remained_to_generate():
    all_seqs = pd.read_csv(data_base_path + "cache/all_sequences.csv")["seq"].tolist()
    comet_cache = read_json(data_base_path + "cache/nov2022/comet.json")

    seqs_remained = []
    comet_cache_subset = {}
    for seq in tqdm(all_seqs):
        if seq not in comet_cache:
            seqs_remained.append(seq)
            # print(seq)
        elif "prob_1" in comet_cache[seq]:
            seqs_remained.append(seq)
        else:
            comet_cache_subset[seq] = comet_cache[seq]

    print(len(seqs_remained))

    df_data_to_save = pd.DataFrame()
    df_data_to_save["seq"] = seqs_remained
    df_data_to_save.to_csv(data_base_path + "cache/comet_sequences_remained.csv", index=False)
    print(df_data_to_save)
    # save_json(data_base_path + "cache/comet_subset.json", comet_cache_subset)



def compile_delphi_subset():
    print("-" * 10, "compile delphi subset", "-" * 10)
    all_seqs = pd.read_csv(data_base_path + "cache/all_sequences.csv")["seq"].tolist()
    delphi_cache = read_json(data_base_path + "cache/nov2022/delphi_scores.json")
    print(len(delphi_cache))

    seqs_remained = []
    delphi_cache_subset = {}
    for seq in tqdm(all_seqs):
        if seq not in delphi_cache:
            seqs_remained.append(seq)
            # print(seq)
        else:
            delphi_cache_subset[seq] = delphi_cache[seq]

    save_json(data_base_path + "cache/delphi_subset.json", delphi_cache_subset)


def compile_comet_subset():
    print("-" * 10, "compile comet subset", "-" * 10)
    all_seqs = pd.read_csv(data_base_path + "cache/all_sequences.csv")["seq"].tolist()
    comet_cache = read_json(data_base_path + "cache/nov2022/comet.json")

    seqs_remained = []
    comet_cache_subset = {}
    for seq in tqdm(all_seqs):
        if seq not in comet_cache:
            seqs_remained.append(seq)
            # print(seq)
        else:
            comet_cache_subset[seq] = comet_cache[seq]

    save_json(data_base_path + "cache/comet_subset.json", comet_cache_subset)


def create_splits_old():
    events_to_exclude = [ "playing a video game with tamim",
                          "You use the word pog and poggers",
                          "Beitar Jerusalem football club",
                          "Buying a lootbox in my favourite online game",
                          "Lolong is going to beat Ohma",
                          "Up tilting Falcon 4 times in a row",
                          "Playing new world with janus and cake",
                          "drawing marianne and amita as sumo wrestlers",
                          "Up tilting Falcon 4 times in a row",
                          "Jail people who put ananas on pizzas",
                          "say I love you to kenza sebti",
                          "Someone is rubbing his hands intensly",
                          "Anything to do with Rachel",
                          "Solo open world pvp like a chad",
                          "Allen institute for artificial intelligence",
                          "Why is veganism not morally obligatory",
                          "asking which moron decided this machines moral compass",
                          "Make a robot to evaluate human morals",
                          "tickling jimins balls backstage at the concert for 7 hours",
                          "Triggering a Friend by killing his imagenary waifu",
                          "Telling everyone that Andrej got peged with a cone",
                          "Giving a nerd a swirlie because he enjoys Heidegger",
                          "playing fire emblem heroes",
                          "Being a bot that evaluates the morality of statements",
                          "Smoking weed because you are snoop dogg",
                          "Sending a kid directly to hell for losing a Yugioh duel",
                          "Committing targeted genocide towards children in Fortnite",
                          "let my mom do the dish when  im not working",
                          "Stealing a road sign for decorating my uni flat",
                          "Contributing to a mural for Watsuki",
                          "Watching Jojo's bizarre adventure"]


    data_path = data_base_path + "cache/nov2022/data_gold_labels.csv"
    df_data = pd.read_csv(data_path)
    print(df_data.shape)
    df_data = df_data[~df_data["event"].isin(events_to_exclude)]
    print(df_data.shape)
    df_data = df_data.drop_duplicates(subset=["event"])
    print(df_data.shape)

    split_labels = ["train" for _ in range(int(0.6 * df_data.shape[0]) + 2)] \
                   + ["dev" for _ in range(int(0.2 * df_data.shape[0]))] \
                   + ["test" for _ in range(int(0.2 * df_data.shape[0]))]

    random.shuffle(split_labels)
    print(len(split_labels))

    df_data["split"] = split_labels

    print(df_data["split"].value_counts())


    for agreement_rate in [1, 0.8, 0.6]:
        print("-" * 10, agreement_rate, "-" * 10)
        df_data_sub = df_data[df_data["agreement_rate"] == agreement_rate]
        print(df_data_sub["split"].value_counts())

    df_data.to_csv(data_base_path + "cache/data_gold_labels.csv", index=False)


def create_splits():
    data_path = data_base_path + "cache/nov2022/data_gold_labels_old_split.csv"
    df_data = pd.read_csv(data_path)
    # print(df_data.shape)
    df_data = df_data.drop_duplicates(subset=["event"])
    # print(df_data.shape)

    df_data_version = pd.read_csv("data/demo/mturk/split/event_only_v5.csv", sep="\t")
    # print(df_data_version["source"].value_counts())
    # v23_events = df_data_version[df_data_version["source"] == "v23"]["clean_event"].tolist()
    v4_events = df_data_version[df_data_version["source"] == "v4"]["clean_event"].tolist()


    df_data_v23 = df_data[~df_data["event"].isin(v4_events)]
    df_data_v23["split"] = "train"


    df_data_v4 = df_data[df_data["event"].isin(v4_events)]

    df_data_v4_dev = df_data_v4[df_data_v4["split"].isin(["train", "dev"])]
    df_data_v4_test = df_data_v4[df_data_v4["split"].isin(["test"])]

    df_data_v4_dev["split"] = "dev"
    df_data_v4_test["split"] = "test"

    print(df_data_v4.shape)
    print(df_data_v4_dev.shape)
    print(df_data_v4_test.shape)


    # split_labels = ["dev" for _ in range(int(0.5 * df_data_v4.shape[0]) + 1)] + ["test" for _ in range(int(0.5 * df_data_v4.shape[0]))]
    # random.shuffle(split_labels)
    # df_data_v4["split"] = split_labels

    print(df_data)

    df_data = pd.concat([df_data_v4_dev, df_data_v4_test, df_data_v23], ignore_index=True)
    print(df_data)

    for agreement_rate in [1, 0.8, 0.6]:
        print("-" * 10, agreement_rate, "-" * 10)
        df_data_sub = df_data[df_data["agreement_rate"] == agreement_rate]
        print(df_data_sub["split"].value_counts())

    # df_data.to_csv(data_base_path + "cache/data_gold_labels.csv", index=False)


def data_info():
    df_data =  pd.read_csv(data_base_path + "cache/data_gold_labels.csv")

    # for agreement_rate in [1, 0.8, 0.6]:
    #     print("-" * 10, agreement_rate, "-" * 10)
    #     df_data_sub = df_data[df_data["agreement_rate"] == agreement_rate]
    #     print(df_data_sub["split"].value_counts())

    print("-" * 10, "all", "-" * 10)
    print(df_data["split"].value_counts())

    print("-" * 10, "certain", "-" * 10)
    df_data_sub = df_data[df_data["agreement_rate"].isin([1])]
    print(df_data_sub["split"].value_counts())


    print("-" * 10, "ambiguous", "-" * 10)
    df_data_sub = df_data[df_data["agreement_rate"].isin([0.6, 0.8])]
    print(df_data_sub["split"].value_counts())



if __name__ == "__main__":
    # all_events = compile_gold_labels()
    # compile_paraphrase(all_events)
    # compile_all_sequences()

    # compile_delphi_remained_to_generate()
    # compile_comet_remained_to_generate()

    # compile_delphi_subset()
    # compile_comet_subset()

    # create_splits()
    data_info()



