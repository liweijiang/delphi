import os
import sys
import tqdm
import json
import argparse
import pandas as pd

sys.path.append(os.getcwd())
from scripts.utils.main_utils import *

from scripts.utils.CompositionalityParser import *
from scripts.utils.CacheHandler.ParaphraseCacheHandler import *
from scripts.utils.CacheHandler.DelphiCacheHandler import *
from scripts.utils.CacheHandler.COMETCacheHandler import *

def get_compositions():
    compose_parser = CompositionalityParser()
    compositions_cache = load_compositions_cache()

    df_data = pd.read_csv(input_file, sep="\t")
    events = df_data["event"].tolist()

    for event in events:
        # print("=" * 10)
        # print(event)
        # print("\n")
        if event not in compositions_cache:
            try:
                compositions = compose_parser.parse_event_by_compositionality(event)
                add_compositions_to_cache(compositions_cache, event, compositions)
            except:
                continue

    return

def prepare_paraphrases(all_events):
    # compose_parser = CompositionalityParser()

    paraphrase_cache_handler = ParaphraseCacheHandler()


    for event in tqdm(all_events):
        instance = paraphrase_cache_handler.save_instance(event)
        print(len(instance))
    #     compositions = compose_parser.get_parsed_event(event)
    #
    #     if compositions["adjunct_event"] != None:
    #         print("-" * 20)
    #         print(event)
    #         print(compositions)
    # print(df_data["split"].value_counts())


def prepare_delphi_scores(all_events, device_id, total_num_device):
    num_event_per_device = int(len(all_events) / total_num_device)
    start_idx = device_id * num_event_per_device
    end_idx = (device_id + 1) * num_event_per_device
    filename = f"temp/delphi_scores_{device_id}"
    print(f"device id: {device_id} / {total_num_device}\nstart id: {start_idx}\nend id: {end_idx}\nfilename: {filename}")
    delphi_cache_handler = DelphiCacheHandler(device_id=device_id, filename=filename)
    delphi_cache_handler.save_all_instance(all_events[start_idx: end_idx], save_interval=100)


def filter_paraphrase_by_nli():
    paraphrases_cache = read_json(data_base_path + "cache/paraphrases.json")
    print("total events with paraphrases:", len(paraphrases_cache))
    # nli_map = read_json(data_base_path + "cache/nli.json")
    nli_map = read_json(data_base_path + "cache/nli.json")
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


def compile_all_events(all_events):
    # nov19_2022 v5 2500 events + paraphrases
    paraphrases = read_json(data_base_path + "cache/paraphrases.json")
    all_events_paraphrases = []

    event_paraphrase_pair_to_include = filter_paraphrase_by_nli()
    for event in all_events:
        all_events_paraphrases.append(event)

        for paraphrase in paraphrases[event]:
            if (event, paraphrase) in event_paraphrase_pair_to_include:
                all_events_paraphrases.append(paraphrase)


    # print(all_events_paraphrases)
    df_data = pd.DataFrame()
    df_data["event"] = all_events_paraphrases

    print(df_data)

    # df_data.to_csv(data_base_path + "cache/nov2022/all_events_filtered_by_nli_nov19_2022.csv", index=False)
    df_data.to_csv(data_base_path + "cache/nov2022/all_events_filtered_by_nli_nov20_2022.csv", index=False)


def combine_delphi_files():
    cache_data_names = ["nov2022/delphi_scores", "new_temp/delphi_scores_0",
                        "new_temp/delphi_scores_1", "new_temp/delphi_scores_2",
                        "new_temp/delphi_scores_3", "new_temp/delphi_scores_4",
                        "new_temp/delphi_scores_5", "new_temp/delphi_scores_6",
                        "new_temp/delphi_scores_7", "new_new_temp/delphi_scores_0",
                        # "new_new_temp/delphi_scores_1", "new_new_temp/delphi_scores_2",
                        # "new_new_temp/delphi_scores_3", "new_new_temp/delphi_scores_4",
                        # "new_new_temp/delphi_scores_5", "new_new_temp/delphi_scores_6",
                        # "new_new_temp/delphi_scores_7",
                        ]

    all_data = {}
    for cache_data_name in cache_data_names:
        path = data_base_path + f"cache/{cache_data_name}.json"
        all_data.update(read_json(path))

        print(cache_data_name, len(all_data))
        #
        # for c in all_data:
        #     if "class_label" not in all_data[c]:
        #         print(c)
    print(len(all_data))
    with open(data_base_path + f"cache/nov2022/delphi_scores.json", 'w') as f:
        json.dump(all_data, f)


    # delphi_cache = read_json(data_base_path + f"cache/nov2022/delphi_scores.json")
    # for i in range(8):
    #     with open(data_base_path + f"cache/new_temp/delphi_scores_{i}.json", 'w') as f:
    #         json.dump(delphi_cache, f)


def combine_comet_files():
    cache_data_names = ["new_temp/comet_0", "new_temp/comet_1",
                        "new_temp/comet_2", "new_temp/comet_3", "new_temp/comet_4",
                        "new_temp/comet_5", "new_temp/comet_6", "new_temp/comet_7",
                        "comet", ]

    all_data = {}
    for cache_data_name in cache_data_names:
        path = data_base_path + f"cache/{cache_data_name}.json"
        all_data.update(read_json(path))

        print(cache_data_name, len(all_data))
        #
        # for c in all_data:
        #     if "class_label" not in all_data[c]:
        #         print(c)
    print(len(all_data))
    with open(data_base_path + f"cache/nov2022/comet.json", 'w') as f:
        json.dump(all_data, f)




    # delphi_cache = read_json(data_base_path + f"cache/nov2022/delphi_scores.json")
    # for i in range(8):
    #     with open(data_base_path + f"cache/new_temp/delphi_scores_{i}.json", 'w') as f:
    #         json.dump(delphi_cache, f)


def get_remained_events():
    df_all_events = pd.read_csv(data_base_path + "cache/nov2022/all_events_filtered_by_nli_nov20_2022.csv", sep="\t")
    all_events = df_all_events["event"]
    delphi_cache = read_json(data_base_path + f"cache/nov2022/delphi_scores.json")
    comet_cache = read_json(data_base_path + f"cache/nov2022/comet.json")

    undone_events = [event for event in all_events if event not in delphi_cache]
    df_data = pd.DataFrame()
    df_data["event"] = undone_events
    print(df_data)
    df_data.to_csv(data_base_path + f"cache/nov2022/all_events_filtered_by_nli_nov20_2022_delphi_undone.csv")


    undone_events = [event for event in all_events if event not in comet_cache]
    df_data = pd.DataFrame()
    df_data["event"] = undone_events
    print(df_data)
    df_data.to_csv(data_base_path + f"cache/nov2022/all_events_filtered_by_nli_nov20_2022_comet_undone.csv")

    # df_delphi_remained = pd.read_csv(data_base_path + f"cache/nov2022/all_events_filtered_by_nli_nov20_2022_delphi_undone.json")
    # df_comet_remained = pd.read_csv(data_base_path + f"cache/nov2022/all_events_filtered_by_nli_nov20_2022_comet_undone.json")
    #
    # print(df_delphi_remained.shape)
    # print(df_comet_remained.shape)


def get_remained_comet_events():
    return


def mosaic_02_delphi_scores(device_id, total_num_device):
    df_all_events = pd.read_csv(data_base_path + "cache/nov2022/all_events_filtered_by_nli_nov19_2022.csv", sep="\t")
    all_events = df_all_events["event"].tolist()
    all_events = all_events[:int(len(all_events) / 2)]

    num_event_per_device = int(len(all_events) / total_num_device)
    start_idx = device_id * num_event_per_device
    end_idx = (device_id + 1) * num_event_per_device
    filename = f"new_temp/delphi_scores_{device_id}"
    print(f"device id: {device_id} / {total_num_device}\nstart id: {start_idx}\nend id: {end_idx}\nfilename: {filename}")
    delphi_cache_handler = DelphiCacheHandler(device_id=device_id, filename=filename)
    delphi_cache_handler.save_all_instance(all_events[start_idx: end_idx], save_interval=500)

    # delphi_cache = read_json(data_base_path + f"cache/nov2022/delphi_scores.json")


def mosaic_04_delphi_scores(device_id, total_num_device):
    df_all_events = pd.read_csv(data_base_path + "cache/nov2022/compositions_nov20_2022.csv", sep=",")
    all_events = df_all_events["event"].tolist()
    # all_events = all_events[int(len(all_events) / 2):]
    # print(df_all_events)

    num_event_per_device = int(len(all_events) / total_num_device)
    start_idx = device_id * num_event_per_device
    end_idx = (device_id + 1) * num_event_per_device
    filename = f"new_temp/delphi_scores_{device_id}"
    print(f"device id: {device_id} / {total_num_device}\nstart id: {start_idx}\nend id: {end_idx}\nfilename: {filename}")
    delphi_cache_handler = DelphiCacheHandler(device_id=device_id, filename=filename)
    delphi_cache_handler.save_all_instance(all_events[start_idx: end_idx], save_interval=500)


def mosaic_05_comet(device_id, total_num_device):
    df_all_events = pd.read_csv(data_base_path + "cache/nov2022/compositions_nov20_2022.csv", sep=",")
    all_events = df_all_events["event"].tolist()
    # all_events = all_events[:int(len(all_events) / 2)]

    num_event_per_device = int(len(all_events) / total_num_device)
    start_idx = device_id * num_event_per_device
    end_idx = (device_id + 1) * num_event_per_device
    filename = f"new_temp/comet_{device_id}"
    print(f"device id: {device_id} / {total_num_device}\nstart id: {start_idx}\nend id: {end_idx}\nfilename: {filename}")
    # comet_cache_handler = DelphiCacheHandler(device_id=device_id, filename=filename)
    # comet_cache_handler.save_all_instance(all_events[start_idx: end_idx], save_interval=500)
    comet_cache_handler = COMETCacheHandler(device_id=device_id, filename=filename)
    comet_cache_handler.save_all_instance(all_events[start_idx: end_idx], save_interval=100)




def generate_delphi_remained(device_id):
    df_delphi_remained = pd.read_csv(data_base_path + f"cache/nov2022/all_events_filtered_by_nli_nov20_2022_delphi_undone.csv")
    delphi_events = df_delphi_remained["event"].tolist()

    filename = f"nov2022/delphi_scores"
    print(f"device id: {device_id}\nfilename: {filename}")
    comet_cache_handler = COMETCacheHandler(device_id=device_id, filename=filename)
    comet_cache_handler.save_all_instance(delphi_events, save_interval=100)


def generate_comet_remained(device_id):
    df_comet_remained = pd.read_csv(data_base_path + f"cache/nov2022/all_events_filtered_by_nli_nov20_2022_comet_undone.csv")
    comet_events = df_comet_remained["event"].tolist()

    filename = f"nov2022/comet"
    print(f"device id: {device_id}\nfilename: {filename}")
    comet_cache_handler = COMETCacheHandler(device_id=device_id, filename=filename)
    comet_cache_handler.save_all_instance(comet_events, save_interval=100)



# def get_compositions():


def compile_compositions():
    df_all_events = pd.read_csv(data_base_path + "cache/nov2022/all_events_filtered_by_nli_nov20_2022.csv", sep="\t")
    events = df_all_events["event"]
    # print(len(events))

    compose_parser = CompositionalityParser()
    all_compositions = []
    for event in tqdm(events):
        parsed_events = compose_parser.get_parsed_event(event)

        main_action = parsed_events["main_event_main_clause"]
        relative_clause = parsed_events["main_event_relative_clause"]
        adjunct_event = parsed_events["adjunct_event"]

        # print("-" * 20)
        # print(main_action)
        # print(relative_clause)
        # print(adjunct_event)

        # if main_action != None and main_action not in events:
        #     all_compositions.append(main_action)

        if relative_clause != None:
            all_compositions.append(main_action)
            all_compositions.append(relative_clause)

        if adjunct_event != None:
            all_compositions.append(main_action)
            all_compositions.append(adjunct_event)

        # if main_action != None and relative_clause != None:
        #     if other_action != None:
        #         all_compositions.append(main_action + " " + relative_pronoun + " " + relative_clause)

        # all_compositions.append(event)

    # print(len(list(set(all_compositions))))

    df_data = pd.DataFrame()
    df_data["event"] = list(set(all_compositions))
    # print(df_data)
    df_data.to_csv(data_base_path + f"cache/nov2022/compositions_nov20_2022.csv")

    return list(set(all_compositions))



def filter_paraphrase_by_nli():
    paraphrases_cache = read_json(data_base_path + "cache/paraphrases.json")
    print("total events with paraphrases:", len(paraphrases_cache))
    # nli_map = read_json(data_base_path + "cache/nli.json")
    nli_map = read_json(data_base_path + "cache/nli.json")
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


def compile_all_events():
    # nov19_2022 v5 2500 events + paraphrases

    df_data = pd.read_csv(args.input_file, sep="\t")
    # all_events = df_data["event"].tolist()

    # print(df_data)

    # gold_data_path = data_base_path + "demo/new_test_set/v5.csv"
    # df_data_gold_label = pd.read_csv(gold_data_path, sep=",", low_memory=False)
    # print(df_data_gold_label)

    paraphrases = read_json(data_base_path + "cache/paraphrases.json")
    all_events_paraphrases = []

    all_events = []
    all_version = []
    all_split = []
    for i, row in df_data.iterrows():
        event = row["event"]
        clean_event = row["clean_event"]
        version = row["source"]
        split = row["split"]

        event_selected = None
        if clean_event in paraphrases:
            event_selected = clean_event
        elif event in paraphrases:
            event_selected = event
        else:
            continue

        all_events.append(event_selected)
        all_version.append(version)
        all_split.append(split)



    # for event in all_events:
    #     print(paraphrases[event])



    # event_paraphrase_pair_to_include = filter_paraphrase_by_nli()
    # for event in all_events:
    #     all_events_paraphrases.append(event)
    #
    #     for paraphrase in paraphrases[event]:
    #         if (event, paraphrase) in event_paraphrase_pair_to_include:
    #             all_events_paraphrases.append(paraphrase)
    #
    #
    # # print(all_events_paraphrases)
    # df_data = pd.DataFrame()
    # df_data["event"] = all_events_paraphrases
    #
    # print(df_data)
    #
    # # df_data.to_csv(data_base_path + "cache/nov2022/all_events_filtered_by_nli_nov19_2022.csv", index=False)
    # df_data.to_csv(data_base_path + "cache/nov2022/all_events_filtered_by_nli_nov20_2022.csv", index=False)




def mosaic_delphi_remained(device_id):
    df_all_events = pd.read_csv(data_base_path + "cache/delphi_sequences_remained.csv", sep=",")
    all_events = df_all_events["seq"].tolist()
    print(len(all_events))

    filename = f"nov2022/delphi_scores"
    print(f"filename: {filename}")
    delphi_cache_handler = DelphiCacheHandler(device_id=device_id, filename=filename)
    delphi_cache_handler.regenerate_all_instance(all_events, save_interval=100)


def mosaic_comet_remained(device_id):
    df_all_events = pd.read_csv(data_base_path + "cache/comet_sequences_remained.csv", sep=",")
    all_events = df_all_events["seq"].tolist()
    print(len(all_events))

    filename = f"nov2022/comet"
    print(f"filename: {filename}")
    comet_cache_handler = COMETCacheHandler(device_id=device_id, filename=filename)
    comet_cache_handler.save_all_instance(all_events, save_interval=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--input_file', type=str, help="location of data file",
                        default="data/demo/mturk/split/event_only_v5.csv")
    parser.add_argument('--device_id', type=int, help="device id", default=0)
    parser.add_argument('--total_num_device', type=int, default=8, help="total number device")
    args = parser.parse_args()

    df_data = pd.read_csv(args.input_file, sep="\t")
    # df_data = df_data[df_data["source"] == "v4"]
    all_events = df_data["clean_event"].tolist()

    # compile_all_events()



    # compile_paraphrases(all_events)

    # prepare_paraphrases(all_events)

    # prepare_delphi_scores(all_events, args.device_id, args.total_num_device)

    # compile_all_events(all_events)

    # mosaic_02_delphi_scores(all_events, args.device_id, args.total_num_device)

    # mosaic_02_delphi_scores(args.device_id, args.total_num_device)
    # compile_all_events(all_events)

    # mosaic_05_comet(args.device_id, args.total_num_device)

    # combine_delphi_files()
    # combine_comet_files()

    # generate_delphi_remained(0)
    # generate_comet_remained(1)

    # get_remained_events()

    # get_compositions()
    # compile_compositions()
    # mosaic_04_delphi_scores
    # mosaic_04_delphi_scores(args.device_id, args.total_num_device)


    mosaic_delphi_remained(args.device_id)
    # mosaic_comet_remained(args.device_id)


