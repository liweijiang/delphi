import os
import sys
from tqdm import tqdm
import argparse
import pandas as pd

sys.path.append(os.getcwd())

from scripts.utils.WANLIScorer import *
from scripts.utils.main_utils import *
from scripts.utils.utils import *


# def filter_by_nli(all_events):
#     wanli_scorer = WANLIScorer()
#     paraphrases_cache = read_json(data_base_path + "cache/paraphrases.json")
#     print(paraphrases_cache)
#
#     nli_to_save = {}
#     for premise in tqdm(all_events):
#         for hypothesis in paraphrases_cache[premise]:
#             # print("-" * 20)
#             prediction = wanli_scorer.get_scores(premise, hypothesis)
#             # print(premise, "|", hypothesis, ":", prediction)
#             prediction["type"] = "event_paraphrase"
#             nli_to_save[premise + " | " + hypothesis] = prediction
#
#             prediction = wanli_scorer.get_scores(hypothesis, premise)
#             # print(hypothesis, "|", premise, ":", prediction)
#             prediction["type"] = "paraphrase_event"
#             nli_to_save[hypothesis + " | " + premise] = prediction
#
#             # print(nli_to_save)
#
#         save_json(data_base_path + "cache/nli.json", nli_to_save)


def generate_nli():
    wanli_scorer = WANLIScorer()
    paraphrases_cache = read_json(data_base_path + "cache/paraphrases.json")
    # print(paraphrases_cache.keys())

    all_events = paraphrases_cache.keys()

    # nli_to_save = {}
    nli_to_save = read_json(data_base_path + "cache/nli.json")
    for premise in tqdm(all_events):
        for hypothesis in paraphrases_cache[premise]:
            if premise + " | " + hypothesis not in nli_to_save:
                prediction = wanli_scorer.get_scores(premise, hypothesis)
                print(premise, "|", hypothesis, ":", prediction)
                prediction["type"] = "event_paraphrase"
                nli_to_save[premise + " | " + hypothesis] = prediction

            if hypothesis + " | " + premise not in nli_to_save:
                prediction = wanli_scorer.get_scores(hypothesis, premise)
                print(hypothesis, "|", premise, ":", prediction)
                prediction["type"] = "paraphrase_event"
                nli_to_save[hypothesis + " | " + premise] = prediction

    save_json(data_base_path + "cache/nli.json", nli_to_save)



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

    generate_nli()


