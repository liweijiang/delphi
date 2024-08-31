import os
import sys
import argparse
import pandas as pd

sys.path.append(os.getcwd())

from scripts.utils.utils import *
from scripts.utils.CacheHandler.ParaphraseCacheHandler import *
from scripts.utils.WANLIScorer import *

def generate_paraphrases():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--section_id", type=int, default=0)
    args = parser.parse_args()
    section_id = args.section_id

    events = []
    for split in ["remainedtest"]:  # , "validation" "validation"
        input_file = data_base_path + f"cache_norm_bank/events/clean_{split}.moral_acceptability.tsv"
        df_data = pd.read_csv(input_file, sep="\t")
        events.extend(df_data["clean_event"].tolist())

    section_num = int(len(events) / 3)
    start_id = section_id * section_num
    end_id = (section_id + 1) * section_num
    events = events[start_id: end_id]

    print(start_id, end_id)

    cache_handler = ParaphraseCacheHandler(filename=f"paraphrases/paraphrases_{section_id}",
                                           cache_dir="cache_norm_bank",
                                           num_paraphrases=10)
    for e in tqdm(events):
        print(len(e))
        cache_handler.save_instance(e)


def generate_nli(dataset_name="3000validation"):
    paraphrases_cache = read_json(f"/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache_norm_bank/paraphrases/{dataset_name}_paraphrases.json")
    wanli_scorer = WANLIScorer()

    input_file = data_base_path + f"cache_norm_bank/events/clean_{dataset_name}.moral_acceptability.tsv"
    df_data = pd.read_csv(input_file, sep="\t")
    all_events = df_data["clean_event"].tolist()

    nli_data_path = data_base_path + "cache_norm_bank/nli/{dataset_name}_nli.json"
    nli_to_save = {}
    if os.path.exists(nli_data_path):
        nli_to_save = read_json(nli_data_path)
    for premise in tqdm(all_events):
        for hypothesis in paraphrases_cache[premise]:
            if premise + " | " + hypothesis not in nli_to_save:
                prediction = wanli_scorer.get_scores(premise, hypothesis)
                # print(premise, "|", hypothesis, ":", prediction)
                prediction["type"] = "event_paraphrase"
                nli_to_save[premise + " | " + hypothesis] = prediction

            if hypothesis + " | " + premise not in nli_to_save:
                prediction = wanli_scorer.get_scores(hypothesis, premise)
                # print(hypothesis, "|", premise, ":", prediction)
                prediction["type"] = "paraphrase_event"
                nli_to_save[hypothesis + " | " + premise] = prediction

    save_json(nli_data_path, nli_to_save)

def filter_by_nli():
    split = "3000test"
    paraphrases_cache = read_json(f"/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache_norm_bank/paraphrases/{split}_paraphrases.json")

    nli_cache_path = data_base_path + f"cache_norm_bank/nli/{split}_nli.json"
    nli_cache = read_json(nli_cache_path)


    input_file = data_base_path + f"cache_norm_bank/events/clean_{split}.moral_acceptability.tsv"
    df_data = pd.read_csv(input_file, sep="\t")
    events = df_data["clean_event"].tolist()

    for THRESHOLD in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        all_data_to_include = {}
        data_to_include_count = 0
        for e in events:
            e_paraphrases = paraphrases_cache[e]
            e_paraphrases_to_include = []
            for p in e_paraphrases:
                e_p = nli_cache[e + " | " + p]
                p_e = nli_cache[p + " | " + e]

                entail_e_p = e_p["entailment"]
                entail_p_e = p_e["entailment"]

                # if entail_e_p > 0.5 and entail_p_e > 0.5:
                #     e_paraphrases_to_include.append(p)

                if entail_e_p > THRESHOLD and entail_p_e > THRESHOLD:
                    e_paraphrases_to_include.append(p)
                    data_to_include_count += 1
            all_data_to_include[e] = e_paraphrases_to_include

            # print(len(e_paraphrases), len(e_paraphrases_to_include))

        print(THRESHOLD, ":", data_to_include_count)
        save_json(f"/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache_norm_bank/paraphrases/{split}_paraphrases_filtered_by_nli_{THRESHOLD}.json", all_data_to_include)

if __name__ == "__main__":
    # generate_nli(dataset_name="3000validation")
    filter_by_nli()

