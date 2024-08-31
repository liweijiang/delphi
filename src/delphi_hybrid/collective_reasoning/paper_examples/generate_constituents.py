import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.utils.utils import *
from scripts.utils.CompositionalityParser import*

def generate_constituents():
    dataset_name = "3000validation"
    paraphrase_cache = read_json(data_base_path + f"cache_norm_bank/paraphrases/3000validation_paraphrases_filtered_by_nli.json")

    events = []
    for split in ["3000test"]:
        input_file = data_base_path + f"cache_norm_bank/events/clean_{split}.moral_acceptability.tsv"
        df_data = pd.read_csv(input_file, sep="\t")
        events += df_data["clean_event"].tolist()

    all_base_events = events[:]
    for event in events:
        all_base_events.extend(paraphrase_cache[event])
    all_base_events.extend(events)

    events = all_base_events[:]

    compositionality_parser = CompositionalityParser()

    data_to_save = {}
    for event in tqdm(events):
        parsed_event = compositionality_parser.get_parsed_event(event)
        data_to_save[event] = parsed_event

    constituent_data_path = data_base_path + f"cache_norm_bank/constituents/constituents.json"
    save_json(constituent_data_path, data_to_save)

    # data = data_to_save
    data = read_json(constituent_data_path)

    data_to_save = []
    for event in data:
        data_to_save.append(event)
        for sub_event in data[event]:
            if data[event][sub_event] != None:
                data_to_save.append(data[event][sub_event])

    # all_sequences_data_path = constituent_data_path.replace("constituents", "all_sequences")
    # print(all_sequences_data_path)
    print(len(data_to_save))
    save_json(data_base_path + f"cache_norm_bank/all_sequences/all_sequences.json", data_to_save)

# data = read_json(data_base_path + f"cache_norm_bank/all_sequences/all_sequences.json")
# data = list(set(data))
# save_json(data_base_path + f"cache_norm_bank/all_sequences/all_sequences.json", data)
# data = read_json(data_base_path + f"cache_norm_bank/all_sequences/all_sequences.json")

if __name__ == "__main__":
    generate_constituents()
