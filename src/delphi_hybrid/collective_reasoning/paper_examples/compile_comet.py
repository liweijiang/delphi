import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
from scripts.utils.main_utils import *
from scripts.utils.utils import *
from scripts.utils.CacheHandler.COMETCacheHandler import *

def compile_batch_results():
    data_type = "3000test"
    num_section = 10

    all_data = {}
    for section_id in range(num_section):
        base_data_path = f"data/norm_bank/comet/{data_type}/"
        data_path = base_data_path + f"comet_{section_id}.json"
        all_data.update(read_json(data_path))
        print(len(all_data))

    save_json(data_base_path + f"cache_norm_bank/comet_gens/3000test_comet.json", all_data)
    print("current:", len(all_data))

    events = read_json(data_base_path + f"cache_norm_bank/all_sequences/3000test_all_sequences.json")
    not_covered_events = [e for e in events if e not in all_data]
    print("not covered:", len(not_covered_events))

    cache_handler = COMETCacheHandler(filename="comet_gens/3000test_comet", cache_dir="cache_norm_bank")
    for event in tqdm(not_covered_events):
        comet_instance = cache_handler.save_instance(event)

    all_data = read_json(data_base_path + f"cache_norm_bank/comet_gens/3000test_comet.json")
    print("after generating extra:", len(all_data))

if __name__ == "__main__":
    compile_batch_results()