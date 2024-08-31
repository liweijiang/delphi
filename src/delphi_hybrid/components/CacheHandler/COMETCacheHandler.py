import os
import sys
import json
from tqdm import tqdm

sys.path.append(os.getcwd())
from scripts.utils.utils import *
from scripts.utils.COMETGenerator import *
from scripts.utils.CacheHandler.CacheHandler import *

class COMETCacheHandler(CacheHandler):
    def __init__(self, filename="comet_subset", cache_dir="cache", device_id=0):
        if filename != None and "comet" not in filename:
            print("ERROR: wrong cache file!")
        super().__init__("comet", cache_dir, filename)
        self.comet_generator = COMETGenerator(device_id=device_id)

    def _generate_instance(self, event):
        return self.comet_generator.generate_all_relations(event)


if __name__ == "__main__":
    # events = []
    # for split in ["test", "validation"]:
    #     input_file = data_base_path + f"cache_norm_bank/events/clean_{split}.moral_acceptability.tsv"
    #     df_data = pd.read_csv(input_file, sep="\t")
    #     events += df_data["clean_event"].tolist()

    events = list(read_json(data_base_path + f"cache_norm_bank/all_sequences.json").keys())
    print(events)

    cache_handler = COMETCacheHandler(filename="norm_bank_comet", cache_dir="cache_norm_bank")
    for event in tqdm(events):
        comet_instance = cache_handler.save_instance(event)
