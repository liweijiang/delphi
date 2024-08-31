
import os
import sys

sys.path.append(os.getcwd())
from src.delphi_hybrid.components.DelphiScorer import *
from src.delphi_hybrid.components.CacheHandler.CacheHandler import *

class DelphiCacheHandler(CacheHandler):
    def __init__(self, filename=None, cache_dir="cache", model="t5-11b-1239200", device_id=0, server="local"): #"beaker_batch"
        if filename != None and "delphi" not in filename:
            print("ERROR: wrong cache file!")
        super().__init__("delphi_scores", filename=filename, cache_dir=cache_dir)
        self.delphi_generator = DelphiScorer(model=model, device_id=device_id, server=server)

    def _generate_instance(self, event):
        class_label, probs, text_label = self.delphi_generator.generate_with_score(event)
        return {"class_label": class_label,
                "prob_1": probs[0],
                "prob_0": probs[1],
                "prob_minus_1": probs[2],
                "text_label": text_label}

if __name__ == "__main__":
    events = []
    for split in ["test", "validation"]:
        input_file = data_base_path + f"cache_norm_bank/events/clean_{split}.moral_acceptability.tsv"
        df_data = pd.read_csv(input_file, sep="\t")
        events += df_data["clean_event"].tolist()

    cache_handler = DelphiCacheHandler(filename="norm_bank_delphi", cache_dir="cache_norm_bank")
    for event in tqdm(events):
        comet_instance = cache_handler.save_instance(event)
