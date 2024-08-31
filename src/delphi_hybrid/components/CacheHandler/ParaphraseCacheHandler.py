import os
import sys
import argparse

sys.path.append(os.getcwd())
from src.delphi_hybrid.components.Paraphraser import *
from src.delphi_hybrid.components.CacheHandler.CacheHandler import *

class ParaphraseCacheHandler(CacheHandler):
    def __init__(self, filename=None, cache_dir="cache", num_paraphrases=8):
        super().__init__("paraphrases", cache_dir, filename)
        self.paraphraser = Paraphraser()
        self.num_paraphrases = num_paraphrases

    def _clean_paraphrases(self, event, paraphrases):
        paraphrases = list(set(paraphrases))
        if event in paraphrases:
            paraphrases.remove(event)

        for paraphrase in paraphrases:
            is_qualified= self.paraphraser.qualify_paraphrase(event, paraphrase)
            if not is_qualified:
                paraphrases.remove(paraphrase)
        return paraphrases

    def save_instance(self, event):
        instance = []
        if event in self.cache:
            instance = self.cache[event]

        if len(instance) < self.num_paraphrases:
            instance += self.paraphraser.generate_paraphrases(event, num_paraphrases=self.num_paraphrases)["paraphrases"]
            instance = list(set(instance))
            self._add_instance(event, instance, is_save=True)
        else:
            print("[Note] Enough paraphrases in cache!")
        print("Num paraphrases:", len(instance))
        return instance

    def update_instance(self, event, is_save=True):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers with GPT-3.")
    parser.add_argument("--section_id", type=int, default="section_id")
    args = parser.parse_args()

    input_file = data_base_path + "cache_norm_bank/events/clean_test.moral_acceptability.tsv"
    df_data = pd.read_csv(input_file, sep="\t")
    events = df_data["clean_event"].tolist()

    section_id = 0

    cache_handler = ParaphraseCacheHandler(filename=f"paraphrases/norm_bank_paraphrases_{section_id}",
                                           cache_dir="cache_norm_bank",
                                           num_paraphrases=10)
    for e in tqdm(events):
        print(len(e))
        cache_handler.save_instance(e)
