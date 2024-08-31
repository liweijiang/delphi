import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
from scripts.utils.main_utils import *
from scripts.utils.utils import *
# from scripts.utils.CacheHandler.COMETCacheHandler import *


# itw_nli_path = "/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache/old/nov2022/nli.json"
# itw_nli_cache = read_json(itw_nli_path)
#
# itw_gold_data_path = "/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache/data_gold_labels.csv"
# df_itw_gold_data = pd.read_csv(itw_gold_data_path, sep=",")
# itw_events = df_itw_gold_data["event"].tolist()


def avg_nli_score():
    ###### Norm Bank ######
    norm_bank_nli_cache = read_json("/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache_norm_bank/nli/3000test_nli.json")
    df_norm_bank_gold_data = pd.read_csv("/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache_norm_bank/gold_data/3000test_gold_data.tsv", sep="\t")
    norm_bank_events = df_norm_bank_gold_data["clean_event"].tolist()
    norm_bank_paraphrases_cache = read_json(f"/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache_norm_bank/paraphrases/3000test_paraphrases_filtered_by_nli.json")

    norm_bank_entailment_scores = []
    for e in norm_bank_events:
        for p in norm_bank_paraphrases_cache[e]:
            e_p = e + " | " + p
            p_e = p + " | " + e
            e_p_nli = norm_bank_nli_cache[e_p]["entailment"]
            p_e_nli = norm_bank_nli_cache[p_e]["entailment"]
            norm_bank_entailment_scores.append(e_p_nli)
            norm_bank_entailment_scores.append(p_e_nli)

            if e_p_nli < 0.6 or p_e_nli < 0.6:
                print(e_p, e_p_nli)
                print(p_e, p_e_nli)

            # if (e_p_nli > 0.6 and e_p_nli < 0.8) or (p_e_nli > 0.6 and p_e_nli < 0.8):
            #     print(e_p, e_p_nli)
            #     print(p_e, p_e_nli)

            # if e_p_nli > 0.9 and p_e_nli > 0.9:
            #     print(e_p, e_p_nli)
            #     print(p_e, p_e_nli)


    print("norm_bank: ", sum(norm_bank_entailment_scores) / len(norm_bank_entailment_scores))

    ###### ITW ######
    itw_nli_cache = read_json("/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache/old/nov2022/nli.json")
    df_itw_gold_data = pd.read_csv("/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache/data_gold_labels.csv", sep=",")
    itw_events = df_itw_gold_data["event"].tolist()
    itw_paraphrases_cache = read_json(f"/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache/paraphrases_filtered_by_nli.json")

    itw_entailment_scores = []
    for e in itw_events:
        for p in itw_paraphrases_cache[e]:
            itw_entailment_scores.append(itw_nli_cache[e + " | " + p]["entailment"])
            itw_entailment_scores.append(itw_nli_cache[p + " | " + e]["entailment"])

    print("itw: ", sum(itw_entailment_scores) / len(itw_entailment_scores))





if __name__ == "__main__":
    avg_nli_score()
