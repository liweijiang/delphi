import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
from scripts.utils.main_utils import *
from scripts.utils.utils import *
from scripts.utils.CacheHandler.DelphiCacheHandler import *


def compile_batch_resutls():
    cache_name = "cache_norm_bank"
    data_type = "3000test"

    all_gold_data_path = data_base_path + f"{cache_name}/events/test.moral_acceptability.tsv"
    df_all_gold_data = pd.read_csv(all_gold_data_path, sep="\t", low_memory=False)

    gold_data_path = data_base_path + f"{cache_name}/events/clean_{data_type}.moral_acceptability.tsv"
    df_gold_data = pd.read_csv(gold_data_path, sep="\t", low_memory=False)
    events = df_gold_data["event"].tolist()

    df_all_gold_data = df_all_gold_data[df_all_gold_data["input_sequence"].isin(events)]
    df_all_gold_data = df_all_gold_data.drop_duplicates(subset=["input_sequence"])
    df_all_gold_data = df_all_gold_data.merge(df_gold_data, left_on="input_sequence", right_on="event")
    df_all_gold_data_select = df_all_gold_data[["event", "clean_event", "class_label"]]
    df_all_gold_data_select["split"] = "test"
    df_all_gold_data_select["agreement_rate"] = 1.0

    samples_indices = df_all_gold_data_select.sample(frac=0.5, replace=False).index
    df_all_gold_data_select.loc[samples_indices, "split"] = "dev"

    # print(df_all_gold_data_select.value_counts("split"))

    df_all_gold_data_select.to_csv(data_base_path + f"{cache_name}/gold_data/{data_type}_gold_data.tsv", index=False, sep="\t")

    # print(df_all_gold_data_select["input_sequence"].value_counts())
    # print(df_all_gold_data_select["event"].value_counts())
    # print(df_all_gold_data_select["clean_event"].value_counts())


if __name__ == "__main__":
    compile_batch_resutls()
