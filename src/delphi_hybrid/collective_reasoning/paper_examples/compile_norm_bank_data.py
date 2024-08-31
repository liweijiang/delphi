import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
from scripts.utils.main_utils import *
from scripts.utils.utils import *


def compile_base_event():
    split = "test"
    data_path = f"data/norm_bank/{split}.moral_acceptability.tsv"
    df_data = pd.read_csv(data_path, delimiter="\t")

    df_data["rid"] = df_data.index
    df_data["event"] = df_data["input_sequence"]
    df_data["clean_event"] = df_data["input_sequence"].apply(normalize_event)

    df_data_to_save = df_data[["rid", "event", "clean_event"]]

    save_data_path = data_path.replace(split, "clean_" + split)
    df_data_to_save.to_csv(save_data_path, index=False, sep="\t")

    for split in ["test", "validation"]:
        data_path = f"data/norm_bank/{split}.moral_acceptability.tsv"
        df_data = pd.read_csv(data_path, delimiter="\t")

        df_data["rid"] = df_data.index
        df_data["event"] = df_data["input_sequence"]
        df_data["clean_event"] = df_data["input_sequence"].apply(normalize_event)

        print(df_data.head(3))

        df_data = df_data[df_data["input_type"].isin(["action", "action_situation", "action_situation_intention"])]

        df_data_to_save = df_data[["rid", "event", "clean_event"]]
        if split == "validation":
            df_data_to_save = df_data_to_save.sample(n=3000, random_state=1)

        save_data_path = data_path.replace(split, "clean_" + split)
        df_data_to_save.to_csv(save_data_path, index=False, sep="\t")

def compile_test_remained():
    # data_path = "/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache_norm_bank/events/clean_3000test.moral_acceptability.tsv"
    # df_sub_data = pd.read_csv(data_path, sep="\t")
    # data_path = "/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache_norm_bank/events/clean_test.moral_acceptability.tsv"
    # df_data = pd.read_csv(data_path, sep="\t")
    # df_data = df_data[~df_data["event"].isin(df_sub_data["event"])]
    # df_data.to_csv("/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache_norm_bank/events/clean_remainedtest.moral_acceptability.tsv", index=False, sep="\t")

    # data_path = "/net/nfs.cirrascale/mosaic/liweij/delphi_algo/data/cache_norm_bank/events/clean_remainedtest.moral_acceptability.tsv"
    # df_data = pd.read_csv(data_path, sep="\t")
    # print(df_data.shape)


if __name__ == "__main__":
    compile_test_remained()
