import sys
import numpy as np
sys.path.append("script/evaluate")
from evaluate_utils import *
from sklearn.metrics import precision_recall_fscore_support


def get_gold_class(data_split):
    """
        Get gold inputs and targets class labels; format (<class>1</class> <text> </text>)
    """
    if data_split == "validation":
        data_split = "dev"

    data_base_path = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/latenthatred/latenthatred_st/{data_split}.tsv"
    df_inputs = pd.read_csv(data_base_path, sep="\t")
    df_inputs["targets"] = df_inputs["targets"].astype(str)

    inputs_all = list(df_inputs["inputs"])
    inputs = [i.split("[moral_single]: ")[-1] for i in inputs_all]

    targets_all = list(df_inputs["targets"])
    targets = [int(i.split("</class>")[0].split("<class>")[-1]) for i in targets_all]
    return inputs, targets


def get_pred_class(bucket, base_path, training_data, check_point, base_model):
    """
        Get preds class labels
    """
    preds_blob = bucket.get_blob(base_path + f"{training_data}_{check_point}_predictions")
    preds_blob_list = preds_blob.download_as_string().decode('utf-8').split("\n")[1:]

    preds_class = []
    for i in preds_blob_list:
        try:
            if "v10" in base_model or "v11" in base_model:
                preds_class.append(int(i.split("[/class]")[0].split("[class]")[-1]))
            else:
                preds_class.append(int(i.split(" ⁇ /class>")[0].split(" ⁇ class>")[-1]))
        except:
            print("output form not identifiable:", i)
            preds_class.append(99)
    return preds_class


def main_get_accuracy(base_path, data_split, check_points=None, is_save_results=True):
    base_model = base_path.split("/")[-4]
    training_data = base_path.split("/")[-3]
    if base_model == "v10-delphi":
        base_path = base_path.replace(training_data, "new_" + training_data)
        task_name = "new_latenthatred"
    else:
        task_name = "latenthatred"

    base_path += f"{data_split}_eval/"
    bucket_name = base_path.split("/")[0]
    result_prefix = "/".join(base_path.split("/")[1:])
    base_path = "/".join(base_path.split("/")[1:])
    # training_data = base_path.split("/")[-4]
    base_model = base_path.split("/")[-5]

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    if check_points == None:
        check_points = get_check_points(client, bucket_name, result_prefix, after_check_point=-1)[1:]
    # check_points.sort(reverse=True)

    for check_point in check_points:
        inputs, targets = get_gold_class(data_split)
        preds = get_pred_class(bucket, base_path, task_name, check_point, base_model)
        if base_model == "v10-delphi":
            preds_new = [t - 1 for t in preds]
            preds = preds_new

        # index_to_remove = []
        # for i, p in enumerate(preds):
        #     if p not in [-1, 1]:
        #         index_to_remove.append(i)
        #         print("remove:", i, p)
        # targets = np.delete(targets, index_to_remove).tolist()
        # preds = np.delete(preds, index_to_remove).tolist()
        #
        # scores = precision_recall_fscore_support(targets, preds, average='binary')
        # acc = get_accuracy(targets, preds, accuracy_type="exact")
        # print(f"{check_point}: accuracy -- {acc}")

        if is_save_results:
            df_data = pd.DataFrame()
            df_data["input"] = inputs
            df_data["target"] = targets
            df_data["pred"] = preds

            # print(f"100_shot-{training_data}-{base_model}-{data_split}.tsv")
            df_data.to_csv(f"zero_shot-{training_data}-{base_model}-{data_split}.tsv", index=False, sep="\t")


def main_all_0001(training_data):
    print("-" * 30, f"{training_data}", "-" * 30)
    base_model_2_ckpt = {"latenthatred": {
                              "v10-delphi": [],
                              "v9-delphi": [],
                              "unicorn-pt": [],
                              "11B": [],},

                         "latenthatred_100_shot": {
                              "v10-delphi": [],
                              "v9-delphi": [],
                              "unicorn-pt": [],
                              "11B": [], }
                         }

    for base_model in ["v9-delphi", "unicorn-pt"]: # "v10-delphi", "v9-delphi", "unicorn-pt", "11B"
        print("=" * 10, base_model)
        base_path = f"ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/finetune/{base_model}/{training_data}/lr-0.0001_bs-16/"
        check_points = None
        main_get_accuracy(base_path, "validation", check_points)
        # check_points = base_model_2_ckpt[training_data][base_model]
        # main_get_accuracy(base_path, "test", check_points)


def main_zero_shot(training_data):
    print("-" * 30, f"{training_data}", "-" * 30)
    base_model_2_ckpt = {"dynahate_round_1_st": {"v10-delphi": [],
                         "v9-delphi": [1341200],
                         "v9-delphi-new": [1264700],
                         "unicorn-pt": [1127000],
                         "11B": [1102000]},

                         "dynahate_all_st": {"v10-delphi": [],
                          "v9-delphi": [1402400],
                          "v9-delphi-new": [1387100],
                          "unicorn-pt": [1157600],
                          "11B": [1132600]},

                         "dynahate_round_1_st_100_shot": {"v10-delphi": [],
                          "v9-delphi": [1325900],
                          "unicorn-pt": [1106600],
                          "11B": [1076500]},

                         "dynahate_all_st_100_shot": {"v10-delphi": [],
                          "v9-delphi": [1392200],
                          "unicorn-pt": [1147400],
                          "11B": [1076500]}
                         }

    for base_model in ["v9-delphi-new", "unicorn-pt", "11B"]:  # "v10-delphi", "v9-delphi", "unicorn-pt", "11B"
        print("=" * 10, base_model)
        base_path = f"ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/finetune/{base_model}/{training_data}/lr-0.0002_bs-16/"
        # check_points = None
        # main_get_accuracy(base_path, "validation", check_points)
        check_points = base_model_2_ckpt[training_data][base_model]
        main_get_accuracy(base_path, "test", check_points)


def main_all(training_data):
    print("-" * 30, f"{training_data}", "-" * 30)
    base_model_2_ckpt = {"latenthatred": {"v11-delphi-declare": [1356500],
                                          "v10-delphi": [1318400],
                                          "v9-delphi": [1397300],
                                          "v9-delphi-new": [1300400],
                                          "unicorn-pt": [1157600],
                                          "11B": [1071400],},

                         "latenthatred_100_shot": {"v11-delphi-declare": [1274900],
                                                   "v10-delphi": [1328600],
                                                   "v9-delphi": [1285100],
                                                   "v9-delphi-new": [1259600],
                                                   "unicorn-pt": [1055600],
                                                   "11B": [1102000], }
                         }

    for base_model in ["v9-delphi-new", "unicorn-pt", "11B"]: # "v11-delphi-declare", "v10-delphi", "v9-delphi", "unicorn-pt", "11B" , "11B"
        print("=" * 10, base_model)
        base_path = f"ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/finetune/{base_model}/{training_data}/lr-0.0002_bs-16/"
        # check_points = None
        # main_get_accuracy(base_path, "validation", check_points)
        check_points = base_model_2_ckpt[training_data][base_model]
        main_get_accuracy(base_path, "test", check_points)




if __name__ == "__main__":
    # main_all("latenthatred")
    # main_all("latenthatred_100_shot")

    main_zero_shot("dynahate_round_1_st")
    # main_zero_shot("dynahate_all_st")
    # main_zero_shot("dynahate_round_1_st_100_shot")
    # main_zero_shot("dynahate_all_st_100_shot")





