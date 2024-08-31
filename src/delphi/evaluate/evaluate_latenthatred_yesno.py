from sklearn.metrics import precision_recall_fscore_support
import sys
import numpy as np
sys.path.append("script/evaluate")
from evaluate_utils import *


def get_gold_class(data_split):
    """
        Get gold inputs and targets class labels; format (<class>1</class> <text> </text>)
    """
    if data_split == "validation":
        data_split = "dev"

    data_base_path = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/latenthatred/latenthatred_yesno/{data_split}.tsv"
    df_inputs = pd.read_csv(data_base_path, sep="\t")
    df_inputs["targets"] = df_inputs["targets"].astype(str)

    inputs_all = list(df_inputs["inputs"])
    inputs = [i.split("[moral_single]: ")[-1] for i in inputs_all]

    targets_all = list(df_inputs["targets"])
    targets = [int(i.split("[/class]")[0].split("[class]")[-1])
               for i in targets_all]
    return inputs, targets


def get_pred_class(bucket, base_path, training_data, check_point, base_model):
    """
        Get preds class labels
    """
    preds_blob = bucket.get_blob(
        base_path + f"{training_data}_{check_point}_predictions")
    preds_blob_list = preds_blob.download_as_string().decode(
        'utf-8').split("\n")[1:]

    preds_class = []
    for i in preds_blob_list:
        try:
            preds_class.append(
                int(i.split("[/class]")[0].split("[class]")[-1]))
        except:
            print("output form not identifiable:", i)
            preds_class.append(99)
    return preds_class


def main_get_accuracy(base_path, data_split, check_points=None):
    base_model = base_path.split("/")[-4]
    training_data = base_path.split("/")[-3]

    base_path += f"{data_split}_eval/"
    bucket_name = base_path.split("/")[0]
    result_prefix = "/".join(base_path.split("/")[1:])
    base_path = "/".join(base_path.split("/")[1:])
    training_data = base_path.split("/")[-4]
    base_model = base_path.split("/")[-5]

    if "100_shot" in training_data:
        training_data = training_data.replace("_100_shot", "")

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    if check_points == None:
        check_points = get_check_points(
            client, bucket_name, result_prefix, after_check_point=-1)[1:]

    for check_point in check_points:
        inputs, targets = get_gold_class(data_split)
        preds = get_pred_class(
            bucket, base_path, training_data, check_point, base_model)
        targets, preds = remove_unknown_elements(targets, preds)
        scores = precision_recall_fscore_support(
            targets, preds, average='binary')
        acc = get_accuracy(targets, preds, accuracy_type="exact")

        print(check_point, scores, acc)


def main_all(training_data):
    print("-" * 30, f"{training_data}", "-" * 30)
    base_model_2_ckpt = {"latenthatred": {"v11-delphi-declare": [],
                                          "v10-delphi": [],
                                          "v9-delphi": [],
                                          "unicorn-pt": [],
                                          "11B": [], },

                         "latenthatred_100_shot": {"v11-delphi-declare": [],
                                                   "v10-delphi": [],
                                                   "v9-delphi": [],
                                                   "unicorn-pt": [],
                                                   "11B": [], }
                         }

    for base_model in ["v11-delphi-declare", "unicorn-pt"]:
        print("=" * 10, base_model)
        base_path = f"ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/finetune/{base_model}/{training_data}/lr-0.0002_bs-16/"
        check_points = None
        main_get_accuracy(base_path, "validation", check_points)


if __name__ == "__main__":
    main_all("latenthatred_yesno")
    main_all("latenthatred_yesno_class_only")
