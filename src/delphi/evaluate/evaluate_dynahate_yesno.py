from sklearn.metrics import f1_score
import sys
import numpy as np
sys.path.append("script/evaluate")
from evaluate_utils import *


def get_gold_class(round_id, data_split, training_data_type):
    """
        Get gold inputs and targets class labels; format (<class>1</class> <text> </text>)
    """
    if data_split == "validation":
        data_split = "dev"
    data_base_path = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/dynahate/{round_id}_{training_data_type}/{data_split}.tsv"
    df_inputs = pd.read_csv(data_base_path, sep="\t")

    inputs_all = list(df_inputs["inputs"])
    inputs = [i.split("[moral_single]: ")[-1] for i in inputs_all]

    targets_all = list(df_inputs["targets"])
    targets = [int(i.split("[/class] [text]")[0].split("[class]")[-1])
               for i in targets_all]
    return inputs, targets


def get_pred_class(bucket, base_path, task_name, check_point, base_model):
    """
        Get preds class labels
    """
    preds_blob = bucket.get_blob(
        base_path + f"{task_name}_{check_point}_predictions")
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


def main_get_accuracy(base_path, data_split, training_data_type, round_ids=None, check_points=None, is_print=False):
    base_path += f"{data_split}_eval/"
    bucket_name = base_path.split("/")[0]
    result_prefix = "/".join(base_path.split("/")[1:])
    base_path = "/".join(base_path.split("/")[1:])
    base_model = base_path.split("/")[-5]
    training_data = base_path.split("/")[-4]

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    if check_points == None:
        check_points = get_check_points(
            client, bucket_name, result_prefix, after_check_point=-1)[1:]

    for check_point in check_points:  # check_points
        all_inputs = []
        all_targets = []
        all_preds = []
        all_f1s = []
        all_accs = []
        for round_id in round_ids:
            task_name = f"dynahate_round_{round_id}_{training_data_type}"

            inputs, targets = get_gold_class(
                round_id, data_split, training_data_type)
            preds = get_pred_class(
                bucket, base_path, task_name, check_point, base_model)
            targets, preds = remove_unknown_elements(targets, preds)
            f1 = f1_score(targets, preds, average='macro')
            acc = get_accuracy(targets, preds, accuracy_type="exact")

            all_inputs += inputs
            all_targets += targets
            all_preds += preds
            all_f1s.append(f1)
            all_accs.append(acc)
            if is_print:
                print(
                    f"round ({round_id}) {check_point}: f1 -- {f1} | accuracy -- {acc}")

        f1 = f1_score(all_targets, all_preds, average='macro')
        acc = get_accuracy(all_targets, all_preds, accuracy_type="exact")
        print(
            f"round {str(round_ids)} {check_point}: f1 -- {f1} | accuracy -- {acc}")


def main_r1_yesno(training_data, training_data_type):
    base_model_2_ckpt = {
        "dynahate_round_1_yesno": {"v11-delphi-declare": [1280000],
                                   "v10-delphi": [],
                                   "v9-delphi": [],
                                   "unicorn-pt": [],
                                   "11B": []},

        "dynahate_round_1_yesno_100_shot": {"v11-delphi-declare": [1290200],
                                            "v10-delphi": [],
                                            "v9-delphi": [],
                                            "unicorn-pt": [],
                                            "11B": []},

        "dynahate_round_1_yesno_class_only": {"v11-delphi-declare": [1290200],
                                              "v10-delphi": [],
                                              "v9-delphi": [],
                                              "unicorn-pt": [1132100],
                                              "11B": []},

        "dynahate_round_1_yesno_class_only_100_shot": {"v11-delphi-declare": [1234100],
                                                       "v10-delphi": [],
                                                       "v9-delphi": [],
                                                       "unicorn-pt": [1035200],
                                                       "11B": []},

        "dynahate_round_1_discriminate": {"v11-delphi-declare": [1331000],
                                          "v10-delphi": [],
                                          "v9-delphi": [],
                                          "unicorn-pt": [],
                                          "11B": []},

        "dynahate_round_1_discriminate_100_shot": {"v11-delphi-declare": [1269800],
                                                   "v10-delphi": [],
                                                   "v9-delphi": [],
                                                   "unicorn-pt": [1086200],
                                                   "11B": []},
    }

    print(training_data)
    # "v11-delphi-declare", "v10-delphi" "v10-delphi", "v9-delphi", "unicorn-pt", "11B"
    for base_model in ["v11-delphi-declare", "unicorn-pt"]:
        print("=" * 10, base_model)
        base_path = f"ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/finetune/{base_model}/{training_data}/lr-0.0002_bs-16/"
        check_points = None
        main_get_accuracy(base_path, "validation", training_data_type, round_ids=[
                          1], check_points=check_points, is_print=False)


def main_all_yesno(training_data, training_data_type):
    base_model_2_ckpt = {
        "dynahate_all_yesno": {"v11-delphi-declare": [],
                               "v10-delphi": [],
                               "v9-delphi": [],
                               "unicorn-pt": [],
                               "11B": []},

        "dynahate_all_yesno_100_shot": {"v11-delphi-declare": [1254500],
                                        "v10-delphi": [],
                                        "v9-delphi": [],
                                        "unicorn-pt": [],
                                        "11B": []},

        "dynahate_all_yesno_class_only": {"v11-delphi-declare": [],
                                          "v10-delphi": [],
                                          "v9-delphi": [],
                                          "unicorn-pt": [],
                                          "11B": []},

        "dynahate_all_yesno_class_only_100_shot": {"v11-delphi-declare": [1336100],
                                                   "v10-delphi": [],
                                                   "v9-delphi": [],
                                                   "unicorn-pt": [1101500],
                                                   "11B": []},

        "dynahate_all_discriminate": {"v11-delphi-declare": [1351400],
                                      "v10-delphi": [],
                                      "v9-delphi": [],
                                      "unicorn-pt": [],
                                      "11B": []},

        "dynahate_all_discriminate_100_shot": {"v11-delphi-declare": [1239200],
                                               "v10-delphi": [],
                                               "v9-delphi": [],
                                               "unicorn-pt": [1070900],
                                               "11B": []},
    }

    print(training_data)
    for base_model in ["v11-delphi-declare",
                       "unicorn-pt"]:
        print("=" * 10, base_model)
        base_path = f"ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/finetune/{base_model}/{training_data}/lr-0.0002_bs-16/"
        check_points = None
        main_get_accuracy(base_path, "validation", training_data_type, round_ids=[
                          1, 2, 3, 4], check_points=check_points, is_print=False)


if __name__ == "__main__":
    main_r1_yesno()
