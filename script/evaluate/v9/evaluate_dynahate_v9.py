import sys
import numpy as np
sys.path.append("script/evaluate")
from evaluate_utils import *
from sklearn.metrics import f1_score


def get_gold_class(round_id, data_split):
    """
        Get gold inputs and targets class labels; format (<class>1</class> <text> </text>)
    """
    if data_split == "validation":
        data_split = "dev"
    data_base_path = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/{round_id}/{data_split}.tsv"
    df_inputs = pd.read_csv(data_base_path, sep="\t")

    inputs_all = list(df_inputs["inputs"])
    inputs = [i.split("[moral_single]: ")[-1] for i in inputs_all]

    targets_all = list(df_inputs["targets"])
    targets = [int(i.split("</class> <text></text>")[0].split("<class>")[-1]) for i in targets_all]
    return inputs, targets


def get_pred_class(bucket, base_path, task_name, check_point, base_model):
    """
        Get preds class labels
    """
    preds_blob = bucket.get_blob(base_path + f"{task_name}_{check_point}_predictions")
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



def main_get_accuracy(base_path, data_split, training_data_type, round_ids=None, check_points=None,
                      is_print=False, all_list=None, is_save_results=True):
    base_path += f"{data_split}_eval/"
    bucket_name = base_path.split("/")[0]
    result_prefix = "/".join(base_path.split("/")[1:])
    base_path = "/".join(base_path.split("/")[1:])
    base_model = base_path.split("/")[-5]
    training_data = base_path.split("/")[-4]

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    if base_model == "v10-delphi":
        base_path = base_path.replace(training_data, "new_" + training_data)

    if check_points == None:
        check_points = get_check_points(client, bucket_name, result_prefix, after_check_point=-1)[1:]
    # check_points.sort(reverse=True)
    for check_point in check_points:
        all_inputs = []
        all_targets = []
        all_preds = []
        all_f1s = []
        all_accs = []
        all_round_ids = []
        for round_id in round_ids:
            if training_data_type == "":
                task_name = f"dynahate_round_{round_id}"
            else:
                task_name = f"dynahate_round_{round_id}_{training_data_type}"

            if base_model == "v10-delphi":
                task_name = "new_" + task_name

            inputs, targets = get_gold_class(round_id, data_split)
            preds = get_pred_class(bucket, base_path, task_name, check_point, base_model)
            if base_model == "v10-delphi":
                preds_new = [t - 1 for t in preds]
                preds = preds_new
            # targets, preds = remove_unknown_elements(targets, preds)

            # for i, p in enumerate(preds):
            #     t = targets[i]
            #     inp = inputs[i]
            #     if int(t) != int(p):
            #         # print('-' * 10)
            #         # print(inp)
            #         # print("preds:", p, "|| targets:", t)
            #
            #         if base_model == "unicorn-pt":
            #             if (inp, p, t) in all_list:
            #                 all_list.remove((inp, p, t))
            #         else:
            #             all_list.append((inp, p, t))

            all_round_ids += [round_id] * len(preds)
            all_inputs += inputs
            all_targets += targets
            all_preds += preds

            f1 = f1_score(targets, preds, average='macro')
            acc = get_accuracy(targets, preds, accuracy_type="exact")

            all_f1s.append(f1)
            all_accs.append(acc)
            if is_print:
                print(f"round ({round_id}) {check_point}: f1 -- {f1} | accuracy -- {acc}")

        f1 = f1_score(all_targets, all_preds, average='macro')
        acc = get_accuracy(all_targets, all_preds, accuracy_type="exact")
        print(f"round (all) {check_point}: f1 -- {f1} | accuracy -- {acc}")
        # print(check_point, all_f1s + all_accs)
        # print(check_point, all_f1s)

        if is_save_results:
            df_data = pd.DataFrame()
            df_data["round_id"] = all_round_ids
            df_data["input"] = all_inputs
            df_data["target"] = all_targets
            df_data["pred"] = all_preds

            # print(f"{training_data}-{base_model}.tsv")
            df_data.to_csv(f"{training_data}-{base_model}-{data_split}.tsv", index=False, sep="\t")


    return all_list


def main_r1_st(training_data):
    base_model_2_ckpt = {
        "dynahate_round_1_st": {"v11-delphi-declare": [1290200],
                                "v10-delphi": [1323500],
                                "v9-delphi": [1341200],
                                "v9-delphi-new": [1264700],
                                "unicorn-pt": [1127000],
                                "11B": [1102000]},

        "dynahate_round_1_st_100_shot": {"v11-delphi-declare": [1290200],
                                         "v10-delphi": [1349000],
                                         "v9-delphi": [1325900],
                                         "v9-delphi-new": [1285100],
                                         "unicorn-pt": [1106600],
                                         "11B": [1076500]},
    }

    print(training_data)
    for base_model in ["v9-delphi-new", "unicorn-pt", "11B"]: # "v11-delphi-declare", "v10-delphi" "v10-delphi", "v9-delphi", "unicorn-pt", "11B"
        print("=" * 10, base_model)
        base_path = f"ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/finetune/{base_model}/{training_data}/lr-0.0002_bs-16/"
        # check_points = None
        #
        check_points = base_model_2_ckpt[training_data][base_model]
        main_get_accuracy(base_path, "validation", "st", [1], check_points)
        main_get_accuracy(base_path, "test", "st", [1, 2, 3, 4], check_points, is_print=True)
#

def main_all_st(training_data):
    base_model_2_ckpt = {
        "dynahate_all_st": {"v11-delphi-declare": [1290200], # 1290200
                            "v10-delphi": [1292900],
                            "v9-delphi": [1402400],
                            "v9-delphi-new": [1387100],
                            "unicorn-pt": [1157600],
                            "11B": [1132600]},

        "dynahate_all_st_100_shot": {"v11-delphi-declare": [1331000],
                                     "v10-delphi": [1354100],
                                     "v9-delphi": [1392200],
                                     "v9-delphi-new": [1407500],
                                     "unicorn-pt": [1147400],
                                     "11B": [1076500]},
    }

    all_list = []

    print(training_data)
    for base_model in ["v9-delphi-new", "unicorn-pt", "11B"]: # "v11-delphi-declare", "v10-delphi", "v10-delphi", "v9-delphi", "unicorn-pt", "11B"
        print("=" * 10, base_model)
        base_path = f"ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/finetune/{base_model}/{training_data}/lr-0.0002_bs-16/"
        # check_points = None
        check_points = base_model_2_ckpt[training_data][base_model]
        main_get_accuracy(base_path, "validation", "st", [1, 2, 3, 4], check_points, is_print=False)
        all_list = main_get_accuracy(base_path, "test", "st", [1, 2, 3, 4], check_points, is_print=True, all_list=all_list)

    return all_list


if __name__ == "__main__":
    # main_r1_st("dynahate_round_1_st")
    main_r1_st("dynahate_round_1_st_100_shot")
    # all_list = main_all_st("dynahate_all_st")
    main_all_st("dynahate_all_st_100_shot")

    # for e in all_list:
    #     print(e[0], "|| pred:", e[1], "|| target:", e[2])

