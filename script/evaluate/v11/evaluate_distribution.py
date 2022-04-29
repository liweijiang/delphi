import sys
sys.path.append("script/evaluate")
from evaluate_utils import *


# ######################## moral acceptability/agreement class ########################
def get_gold_single_input_task_class(bucket, bucket_name, base_path, data_version, task_name, data_split):
    """
    Get gold inputs and targets class labels
    """
    data_base_path = f"gs://{bucket_name}/" + "/".join(base_path.split("/")[:3]) + "/data"
    df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/{task_name}/"
                                             f"{data_split}.{task_name}.tsv", sep="\t")

    inputs_all = list(df_inputs["inputs"])
    inputs = [i.split("[moral_single]: ")[-1] for i in inputs_all]

    targets_all = list(df_inputs["targets"])
    targets = [int(i.split("[/class] [text]")[0].split("[class]")[-1]) for i in targets_all]
    return inputs, targets


def get_pred_single_input_task_class(bucket, base_path, task_name, check_point):
    """
        Get preds class labels
    """
    preds_blob = bucket.get_blob(base_path + f"{task_name}_{check_point}_predictions")
    preds_blob_list = preds_blob.download_as_string().decode('utf-8').split("\n")[1:]

    preds_class = []
    for i in preds_blob_list:
        try:
            preds_class.append(int(i.split("[/class] [text]")[0].split("[class]")[-1]))
        except:
            print("output form not identifiable:", i)
            preds_class.append(1)
    return preds_class


def get_gold_single_input_task_class_wild_v11(bucket, bucket_name, base_path, data_version, task_name, data_split):
    data_base_path = f"gs://{bucket_name}/" + "/".join(base_path.split("/")[:3]) + "/data"

    if data_split == "validation":
        df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/wild/"
                                                 f"dev.tsv", sep="\t")
    # elif data_split == "test" and task_name == "general":
    elif data_split == "test":
        df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/wild/"
                                                 f"{task_name}.tsv", sep="\t")
    # elif data_split == "test" and task_name == "race":
    #     df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/wild/"
    #                                              f"race_test.tsv", sep="\t")
    # elif data_split == "test" and task_name == "gender":
    #     df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/wild/"
    #                                              f"gender_test.tsv", sep="\t")
    else:
        print("ERROR: not validation split")

    inputs_all = list(df_inputs["inputs"])
    targets_all = list(df_inputs["targets"])

    # inputs = [i.split("[moral_single]: ")[-1] for i in inputs_all]
    inputs = []
    for _, i in enumerate(inputs_all):
        if type(i) != type(""):
            print("gold class input error:", _, i, targets_all[_])
            inputs.append("")
        else:
            inputs.append(i.split("[moral_single]: ")[-1])


    # targets = [int(i.split("[/class] [text]")[0].split("[class]")[-1]) for i in targets_all]
    targets = []
    for _, i in enumerate(targets_all):
        if type(i) != type(""):
            print("gold class output error:", _, i, inputs_all[_])
            targets.append(0)
        else:
            targets.append(int(i.split("[/class] [text]")[0].split("[class]")[-1]))

    return inputs, targets


def get_pred_single_input_task_class_wild_v11(bucket, base_path, task_name, check_point):
    if task_name == "general_test":
        preds_blob = bucket.get_blob(base_path + f"wild_{check_point}_predictions")
    else:
        preds_blob = bucket.get_blob(base_path + f"{task_name}_{check_point}_predictions")
    preds_blob_list = preds_blob.download_as_string().decode('utf-8').split("\n")[1:]

    preds_class = []
    for i in preds_blob_list:
        try:
            preds_class.append(int(i.split("[/class] [text]")[0].split("[class]")[-1]))
        except:
            print("output form not identifiable:", i)
            preds_class.append(1)
    return preds_class


########################  moral acceptability/agreement text ########################
def get_gold_single_input_task_text(bucket, bucket_name, base_path, data_version, task_name, data_split):
    data_base_path = f"gs://{bucket_name}/" + "/".join(base_path.split("/")[:3]) + "/data"
    df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/{task_name}/"
                                             f"{data_split}.{task_name}.tsv", sep="\t")
    inputs_all = list(df_inputs["inputs"])
    inputs = [s.split("[moral_single]: ")[-1] for s in inputs_all]

    targets_all = list(df_inputs["targets"])
    targets = [i.split("[/class] [text]")[1].split("[/text]")[0] for i in targets_all]
    return inputs, targets


def get_pred_single_input_task_text(bucket, base_path, task_name, check_point):
    preds_blob = bucket.get_blob(base_path + f"{task_name}_{check_point}_predictions")
    preds_blob_list = preds_blob.download_as_string().decode('utf-8').split("\n")[1:]

    preds_text = []
    for i in preds_blob_list:
        try:
            preds_text.append(i.split("[/class] [text]")[1].split("[/text")[0])
        except:
            print("output form not identifiable:", i)
            preds_text.append("")
    return preds_text


def get_gold_single_input_task_text_wild_v11(bucket, bucket_name, base_path, data_version, task_name, data_split):
    data_base_path = f"gs://{bucket_name}/" + "/".join(base_path.split("/")[:3]) + "/data"
    if data_split == "validation":
        df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/wild/dev.tsv", sep="\t")
    # elif data_split == "test" and task_name == "wild":
    #     df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/wild/general_test.tsv", sep="\t")
    # elif data_split == "test" and task_name == "race_test":
    #     df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/wild/race_test.tsv", sep="\t")
    # elif data_split == "test" and task_name == "gender_test":
    #     df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/wild/gender_test.tsv", sep="\t")
    elif data_split == "test":
        df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/wild/{task_name}.tsv", sep="\t")
    else:
        print("ERROR: not validation split")

    inputs_all = list(df_inputs["inputs"])
    targets_all = list(df_inputs["targets"])

    inputs = []
    for _, i in enumerate(inputs_all):
        if type(i) != type(""):
            print("gold text input error:", _, i, targets_all[_])
            inputs.append("")
        else:
            inputs.append(i.split("[moral_single]: ")[-1])


    # targets = [int(i.split("[/class] [text]")[0].split("[class]")[-1]) for i in targets_all]
    targets = []
    for _, i in enumerate(targets_all):
        if type(i) != type(""):
            print("gold text output error:", _, i, inputs_all[_])
            targets.append(0)
        else:
            targets.append(i.split("[/class] [text]")[1].split("[/text]")[0])

    return inputs, targets


def get_pred_single_input_task_text_wild_v11(bucket, base_path, task_name, check_point):
    if task_name == "general_test":
        preds_blob = bucket.get_blob(base_path + f"wild_{check_point}_predictions")
    else:
        preds_blob = bucket.get_blob(base_path + f"{task_name}_{check_point}_predictions")

    preds_blob_list = preds_blob.download_as_string().decode('utf-8').split("\n")[1:]

    preds_text = []
    for i in preds_blob_list:
        try:
            preds_text.append(i.split("[/class] [text]")[1].split("[/text")[0])
        except:
            print("output form not identifiable:", i)
            preds_text.append("")
    return preds_text


########################  moral comparison class ########################
def get_gold_moral_comparison_class(bucket, bucket_name, base_path, data_version, data_split):
    data_base_path = f"gs://{bucket_name}/" + "/".join(base_path.split("/")[:3]) + "/data"
    df_inputs = pd.read_csv(data_base_path + f"/{data_version}_distribution/moral_comparison/"
                                             f"{data_split}.moral_comparison.tsv", sep="\t")
    inputs_all = list(df_inputs["inputs"])
    inputs = [s.split("[moral_pair]: ")[-1] for s in inputs_all]

    targets_all = list(df_inputs["targets"])
    targets = [int(t) for t in targets_all]

    return inputs, targets


def get_pred_moral_comparison_class(bucket, base_path, check_point):
    preds_blob = bucket.get_blob(base_path + f"moral_comparison_{check_point}_predictions")
    preds_blob_list = preds_blob.download_as_string().decode('utf-8').split("\n")[1:]
    return [int(i) for i in preds_blob_list]


########################  main ########################
def main_get_accuracy(base_path, data_split, check_points=None,
                      is_include_accept_class=True,
                      is_include_accept_text=True,
                      is_include_agree_class=True,
                      is_include_agree_text=True,
                      is_include_compare=True,):
    base_path += f"{data_split}_eval/"

    bucket_name = base_path.split("/")[0]
    result_prefix = "/".join(base_path.split("/")[1:])
    data_version = base_path.split("/")[5]
    model_type = base_path.split("/")[7]
    # eval_data = data_version + "_sbic_" + model_type.split("_")[-3]
    base_path = "/".join(base_path.split("/")[1:])

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    if check_points == None:
        check_points = get_check_points(client, bucket_name, result_prefix, after_check_point=-1)[2:]
    # check_points.sort(reverse=True)
    for check_point in check_points:
        print("=" * 40, check_point, "=" * 40)

        # main_moral_acceptability(client, bucket, bucket_name, base_path, check_point,
        #                          data_version, data_split,
        #                          get_gold_single_input_task_class,
        #                          get_pred_single_input_task_class,
        #                          get_gold_single_input_task_text,
        #                          get_pred_single_input_task_text,
        #                          is_include_accept_class,
        #                          is_include_accept_text,
        #                          task_name="moral_acceptability")
        #
        # main_moral_agreement(client, bucket, bucket_name, base_path, check_point,
        #                      data_version, data_split,
        #                      get_gold_single_input_task_class,
        #                      get_pred_single_input_task_class,
        #                      get_gold_single_input_task_text,
        #                      get_pred_single_input_task_text,
        #                      is_include_agree_class,
        #                      is_include_agree_text,
        #                      task_name="moral_agreement")
        #
        # if is_include_compare:
        #     main_moral_comparison(client, bucket, bucket_name, base_path, check_point,
        #                           data_version, data_split,
        #                           get_gold_moral_comparison_class,
        #                           get_pred_moral_comparison_class)

        main_wild_v11(client, bucket, bucket_name, base_path, check_point,
                     data_version, data_split,
                     get_gold_single_input_task_class_wild_v11,
                     get_pred_single_input_task_class_wild_v11,
                     get_gold_single_input_task_text_wild_v11,
                     get_pred_single_input_task_text_wild_v11, "general_test")

        if data_split == "test":
            main_wild_v11(client, bucket, bucket_name, base_path, check_point,
                         data_version, data_split,
                         get_gold_single_input_task_class_wild_v11,
                         get_pred_single_input_task_class_wild_v11,
                         get_gold_single_input_task_text_wild_v11,
                         get_pred_single_input_task_text_wild_v11, "race_test")

            main_wild_v11(client, bucket, bucket_name, base_path, check_point,
                         data_version, data_split,
                         get_gold_single_input_task_class_wild_v11,
                         get_pred_single_input_task_class_wild_v11,
                         get_gold_single_input_task_text_wild_v11,
                         get_pred_single_input_task_text_wild_v11, "gender_test")


if __name__ == "__main__":
    base_path = "ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/v11/unicorn-pt/distribution/lr-0.0001_bs-16/"
    check_points = None
    # main_get_accuracy(base_path, "validation", check_points)
    main_get_accuracy(base_path, "test", [1249400])

