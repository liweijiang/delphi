import sys
sys.path.append("script/evaluate")
from evaluate_utils import *


# ######################## moral acceptability/agreement class ########################
def get_gold_single_input_task_class(bucket, bucket_name, base_path, data_version, task_name, data_split):
    """
    Get gold inputs and targets class labels
    """
    data_base_path = f"gs://{bucket_name}/" + "/".join(base_path.split("/")[:3]) + "/data"
    df_inputs = pd.read_csv(data_base_path + f"/{data_version}_declare_only/{task_name}/{data_split}.tsv", sep="\t")

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



########################  moral acceptability/agreement text ########################
def get_gold_single_input_task_text(bucket, bucket_name, base_path, data_version, task_name, data_split):
    data_base_path = f"gs://{bucket_name}/" + "/".join(base_path.split("/")[:3]) + "/data"
    df_inputs = pd.read_csv(data_base_path + f"/{data_version}_declare_only/{task_name}/{data_split}.tsv", sep="\t")
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


########################  main ########################
def main_get_accuracy(base_path, data_split, check_points=None,
                      is_include_accept_class=True,
                      is_include_accept_text=True,
                      is_include_agree_class=True,
                      is_include_agree_text=True,
                      is_include_compare=True,):
    base_path += f"{data_split}_eval/"

    print("=" * 20)
    bucket_name = base_path.split("/")[0]
    result_prefix = "/".join(base_path.split("/")[1:])
    data_version = base_path.split("/")[5]
    base_path = "/".join(base_path.split("/")[1:])

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    if check_points == None:
        check_points = get_check_points(client, bucket_name, result_prefix, after_check_point=-1)[1:]
    # check_points.sort(reverse=True)

    for check_point in check_points:
        print("=" * 40, check_point, "=" * 40)

        main_moral_acceptability(client, bucket, bucket_name, base_path, check_point,
                                 data_version, data_split,
                                 get_gold_single_input_task_class,
                                 get_pred_single_input_task_class,
                                 get_gold_single_input_task_text,
                                 get_pred_single_input_task_text,
                                 is_include_accept_class,
                                 is_include_accept_text)

        main_moral_agreement(client, bucket, bucket_name, base_path, check_point,
                             data_version, data_split,
                             get_gold_single_input_task_class,
                             get_pred_single_input_task_class,
                             get_gold_single_input_task_text,
                             get_pred_single_input_task_text,
                             is_include_agree_class,
                             is_include_agree_text)


if __name__ == "__main__":
    base_path = "ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/v11/unicorn-pt/declare_only/lr-0.0001_bs-16/"
    # check_points = [1266200]
    check_points = None
    main_get_accuracy(base_path, "validation", check_points)
    # main_get_accuracy(base_path, "test", check_points)

    # base_path = "ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/v11/11B/declare_only/lr-0.0001_bs-16/"
    # # check_points = [1266200]
    # check_points = None
    # main_get_accuracy(base_path, "validation", check_points)
    # # main_get_accuracy(base_path, "test", check_points)
