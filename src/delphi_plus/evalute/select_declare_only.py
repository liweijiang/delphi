import sys
sys.path.append("script/evaluate")
from evaluate_utils import *

def eval_accept(row_accuracies, df_results):
    # df_results["freeform_class_targets"] = df_results["freeform_class_targets"]
    # df_results["moral_acceptability_class_preds"] = df_results["moral_acceptability_class_preds"]
    class_targets = df_results["freeform_class_targets"].tolist()
    class_preds = df_results["freeform_class_preds"].tolist()
    row_accuracies.append(get_accuracy(class_targets, class_preds, accuracy_type="exact"))
    row_accuracies.append(get_accuracy(class_targets, class_preds, accuracy_type="binary"))
    # print("accept class exact:", get_accuracy(class_targets, class_preds, accuracy_type="exact"))
    # print("accept class binary:", get_accuracy(class_targets, class_preds, accuracy_type="binary"))

    text_class_targets = df_results["moral_acceptability_text_2_class_targets"].tolist()
    text_class_preds = df_results["moral_acceptability_text_2_class_preds"].tolist()
    row_accuracies.append(get_accuracy(text_class_targets, text_class_preds, accuracy_type="binary"))
    # print("accept text binary:", get_accuracy(text_class_targets, text_class_preds, accuracy_type="binary"))

    text_targets = df_results["moral_acceptability_text_targets"].tolist()
    text_preds = df_results["moral_acceptability_text_preds"].tolist()
    exact_match_accuracy = get_moral_acceptability_text_exact_match_accuracy(text_targets, text_preds)
    # print("accept text exact:", exact_match_accuracy)
    return row_accuracies


def eval_agree(row_accuracies, df_results):
    # df_results["moral_agreement_class_targets"] = df_results["moral_agreement_class_targets"]
    # df_results["moral_agreement_class_preds"] = df_results["moral_agreement_class_preds"]
    class_targets = df_results["yesno_class_targets"].tolist()
    class_preds = df_results["yesno_class_preds"].tolist()
    row_accuracies.append(get_accuracy(class_targets, class_preds, accuracy_type="binary"))
    # print("agree class exact:", get_accuracy(class_targets, class_preds, accuracy_type="exact"))

    text_targets = df_results["moral_agreement_text_targets"].tolist()
    text_preds = df_results["moral_agreement_text_preds"].tolist()
    exact_match_accuracy, polarity_align_accuracy = get_moral_agreement_text_accuracy(text_targets, text_preds)
    row_accuracies.append(polarity_align_accuracy)
    # print("agree text binary:", polarity_align_accuracy)
    return row_accuracies



def select_check_point(data_split, model_type, pt_model, bs, check_points):
    data_version = "v11"
    bucket_name = "ai2-tpu-europe-west4"
    lr = 0.0001

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    print("model_type:", model_type)
    print("lr:", lr)
    print("bs:", bs)

    result_prefix = f"projects/liweij/mosaic-commonsense-morality/results/{data_version}/" \
                      f"{pt_model}/{model_type}/lr-{lr}_bs-{bs}/" \
                      f"freeform/{data_split}/"

    if check_points == None:
        check_points = get_result_check_points(client, bucket_name, result_prefix, after_check_point=-1)[2:]

    accuracies = []
    for check_point in check_points:
        row_accuracies = [check_point]

        ##################### accept #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "freeform", lr, bs, pt_model)
        row_accuracies = eval_accept(row_accuracies, df_results)

        ##################### agree #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "yesno", lr, bs, pt_model)
        row_accuracies = eval_agree(row_accuracies, df_results)

        accuracies.append(row_accuracies)
        print("-- check point:", check_point, row_accuracies)

        df_to_save = pd.DataFrame(accuracies)
        df_to_save.to_csv("temp_result_file_2.csv", index=False)


if __name__ == "__main__":
    # select_check_point()
    # select_check_point_large_wild_ablation()

    model_type = "declare_only"
    pt_model = "unicorn-pt"
    bs = 16
    # check_points = [1266200]
    check_points = None
    select_check_point("validation", model_type, pt_model, bs, check_points)
    # select_check_point_certain("test", model_type, pt_model, bs, check_points)

    # model_type = "declare_only"
    # pt_model = "11B"
    # bs = 16
    # # check_points = [1266200]
    # check_points = None
    # select_check_point("validation", model_type, pt_model, bs, check_points)
    # # select_check_point_certain("test", model_type, pt_model, bs, check_points)

