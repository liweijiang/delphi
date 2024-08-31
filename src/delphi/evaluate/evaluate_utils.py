from text2class import *
from google.cloud import storage
import pandas as pd
import numpy as np
import sys
sys.path.append("script/evaluate")

pd.options.display.max_rows = 500
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 512


################# handle files #################
def get_check_points(client, bucket_name, result_prefix, after_check_point=-1):
    """
    Get prediction checkpoints
    """
    check_points = []
    for blob in client.list_blobs(bucket_name, prefix=result_prefix):
        blob_name = str(blob).split("_predictions")
        if len(blob_name) > 1:
            check_point = int(blob_name[-2].split("_")[-1])
            if check_point > after_check_point:
                check_points.append(check_point)
    check_points = list(set(check_points))
    check_points.sort()
    return check_points


def get_result_check_points(client, bucket_name, result_prefix, after_check_point=-1):
    check_points = []
    for blob in client.list_blobs(bucket_name, prefix=result_prefix):
        blob_name = str(blob).split("/")[-1]
        if "moral" in blob_name:
            check_point = int(blob_name.split(".csv, ")[-2].split("_")[-1])
            if check_point > after_check_point:
                check_points.append(check_point)
    return check_points


def read_result_file(bucket_name, data_version, model_type, check_point, data_split, task_name, lr, bs, pt_model):
    result_path = f"gs://{bucket_name}/projects/liweij/mosaic-commonsense-morality/results/{data_version}/" \
                  f"{pt_model}/{model_type}/lr-{lr}_bs-{bs}/" \
                  f"{task_name}/{data_split}/{task_name}_{check_point}.csv"
    return pd.read_csv(result_path, sep=",")


################# get full reference data #################
def get_gold_single_input_task_full(task_name, data_version, data_split):
    file_path = f"data/{data_version}_full/{data_version}_sbic_joint/{task_name}/{data_split}.{task_name}.tsv"
    df_data = pd.read_csv(file_path, sep="\t")

    if "pattern" in df_data.columns:
        df_data_gold_full = df_data[[
            "noisy_input_sequence", "class_label", "text_label", "source", "input_type", "pattern"]]
    else:
        df_data_gold_full = df_data[[
            "noisy_input_sequence", "class_label", "text_label", "source", "input_type"]]
    df_data_gold_full = df_data_gold_full.rename(
        columns={"noisy_input_sequence": "input_sequence"})

    return df_data_gold_full


def get_gold_moral_comparison_full(data_version, data_split, is_noisy=True):
    file_path = f"data/{data_version}_full/{data_version}_sbic_joint/moral_comparison/{data_split}.moral_comparison.tsv"
    df_data = pd.read_csv(file_path, sep="\t")
    if is_noisy:
        df_data["inputs"] = df_data["noisy_inputs"].str.replace(
            "[moral_pair]: ", "", regex=False)
    else:
        df_data["inputs"] = df_data["inputs"].str.replace(
            "[moral_pair]: ", "", regex=False)
    return df_data


def get_gold_single_input_task_full_wild_v9(task_name, data_version, data_split, is_noisy=True):
    file_path = f"data/{data_version}_full/{data_version}_sbic_joint/{task_name}/{data_split}.{task_name}.tsv"
    df_data = pd.read_csv(file_path, sep="\t")
    df_data_gold_full = df_data[["input", "class_label", "text_label"]]
    df_data_gold_full = df_data_gold_full.rename(
        columns={"input": "input_sequence"})
    return df_data_gold_full


################# main function per task #################
def main_moral_acceptability(client, bucket, bucket_name, base_path, check_point, data_version, data_split,
                             get_gold_single_input_task_class, get_pred_single_input_task_class,
                             get_gold_single_input_task_text, get_pred_single_input_task_text,
                             is_include_class, is_include_text):
    if is_include_class or is_include_text:
        task_name = "moral_acceptability"
        print("~" * 30, task_name)
        df_moral_acceptability_full = get_gold_single_input_task_full(
            task_name, data_version, data_split)

    if is_include_class:
        inputs, targets, preds = evaluate_single_input_task_class_task(bucket,
                                                                       bucket_name,
                                                                       base_path,
                                                                       check_point,
                                                                       data_version,
                                                                       task_name,
                                                                       data_split,
                                                                       get_gold_single_input_task_class,
                                                                       get_pred_single_input_task_class)

        df_moral_acceptability_full = merge_single_input_task_class_results(df_moral_acceptability_full, inputs,
                                                                            targets, preds, task_name)

    if is_include_text:
        inputs, targets, targets_class, preds, preds_class = \
            evaluate_moral_acceptability_text(bucket,
                                              bucket_name,
                                              base_path,
                                              check_point,
                                              data_version,
                                              data_split,
                                              get_gold_single_input_task_text,
                                              get_pred_single_input_task_text)

        df_moral_acceptability_full = merge_moral_acceptability_text_results(df_moral_acceptability_full, inputs,
                                                                             targets, preds, targets_class, preds_class)

    if is_include_class or is_include_text:
        save_result_file(df_moral_acceptability_full, client,
                         bucket, base_path, check_point, task_name, data_split)


def main_wild_v9(client, bucket, bucket_name, base_path, check_point, data_version, data_split,
                 get_gold_single_input_task_class, get_pred_single_input_task_class,
                 get_gold_single_input_task_text, get_pred_single_input_task_text, task_name):
    # if is_include_class or is_include_text:
    print("~" * 30, task_name)
    df_full = get_gold_single_input_task_full_wild_v9(
        task_name, data_version, data_split)

    inputs, targets, preds = evaluate_single_input_task_class_task(bucket,
                                                                   bucket_name,
                                                                   base_path,
                                                                   check_point,
                                                                   data_version,
                                                                   task_name,
                                                                   data_split,
                                                                   get_gold_single_input_task_class,
                                                                   get_pred_single_input_task_class)
    df_full = merge_single_input_task_class_results_wild_v9(
        df_full, inputs, targets, preds, task_name)

    inputs, targets, targets_class, preds, preds_class = \
        evaluate_wild_v9_text(bucket,
                              bucket_name,
                              base_path,
                              check_point,
                              data_version,
                              data_split,
                              get_gold_single_input_task_text,
                              get_pred_single_input_task_text, task_name)

    df_full = merge_wild_v9_text_results(
        df_full, inputs, targets, preds, targets_class, preds_class)
    save_result_file(df_full, client, bucket, base_path,
                     check_point, task_name, data_split)


def main_moral_agreement(client, bucket, bucket_name, base_path, check_point, data_version, data_split,
                         get_gold_single_input_task_class, get_pred_single_input_task_class,
                         get_gold_single_input_task_text, get_pred_single_input_task_text,
                         is_include_class, is_include_text):
    if is_include_class or is_include_text:
        task_name = "moral_agreement"
        print("~" * 30, task_name)
        df_moral_agreement_full = get_gold_single_input_task_full(
            task_name, data_version, data_split)

    if is_include_class:
        inputs, targets, preds = evaluate_single_input_task_class_task(bucket,
                                                                       bucket_name,
                                                                       base_path,
                                                                       check_point,
                                                                       data_version,
                                                                       task_name,
                                                                       data_split,
                                                                       get_gold_single_input_task_class,
                                                                       get_pred_single_input_task_class)
        df_moral_agreement_full = merge_single_input_task_class_results(df_moral_agreement_full, inputs, targets,
                                                                        preds, task_name)
    if is_include_text:
        inputs, targets, preds = evaluate_moral_agreement_text(bucket,
                                                               bucket_name,
                                                               base_path,
                                                               check_point,
                                                               data_version,
                                                               data_split,
                                                               get_gold_single_input_task_text,
                                                               get_pred_single_input_task_text)
        df_moral_agreement_full = merge_moral_agreement_text_results(
            df_moral_agreement_full, inputs, targets, preds)

    if is_include_class or is_include_text:
        save_result_file(df_moral_agreement_full, client, bucket,
                         base_path, check_point, task_name, data_split)


def main_moral_comparison(client, bucket, bucket_name, base_path, check_point, data_version, data_split,
                          get_gold_moral_comparison_class, get_pred_moral_comparison_class):
    print("~" * 30, "moral_comparison")
    df_moral_comparison_full = get_gold_moral_comparison_full(
        data_version, data_split)
    inputs, targets, preds = evaluate_moral_comparison_class(bucket,
                                                             bucket_name,
                                                             base_path,
                                                             check_point,
                                                             data_version,
                                                             data_split,
                                                             get_gold_moral_comparison_class,
                                                             get_pred_moral_comparison_class)
    df_moral_comparison_full = merge_moral_comparison_class_results(df_moral_comparison_full, inputs,
                                                                    targets, preds)
    save_result_file(df_moral_comparison_full, client, bucket,
                     base_path, check_point, "moral_comparison", data_split)


################# handle result files #################
def save_result_file(df_to_save, client, bucket, base_path, check_point, task_name, data_split):
    result_base_path = base_path.replace("model", "results")
    result_base_path = result_base_path.replace(
        f"{data_split}_eval", task_name) + f"{data_split}/"
    create_folder(client, bucket, result_base_path)
    df_to_save.to_csv("temp_result_file.csv", index=False)
    result_path = result_base_path + f"{task_name}_{check_point}.csv"
    target_blob = bucket.blob(result_path)
    target_blob.upload_from_filename("temp_result_file.csv")
    print("Result file saved:", result_path, "\nnum:", df_to_save.shape[0])


def create_folder(client, bucket, destination_folder_name):
    if not storage.Blob(bucket=bucket, name=destination_folder_name).exists(client):
        blob = bucket.blob(destination_folder_name)
        blob.upload_from_string('')


################# merge results #################
def merge_single_input_task_class_results(df_full, inputs, targets, preds, task_name):
    added_columns = [f"{task_name}_class_targets", f"{task_name}_class_preds"]

    df_results = pd.DataFrame(list(zip(*[inputs, targets, preds])),
                              columns=["inputs"] + added_columns)

    df_joined = pd.merge(left=df_full, right=df_results,
                         left_on='input_sequence', right_on='inputs', how='left')
    df_joined = df_joined.drop_duplicates(
        subset=['input_sequence'] + added_columns, keep="first")
    df_joined = df_joined[df_joined[f"{task_name}_class_targets"]
                          == df_joined["class_label"]]

    if "inputs_x" in df_joined.columns:
        df_joined = df_joined.drop(["inputs_x"], axis=1)
    if "inputs_y" in df_joined.columns:
        df_joined = df_joined.drop(["inputs_y"], axis=1)
    if "inputs" in df_joined.columns:
        df_joined = df_joined.drop(["inputs"], axis=1)
    return df_joined


def merge_single_input_task_class_results_wild_v9(df_full, inputs, targets, preds, task_name):
    added_columns = [f"{task_name}_class_targets", f"{task_name}_class_preds"]
    df_results = pd.DataFrame(list(zip(*[inputs, targets, preds])),
                              columns=["inputs"] + added_columns)
    df_joined = pd.merge(left=df_full, right=df_results,
                         left_on='input_sequence', right_on='inputs', how='left')
    df_joined[f"{task_name}_class_targets"] = df_joined["class_label"]
    df_joined = df_joined.drop_duplicates(
        subset=["input_sequence", "class_label"], keep="first")

    if "inputs_x" in df_joined.columns:
        df_joined = df_joined.drop(["inputs_x"], axis=1)
    if "inputs_y" in df_joined.columns:
        df_joined = df_joined.drop(["inputs_y"], axis=1)
    if "inputs" in df_joined.columns:
        df_joined = df_joined.drop(["inputs"], axis=1)
    return df_joined


def merge_moral_acceptability_text_results(df_full, inputs, targets, preds, targets_class, preds_class):
    added_columns = [f"moral_acceptability_text_targets",
                     f"moral_acceptability_text_preds",
                     f"moral_acceptability_text_2_class_targets",
                     f"moral_acceptability_text_2_class_preds"]
    df_results = pd.DataFrame(list(zip(*[inputs, targets, preds, targets_class, preds_class])),
                              columns=["inputs"] + added_columns)
    df_joined = pd.merge(left=df_full, right=df_results,
                         left_on='input_sequence', right_on='inputs', how='left')
    df_joined = df_joined[df_joined["moral_acceptability_text_targets"]
                          == df_joined["text_label"]]

    df_joined = df_joined.drop_duplicates(
        subset=['input_sequence'] + added_columns, keep="first")
    df_joined = df_joined.drop(["inputs", "class_label", "text_label"], axis=1)
    return df_joined


def merge_wild_v9_text_results(df_full, inputs, targets, preds, targets_class, preds_class):
    added_columns = [f"wild_v9_text_targets",
                     f"wild_v9_text_preds",
                     f"wild_v9_text_2_class_targets",
                     f"wild_v9_text_2_class_preds"]
    df_results = pd.DataFrame(list(zip(*[inputs, targets, preds, targets_class, preds_class])),
                              columns=["inputs"] + added_columns)
    df_joined = pd.merge(left=df_full, right=df_results,
                         left_on='input_sequence', right_on='inputs', how='left')
    df_joined = df_joined[df_joined["text_label"]
                          == df_joined["wild_v9_text_targets"]]

    df_joined = df_joined.drop_duplicates(
        subset=["input_sequence", "text_label"], keep="first")
    df_joined = df_joined.drop(["inputs", "class_label", "text_label"], axis=1)
    return df_joined


def merge_moral_agreement_text_results(df_full, inputs, targets, preds):
    added_columns = [f"moral_agreement_text_targets",
                     f"moral_agreement_text_preds"]
    df_results = pd.DataFrame(list(zip(*[inputs, targets, preds])),
                              columns=["inputs"] + added_columns)
    df_joined = pd.merge(left=df_full, right=df_results,
                         left_on='input_sequence', right_on='inputs', how='left')
    df_joined = df_joined[df_joined["moral_agreement_text_targets"]
                          == df_joined["text_label"]]

    df_joined = df_joined.drop_duplicates(
        subset=['input_sequence'] + added_columns, keep="first")
    df_joined = df_joined.drop(
        ["inputs", "source", "text_label", "class_label"], axis=1)
    return df_joined


def merge_moral_comparison_class_results(df_full, inputs, targets, preds):
    added_columns = [f"moral_comparison_class_targets",
                     f"moral_comparison_class_preds"]
    df_results = pd.DataFrame(list(zip(*[inputs, targets, preds])),
                              columns=["inputs"] + added_columns)
    df_joined = pd.merge(left=df_full, right=df_results,
                         left_on='inputs', right_on='inputs', how='left')
    df_joined = df_joined[df_joined["moral_comparison_class_targets"]
                          == df_joined["targets"]]

    df_joined = df_joined.drop_duplicates(
        subset=['inputs'] + added_columns, keep="first")
    df_joined = df_joined.drop(["targets", "inputs"], axis=1)
    return df_joined


################# get inputs and preds #################
def evaluate_single_input_task_class_task(bucket, bucket_name, base_path, check_point, data_version, task_name,
                                          data_split, get_gold, get_pred):
    inputs, targets = get_gold(
        bucket, bucket_name, base_path, data_version, task_name, data_split)
    preds = get_pred(bucket, base_path, task_name, check_point)
    if len(inputs) != len(preds):
        print(
            f"ERROR: inputs {len(inputs)} and preds {len(preds)} have different length")
    return inputs, targets, preds


def evaluate_moral_acceptability_text(bucket, bucket_name, base_path, check_point, data_version, data_split,
                                      get_gold, get_pred):
    inputs, targets = get_gold(
        bucket, bucket_name, base_path, data_version, "moral_acceptability", data_split)
    preds = get_pred(bucket, base_path, "moral_acceptability", check_point)
    targets_class, preds_class = convert_moral_acceptability_text_to_class(
        targets, preds)
    if len(inputs) != len(preds):
        print(
            f"ERROR: inputs {len(inputs)} and preds {len(preds)} have different length")
    return inputs, targets, targets_class, preds, preds_class


def evaluate_moral_agreement_text(bucket, bucket_name, base_path, check_point, data_version, data_split,
                                  get_gold, get_pred):
    inputs, targets = get_gold(
        bucket, bucket_name, base_path, data_version, "moral_agreement", data_split)
    preds = get_pred(bucket, base_path, "moral_agreement", check_point)
    if len(inputs) != len(preds):
        print(
            f"ERROR: inputs {len(inputs)} and preds {len(preds)} have different length")
    return inputs, targets, preds


def evaluate_wild_v9_text(bucket, bucket_name, base_path, check_point, data_version, data_split,
                          get_gold, get_pred, task_name):
    inputs, targets = get_gold(
        bucket, bucket_name, base_path, data_version, task_name, data_split)
    preds = get_pred(bucket, base_path, task_name, check_point)
    targets_class, preds_class = convert_moral_acceptability_text_to_class_wild_v9(
        targets, preds)
    if len(inputs) != len(preds):
        print(
            f"ERROR: inputs {len(inputs)} and preds {len(preds)} have different length")
    return inputs, targets, targets_class, preds, preds_class


def evaluate_moral_comparison_class(bucket, bucket_name, base_path, check_point, data_version, data_split,
                                    get_gold, get_pred):
    inputs, targets = get_gold(
        bucket, bucket_name, base_path, data_version, data_split)
    preds = get_pred(bucket, base_path, check_point)
    if len(inputs) != len(preds):
        print(
            f"ERROR: inputs {len(inputs)} and preds {len(preds)} have different length")
    return inputs, targets, preds


################# convert text to class #################
def convert_moral_acceptability_text_to_class_wild_v9(targets, preds):
    preds_classes = []
    target_classes = []
    covered_count = 0
    label_map_count = {}
    for i in range(len(targets)):
        target = normalize_label(targets[i])
        pred = normalize_label(preds[i])

        if ("no, " in target and "no, " in pred) or ("yes, " in target and "yes, " in pred):
            if "no, " in target:
                yesno_target = target.split(", ")[0]
                text_target = normalize_label(target.split("no, ")[1])
                yesno_pred = pred.split(", ")[0]
                text_pred = normalize_label(pred.split("no, ")[1])
            else:
                yesno_target = target.split(", ")[0]
                text_target = normalize_label(target.split("yes, ")[1])
                yesno_pred = pred.split(", ")[0]
                text_pred = normalize_label(pred.split("yes, ")[1])

            if target == pred:
                if yesno_target == "yes":
                    target_classes.append(1)
                    preds_classes.append(1)
                else:
                    target_classes.append(-1)
                    preds_classes.append(-1)
                covered_count += 1

            elif text_target in text2class and text_pred in text2class:
                text_target_class = text2class[text_target]
                text_pred_class = text2class[text_pred]

                if text_target_class == text_pred_class:
                    if yesno_target == "yes":
                        target_classes.append(1)
                        preds_classes.append(1)
                    else:
                        target_classes.append(-1)
                        preds_classes.append(-1)
                else:
                    target_classes.append(1)
                    preds_classes.append(-1)
                covered_count += 1

            else:
                if text_target not in text2class:
                    if text_target not in label_map_count:
                        label_map_count[text_target] = 1
                    else:
                        label_map_count[text_target] += 1

                if text_pred not in text2class:
                    if text_pred not in label_map_count:
                        label_map_count[text_pred] = 1
                    else:
                        label_map_count[text_pred] += 1

                target_classes.append(99)
                preds_classes.append(99)
            continue

        if "yes, " in target:
            text_target = normalize_label(target.split("yes, ")[1])
            if text_target in text2class and text2class[text_target] == 1:
                target = text_target

        elif "no, " in target:
            text_target = normalize_label(target.split("no, ")[1])
            if text_target in text2class and text2class[text_target] == -1:
                target = text_target

        elif "yes, " in pred:
            text_pred = normalize_label(pred.split("yes, ")[1])
            if text_pred in text2class and text2class[text_pred] == 1:
                pred = text_pred

        elif "no, " in pred:
            text_pred = normalize_label(pred.split("no, ")[1])
            if text_pred in text2class and text2class[text_pred] == -1:
                pred = text_pred

        if target not in text2class:
            if target not in label_map_count:
                label_map_count[target] = 1
            else:
                label_map_count[target] += 1

        if pred not in text2class:
            if pred not in label_map_count:
                label_map_count[pred] = 1
            else:
                label_map_count[pred] += 1

        if target in text2class and pred in text2class:
            target_class = text2class[target]
            pred_class = text2class[pred]

            target_classes.append(target_class)
            preds_classes.append(pred_class)
            covered_count += 1
        else:
            target_classes.append(99)
            preds_classes.append(99)

    label_map_count = sorted(label_map_count.items(),
                             key=lambda x: x[1], reverse=True)

    print(covered_count, len(preds_classes), "label coverage rate: %0.3f" %
          (covered_count / len(preds_classes)))

    return target_classes, preds_classes


def convert_moral_acceptability_text_to_class(targets, preds):
    preds_classes = []
    target_classes = []
    covered_count = 0
    label_map_count = {}
    for i in range(len(targets)):
        target = normalize_label(targets[i])
        pred = normalize_label(preds[i])

        if target not in text2class:
            if target not in label_map_count:
                label_map_count[target] = 1
            else:
                label_map_count[target] += 1

        if pred not in text2class:
            if pred not in label_map_count:
                label_map_count[pred] = 1
            else:
                label_map_count[pred] += 1

        if target in text2class and pred in text2class:
            target_class = text2class[target]
            pred_class = text2class[pred]

            target_classes.append(target_class)
            preds_classes.append(pred_class)
            covered_count += 1
        else:
            target_classes.append(99)
            preds_classes.append(99)

    label_map_count = sorted(label_map_count.items(),
                             key=lambda x: x[1], reverse=True)
    print("label coverage rate: %0.3f" %
          (1 - (preds_classes.count(99) / len(preds_classes))))

    return target_classes, preds_classes


################# get accuracy #################
def get_moral_acceptability_text_exact_match_accuracy(targets, preds):
    targets_clean = [normalize_label(t) for t in targets]
    preds_clean = [normalize_label(p) for p in preds]

    exact_match_accuracies = [int(targets_clean[i] == preds_clean[i])
                              for i in range(len(targets_clean))]
    exact_match_accuracy = float(
        sum(exact_match_accuracies) / len(exact_match_accuracies))
    return exact_match_accuracy


def get_moral_agreement_text_accuracy(targets, preds):
    targets_clean = [normalize_label(t) for t in targets]
    preds_clean = [normalize_label(p) for p in preds]

    exact_match_accuracies = []
    polarity_align_accuracies = []
    for i in range(len(preds_clean)):
        exact_match_accuracies.append(int(preds_clean[i] == targets_clean[i]))

        if (targets_clean[i] != preds_clean[i]):
            bare_min_target = targets_clean[i].replace("yes, ", "")
            bare_min_target = bare_min_target.replace("no, ", "")
            bare_min_target = normalize_label(bare_min_target)

            bare_min_pred = preds_clean[i].replace("yes, ", "")
            bare_min_pred = bare_min_pred.replace("no, ", "")
            bare_min_pred = normalize_label(bare_min_pred)

            if bare_min_pred in text2class and bare_min_target in text2class:
                bare_min_pred_binary = text2class[bare_min_pred]
                bare_min_target_binary = text2class[bare_min_target]

                if (("yes," in preds_clean[i] and "yes," in targets_clean[i])
                        or ("no," in preds_clean[i] and "no," in targets_clean[i])):
                    if (bare_min_pred_binary == bare_min_target_binary):
                        polarity_align_accuracies.append(1)
                    else:
                        polarity_align_accuracies.append(0)
                else:
                    polarity_align_accuracies.append(0)
        else:
            polarity_align_accuracies.append(1)

    exact_match_accuracy = float(
        sum(exact_match_accuracies) / len(exact_match_accuracies))
    polarity_align_accuracy = float(
        sum(polarity_align_accuracies) / len(polarity_align_accuracies))

    return exact_match_accuracy, polarity_align_accuracy


def get_accuracy(targets, preds, accuracy_type="exact"):
    accuracies = []
    for i in range(len(targets)):
        t_c = targets[i]
        p_c = preds[i]
        if t_c != 99 and p_c != 99:
            if accuracy_type == "exact":
                accuracies.append(int(t_c == p_c))
            elif accuracy_type == "non conflict":
                accuracies.append(int(t_c * p_c >= 0))
            else:
                accuracies.append(
                    not int((t_c == -1 or p_c == -1) and (t_c * p_c != 1)))

    return sum(accuracies) / len(accuracies)


def most_frequent(List):
    return max(set(List), key=List.count)


def get_majority_votes(d):
    annots = [d["annot1"], d["annot2"], d["annot3"]]
    maj_annot = most_frequent(annots)
    return maj_annot


def is_ambiguous(d):
    return (d["annot1"] != d["annot2"] or d["annot2"] != d["annot3"] or d["annot1"] != d["annot3"])


def remove_unknown_elements(targets, preds):
    index_to_remove = []
    for i, p in enumerate(preds):
        if p not in [-1, 1]:
            index_to_remove.append(i)
            print("remove:", i, p)
    targets = np.delete(targets, index_to_remove).tolist()
    preds = np.delete(preds, index_to_remove).tolist()
    return targets, preds
