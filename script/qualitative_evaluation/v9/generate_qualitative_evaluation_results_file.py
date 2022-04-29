import pandas as pd
from google.cloud import storage

pd.options.display.max_rows = 500
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 512


def isInt(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def save_result_file(df_to_save, bucket, result_path):
    df_to_save.to_csv("temp_result_file_3.csv", sep=",", index=False)
    target_blob = bucket.blob(result_path)
    target_blob.upload_from_filename("temp_result_file_3.csv")
    print("Result file saved:", result_path)


def generate_joint_pair_result_file(bucket, bucket_name, eval_data, preds_base_dir, check_point):
    inputs_filename = "gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/" \
                     f"data/qualitative_eval/joint/" + eval_data + "_qualitative_eval.tsv"
    preds_filename = preds_base_dir + "/raw/" + eval_data + f"_qualitative_eval.tsv-{check_point}"

    results_filename = preds_filename.replace("/raw", "").replace(f"gs://{bucket_name}/", "").replace(".tsv", ".csv")

    df_inputs = pd.read_csv(inputs_filename, "\t")
    df_preds = pd.read_csv(preds_filename, "\t")

    df_inputs["preds"] = df_preds.iloc[0:]
    df_inputs["action_1"] = df_inputs["inputs"].str.split("</action1> <action2>").str[0].str[len("[moral_pair]: <action1>"):]
    df_inputs["action_2"] = df_inputs["inputs"].str.split("</action1> <action2>").str[-1].str.split("</action2>").str[0]

    df_inputs = df_inputs.drop(columns=["inputs"])
    save_result_file(df_inputs, bucket, results_filename)


def generate_joint_result_file(bucket, bucket_name, eval_data, preds_base_dir, check_point):
    inputs_filename = "gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/" \
                     f"data/qualitative_eval/joint/" + eval_data + "_qualitative_eval.tsv"
    preds_filename = preds_base_dir + "/raw/" + eval_data + f"_qualitative_eval.tsv-{check_point}"

    results_filename = preds_filename.replace("/raw", "").replace(f"gs://{bucket_name}/", "").replace(".tsv", ".csv")

    df_inputs = pd.read_csv(inputs_filename, "\t")
    df_preds = pd.read_csv(preds_filename, "\t")

    df_inputs["text_label"] = df_preds.iloc[:, 0].str.split(" ⁇ /class>  ⁇ text>").str[-1].str.split(" ⁇ /text>").str[0]
    # df_inputs["class_label"] = df_preds.iloc[:, 0].str.split(" ⁇ /class>  ⁇ text>").str[0].str.split(" ⁇ class>").str[-1].astype(int)

    df_inputs["class_label_text"] = df_preds.iloc[:, 0].str.split(" ⁇ /class>  ⁇ text>").str[0].str.split(" ⁇ class>").str[-1]
    df_inputs["class_label_is_int"] = df_inputs["class_label_text"].apply(isInt)

    df_inputs['class_label'] = 0
    df_inputs['class_label'] = df_inputs["class_label_text"].where(df_inputs['class_label_is_int'], 0)
    df_inputs['class_label'] = df_inputs['class_label'].astype(int)

    df_inputs = df_inputs[["inputs", "text_label", "class_label"]]

    save_result_file(df_inputs, bucket, results_filename)


def generate_separate_result_file(bucket, bucket_name, eval_data, preds_base_dir, check_point):
    inputs_filename = "gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/" \
                     f"data/qualitative_eval/joint/" + eval_data + "_qualitative_eval.tsv"
    preds_class_filename = preds_base_dir + "/raw/" + eval_data + f"_class_qualitative_eval.tsv-{check_point}"
    preds_text_filename = preds_base_dir + "/raw/" + eval_data + f"_text_qualitative_eval.tsv-{check_point}"

    results_filename = preds_text_filename.replace("/raw", "").replace(f"gs://{bucket_name}/", "").replace(".tsv", ".csv").replace("_text", "")

    df_inputs = pd.read_csv(inputs_filename, "\t")
    df_preds_class = pd.read_csv(preds_class_filename, "\t")
    df_preds_text = pd.read_csv(preds_text_filename, "\t")

    df_inputs["text_label"] = df_preds_text.iloc[:, 0]
    df_inputs["class_label"] = df_preds_class.iloc[:, 0]

    save_result_file(df_inputs, bucket, results_filename)


def generate_result_file():
    # check_point_map = {"sbic_commonsense_morality_separate_all_proportional": 1045400,
    #                    "commonsense_morality_separate_all_proportional": 1040300,
    #                    "sbic_commonsense_morality_joint_all_proportional": 1081100,
    #                    "commonsense_morality_joint_all_proportional": 1065800,}

    # eval_data = "acceptability_subset"
    # eval_data = "agreement_subset"
    # eval_data = "comparison_subset"
    # eval_data = "moral_acceptability"
    # eval_data = "moral_agreement"
    # eval_data = "moral_comparison"

    # eval_data = "cm_test"
    # eval_data = "cm_test_hard"
    # eval_data = "justice_test"
    # eval_data = "justice_test_hard"
    # eval_data = "deontology_test"
    # eval_data = "deontology_test_hard"
    # eval_data = "util_test"
    # eval_data = "util_test_hard"
    # eval_data = "virtue_test"
    # eval_data = "virtue_test_hard"
    # eval_data = "jiminy_cricket"
    # eval_data = "mturk"
    # eval_data = "gender_topk_batch6to10"
    # eval_data = "race_topk_batch6to10"
    # eval_data = "nature_paper"
    eval_data = "UNDHR.idty.0"

    data_version = "v9"
    model_type = "sbic_commonsense_morality_joint_all_proportional"
    # check_point = check_point_map[model_type]
    # check_point = 1264700
    check_point = 1239200

    bucket_name = "ai2-tpu-europe-west4"
    preds_base_dir = f"gs://{bucket_name}/projects/liweij/mosaic-commonsense-morality/preds/{data_version}/" \
                  f"unicorn-pt/{model_type}/lr-0.0001_bs-16"
    training_type = model_type.split("_")[-3]

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    if training_type == "joint":
        if "pair" in eval_data or "comparison" in eval_data or "util_test" in eval_data:
            generate_joint_pair_result_file(bucket, bucket_name, eval_data, preds_base_dir, check_point)
        else:
            generate_joint_result_file(bucket, bucket_name, eval_data, preds_base_dir, check_point)
    else:
        generate_separate_result_file(bucket, bucket_name, eval_data, preds_base_dir, check_point)


if __name__ == "__main__":
    generate_result_file()  # pylint: disable=no-value-for-parameter
