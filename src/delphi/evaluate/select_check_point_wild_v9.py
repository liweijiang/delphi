import sys
sys.path.append("script/evaluate")
from evaluate_utils import *


def eval_accept(row_accuracies, df_results):
    class_targets = df_results["moral_acceptability_class_targets"].tolist()
    class_preds = df_results["moral_acceptability_class_preds"].tolist()
    row_accuracies.append(get_accuracy(
        class_targets, class_preds, accuracy_type="exact"))
    row_accuracies.append(get_accuracy(
        class_targets, class_preds, accuracy_type="binary"))

    text_class_targets = df_results["moral_acceptability_text_2_class_targets"].tolist(
    )
    text_class_preds = df_results["moral_acceptability_text_2_class_preds"].tolist(
    )
    row_accuracies.append(get_accuracy(
        text_class_targets, text_class_preds, accuracy_type="binary"))

    text_targets = df_results["moral_acceptability_text_targets"].tolist()
    text_preds = df_results["moral_acceptability_text_preds"].tolist()
    exact_match_accuracy = get_moral_acceptability_text_exact_match_accuracy(
        text_targets, text_preds)
    return row_accuracies


def eval_agree(row_accuracies, df_results):
    class_targets = df_results["moral_agreement_class_targets"].tolist()
    class_preds = df_results["moral_agreement_class_preds"].tolist()
    row_accuracies.append(get_accuracy(
        class_targets, class_preds, accuracy_type="binary"))

    text_targets = df_results["moral_agreement_text_targets"].tolist()
    text_preds = df_results["moral_agreement_text_preds"].tolist()
    exact_match_accuracy, polarity_align_accuracy = get_moral_agreement_text_accuracy(
        text_targets, text_preds)
    row_accuracies.append(polarity_align_accuracy)
    return row_accuracies


def eval_compare(row_accuracies, df_results):
    class_targets = df_results["moral_comparison_class_targets"].tolist()
    class_preds = df_results["moral_comparison_class_preds"].tolist()
    row_accuracies.append(get_accuracy(
        class_targets, class_preds, accuracy_type="exact"))
    return row_accuracies


def eval_wild_general(row_accuracies, df_results):
    # classification accuracy
    class_targets = df_results["wild_train_100_class_targets"].tolist()
    class_preds = df_results["wild_train_100_class_preds"].tolist()
    row_accuracies.append(get_accuracy(
        class_targets, class_preds, accuracy_type="exact"))
    row_accuracies.append(get_accuracy(
        class_targets, class_preds, accuracy_type="binary"))

    # open-text accuracy
    text_class_targets = df_results["wild_v9_text_2_class_targets"].tolist()
    text_class_preds = df_results["wild_v9_text_2_class_preds"].tolist()

    row_accuracies.append(get_accuracy(
        text_class_targets, text_class_preds, accuracy_type="binary"))
    return row_accuracies


def eval_race_wild(row_accuracies, df_results):
    class_targets = df_results["race_test_class_targets"].tolist()
    class_preds = df_results["race_test_class_preds"].tolist()
    row_accuracies.append(get_accuracy(
        class_targets, class_preds, accuracy_type="exact"))
    row_accuracies.append(get_accuracy(
        class_targets, class_preds, accuracy_type="binary"))

    text_class_targets = df_results["wild_v9_text_2_class_targets"].tolist()
    text_class_preds = df_results["wild_v9_text_2_class_preds"].tolist()
    row_accuracies.append(get_accuracy(
        text_class_targets, text_class_preds, accuracy_type="binary"))
    return row_accuracies


def eval_gender_wild(row_accuracies, df_results):
    class_targets = df_results["gender_test_class_targets"].tolist()
    class_preds = df_results["gender_test_class_preds"].tolist()
    row_accuracies.append(get_accuracy(
        class_targets, class_preds, accuracy_type="exact"))
    row_accuracies.append(get_accuracy(
        class_targets, class_preds, accuracy_type="binary"))

    text_class_targets = df_results["wild_v9_text_2_class_targets"].tolist()
    text_class_preds = df_results["wild_v9_text_2_class_preds"].tolist()
    row_accuracies.append(get_accuracy(
        text_class_targets, text_class_preds, accuracy_type="binary"))
    # print("accept text binary:", get_accuracy(text_class_targets, text_class_preds, accuracy_type="binary"))
    return row_accuracies


def select_check_point_large_wild_ablation():
    sets = ["0", "10", "20", "40", "60", "80", "100", "woz_100"]
    set_to_checkpoints = {
        "0": [1643100],
        "10": [1641900],
        "20": [1702200],
        "40": [1561500],
        "60": [1662000],
        "80": [1702200],
        "100": [1662000],
        "woz_100": [1702200],
    }

    accuracies = []
    for s in sets:
        print("=" * 20, s, "=" * 20)

        if s == "0":
            model_type = f"sbic_commonsense_morality_joint_all_proportional"
        else:
            model_type = f"sbic_commonsense_morality_joint_all_proportional_wild_{s}"

        data_version = "v9"
        data_split = "validation"
        bucket_name = "ai2-tpu-europe-west4"
        lr = 0.0001
        bs = 8
        pt_model = "large"

        print("model_type:", model_type)
        print("lr:", lr)
        print("bs:", bs)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        check_point = set_to_checkpoints[s][0]
        row_accuracies = [check_point]

        ##################### accept #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_acceptability", lr, bs, pt_model)
        row_accuracies = eval_accept(row_accuracies, df_results)

        ##################### agree #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_agreement", lr, bs, pt_model)
        row_accuracies = eval_agree(row_accuracies, df_results)

        ##################### compare #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_comparison", lr, bs, pt_model)
        row_accuracies = eval_compare(row_accuracies, df_results)

        ##################### wild general #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "wild_train_100", lr, bs, pt_model)
        row_accuracies = eval_wild_general(row_accuracies, df_results)

        if data_split == "test":
            ##################### wild race #####################
            df_results = read_result_file(bucket_name, data_version, model_type,
                                          check_point, data_split, "race_test", lr, bs, pt_model)
            row_accuracies = eval_race_wild(row_accuracies, df_results)

            ##################### wild gender #####################
            df_results = read_result_file(bucket_name, data_version, model_type,
                                          check_point, data_split, "gender_test", lr, bs, pt_model)
            row_accuracies = eval_gender_wild(row_accuracies, df_results)

        accuracies.append(row_accuracies)
        print("-- check point:", check_point, row_accuracies)

    df_to_save = pd.DataFrame(accuracies)
    df_to_save.to_csv("temp_result_file_2.csv", index=False)


def select_check_point_large_compositionality_ablation():
    sets = ["0.1", "1", "10", "30", "60", "100", "base"]
    set_to_checkpoints = {
        "0.1": [1382600],
        "1": [1362500],
        "10": [1420800],
        "30": [1561500],
        "60": [1581600],
        "100": [1643100],
        "base": [1302200],
    }

    accuracies = []
    for s in sets:
        print("=" * 20, s, "=" * 20)

        if s == "100":
            model_type = f"sbic_commonsense_morality_joint_all_proportional"
        else:
            model_type = f"sbic_commonsense_morality_joint_all_proportional_new_{s}"

        data_version = "v9"
        data_split = "test"
        bucket_name = "ai2-tpu-europe-west4"
        lr = 0.0001
        bs = 8
        pt_model = "large"

        print("model_type:", model_type)
        print("lr:", lr)
        print("bs:", bs)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        check_point = set_to_checkpoints[s][0]
        row_accuracies = [check_point]

        ##################### accept #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_acceptability", lr, bs, pt_model)
        row_accuracies = eval_accept(row_accuracies, df_results)

        ##################### agree #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_agreement", lr, bs, pt_model)
        row_accuracies = eval_agree(row_accuracies, df_results)

        ##################### compare #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_comparison", lr, bs, pt_model)
        row_accuracies = eval_compare(row_accuracies, df_results)

        accuracies.append(row_accuracies)
        print("-- check point:", check_point, row_accuracies)

    df_to_save = pd.DataFrame(accuracies)
    df_to_save.to_csv("temp_result_file_2.csv", index=False)


def select_check_point(data_split, model_type, pt_model, bs, check_points):
    data_version = "v9"
    bucket_name = "ai2-tpu-europe-west4"
    lr = 0.0001

    print("model_type:", model_type)
    print("lr:", lr)
    print("bs:", bs)

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    result_prefix = f"projects/liweij/mosaic-commonsense-morality/results/{data_version}/" \
        f"{pt_model}/{model_type}/lr-{lr}_bs-{bs}/" \
        f"moral_acceptability/{data_split}/"

    if check_points == None:
        check_points = get_result_check_points(
            client, bucket_name, result_prefix, after_check_point=-1)
        # [1:]

    accuracies = []
    for check_point in check_points:
        row_accuracies = [check_point]

        ##################### accept #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_acceptability", lr, bs, pt_model)
        row_accuracies = eval_accept(row_accuracies, df_results)

        ##################### agree #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_agreement", lr, bs, pt_model)
        row_accuracies = eval_agree(row_accuracies, df_results)

        ##################### compare #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_comparison", lr, bs, pt_model)
        row_accuracies = eval_compare(row_accuracies, df_results)

        ##################### wild general #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "wild_train_100", lr, bs, pt_model)

        row_accuracies = eval_wild_general(row_accuracies, df_results)

        if data_split == "test":
            ##################### wild race #####################
            df_results = read_result_file(bucket_name, data_version, model_type,
                                          check_point, data_split, "race_test", lr, bs, pt_model)
            row_accuracies = eval_race_wild(row_accuracies, df_results)

            ##################### wild gender #####################
            df_results = read_result_file(bucket_name, data_version, model_type,
                                          check_point, data_split, "gender_test", lr, bs, pt_model)
            row_accuracies = eval_gender_wild(row_accuracies, df_results)

        accuracies.append(row_accuracies)
        print("-- check point:", check_point, row_accuracies)

        df_to_save = pd.DataFrame(accuracies)
        df_to_save.to_csv("temp_result_file_2.csv", index=False)


def select_check_point_certain(data_split, model_type, pt_model, bs, check_points):
    data_version = "v9"
    bucket_name = "ai2-tpu-europe-west4"
    lr = 0.0001

    print("model_type:", model_type)
    print("lr:", lr)
    print("bs:", bs)

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    result_prefix = f"projects/liweij/mosaic-commonsense-morality/results/{data_version}/" \
        f"{pt_model}/{model_type}/lr-{lr}_bs-{bs}/" \
        f"moral_acceptability/{data_split}/"

    if check_points == None:
        check_points = get_result_check_points(
            client, bucket_name, result_prefix, after_check_point=-1)
        # [1:]

    accuracies = []
    for check_point in check_points:
        row_accuracies = [check_point]

        ##################### accept #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_acceptability", lr, bs, pt_model)
        row_accuracies = eval_accept(row_accuracies, df_results)

        ##################### agree #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_agreement", lr, bs, pt_model)
        row_accuracies = eval_agree(row_accuracies, df_results)

        ##################### compare #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "moral_comparison", lr, bs, pt_model)
        row_accuracies = eval_compare(row_accuracies, df_results)

        ##################### wild general #####################
        df_results = read_result_file(bucket_name, data_version, model_type,
                                      check_point, data_split, "wild_train_100", lr, bs, pt_model)
        df_results = df_results[df_results["wild_train_100_class_targets"] < 90]
        row_accuracies = eval_wild_general(row_accuracies, df_results)

        if data_split == "test":
            ##################### wild race #####################
            df_results = read_result_file(bucket_name, data_version, model_type,
                                          check_point, data_split, "race_test", lr, bs, pt_model)
            df_results = df_results[df_results["race_test_class_targets"] < 90]
            row_accuracies = eval_race_wild(row_accuracies, df_results)

            ##################### wild gender #####################
            df_results = read_result_file(bucket_name, data_version, model_type,
                                          check_point, data_split, "gender_test", lr, bs, pt_model)
            df_results = df_results[df_results["gender_test_class_targets"] < 90]
            row_accuracies = eval_gender_wild(row_accuracies, df_results)

        accuracies.append(row_accuracies)
        print("-- check point:", check_point, row_accuracies)

        df_to_save = pd.DataFrame(accuracies)
        df_to_save.to_csv("temp_result_file_2.csv", index=False)


if __name__ == "__main__":
    check_points = None

    model_type = "sbic_commonsense_morality_joint_all_proportional"
    pt_model = "unicorn-pt"
    bs = 16
    check_points = [1264700, 1239200]
    select_check_point_certain("test", model_type, pt_model, bs, check_points)
