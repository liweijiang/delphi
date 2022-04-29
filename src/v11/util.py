"""
Util functions for fine-tuning and evaluating models
"""
import seqio
import pandas as pd
from google.cloud import storage
import tensorflow_datasets as tfds
import tensorflow as tf


def create_folder(client, bucket, destination_folder_name):
    """
    Create a folder in Google Cloud Storage if such folder doesn't exist already
    """
    if not storage.Blob(bucket=bucket, name=destination_folder_name).exists(client):
        blob = bucket.blob(destination_folder_name)
        blob.upload_from_string('')
        print('Created: {}'.format(destination_folder_name))
    else:
        print('Exists: {}'.format(destination_folder_name))


def print_task_examples(task_name, split="validation", num_ex=1):
    """
    Print examples from tasks
    """
    print("#" * 20, task_name, "#" * 20)
    task = seqio.TaskRegistry.get(task_name)
    ds = task.get_dataset(split=split, sequence_length={"inputs": 512, "targets": 128})
    for i, ex in enumerate(tfds.as_numpy(ds.take(num_ex))):
        print(i, ex)
    print("test", task.num_input_examples("test"))
    print("train", task.num_input_examples("train"))
    print("validation", task.num_input_examples("validation"))


def print_mixture_examples(mixture_name, split="validation", num_ex=1):
    """
    Print examples from mixtures
    """
    print("#" * 20, mixture_name, "#" * 20)
    mixture = seqio.MixtureRegistry.get(mixture_name)
    ds = mixture.get_dataset(split=split,
                             sequence_length={"inputs": 512, "targets": 128})

    for i, ex in enumerate(tfds.as_numpy(ds.take(num_ex))):
        print(i, ex)
    print("test", mixture.num_input_examples("test"))
    print("train", mixture.num_input_examples("train"))
    print("validation", mixture.num_input_examples("validation"))


def get_num_elements_csv(file_name):
    """
    Get the total number of elements in a given csv/tsv file
    """
    df = pd.read_csv(file_name, delimiter="\t")
    return df.shape[0]


def get_num_elements_split(split_paths):
    """
    Get the number of elements in each split of a dataset
    """
    num_elements_split = {}
    for split, path in split_paths.items():
        num_elements_split[split] = get_num_elements_csv(path)
    return num_elements_split


def get_result_check_points(result_prefix, split, eval_data_type, after_check_point=-1):
    """
    Get a list of model checkpoints that haven't generated on the designated data split yet
    """
    client = storage.Client()
    bucket_name = "ai2-tpu-europe-west4"
    result_prefix = result_prefix.split(bucket_name + "/")[-1] + "/"

    check_points = []
    done_check_points = []
    for blob in client.list_blobs(bucket_name, prefix=result_prefix):
        blob_name = str(blob).split("/")[-1]
        if ".meta" in blob_name:
            check_point = int(blob_name.split(".meta")[0].split("-")[-1])
            if check_point > after_check_point:
                check_points.append(check_point)

    print("-" * 10, "checkpoints all", "-" * 10)
    print(check_points)

    for blob in client.list_blobs(bucket_name, prefix=result_prefix + f"{split}_eval/"):
        blob_name = str(blob).split("/")[-1]
        if "_predictions" in blob_name and eval_data_type in blob_name and "_predictions_clean" not in blob_name:
            check_point_done = int(blob_name.split("_predictions")[0].split("_")[-1])
            # check_point_done = int(blob_name.split("_")[0].split("_")[-1])
            if check_point_done in check_points:
                done_check_points.append(check_point_done)
                check_points.remove(check_point_done)

    print("-" * 10, "checkpoints done", "-" * 10)
    print(done_check_points)
    return check_points


def validate_path(results_dir, pretrained_model=None, PRETRAINED_MODELS=None):
    """
    Validate result path
    """
    if PRETRAINED_MODELS != None:
        if not results_dir.startswith("gs://"):
            raise ValueError(f"RESULTS_DIR ({results_dir}) must be a GCS path.")

        if pretrained_model.startswith("gs://"):
            if not tf.io.gfile.exists(pretrained_model):
                raise IOError(
                    f"--pretrained-model ({pretrained_model}) does not exist."
                )
        else:
            if pretrained_model not in PRETRAINED_MODELS:
                raise ValueError(
                    f"--pretrained-model ({pretrained_model}) not recognized. It"
                    f" must either be a GCS path or one of"
                    f' {", ".join(PRETRAINED_MODELS.keys())}.')
    else:
        if not results_dir.startswith("gs://"):
            raise ValueError(f"RESULTS_DIR ({results_dir}) must be a GCS path.")
        elif not tf.io.gfile.exists(results_dir):
            raise IOError(f"RESULTS_DIR ({results_dir}) doesn't exist.")


def print_arguments(result_path, results_dir, mixture, split, pretrained_model,
                    pretrained_checkpoint_step, n_steps, batch_size, model_parallelism,
                    save_checkpoints_steps, n_checkpoints_to_keep, learning_rate,
                    tpu_name, tpu_topology, tasks, continue_finetune):
    print("=" * 10, "results_dir")
    print(results_dir)

    print("=" * 10, "mixture")
    print(mixture)

    print("=" * 10, "split")
    print(split)

    print("=" * 10, "pretrained_model")
    print(pretrained_model)

    print("=" * 10, "pretrained_checkpoint_step")
    print(pretrained_checkpoint_step)

    print("=" * 10, "n_steps")
    print(n_steps)

    print("=" * 10, "batch_size")
    print(batch_size)

    print("=" * 10, "model_parallelism")
    print(model_parallelism)

    print("=" * 10, "save_checkpoints_steps")
    print(save_checkpoints_steps)

    print("=" * 10, "n_checkpoints_to_keep")
    print(n_checkpoints_to_keep)

    print("=" * 10, "learning_rate")
    print(learning_rate)

    print("=" * 10, "tpu_name")
    print(tpu_name)

    print("=" * 10, "tpu_topology")
    print(tpu_topology)

    print("=" * 10, "result_path")
    print(result_path)

    print("=" * 10, "data_version")
    print(tasks.data_version)

    print("=" * 10, "continue_finetune")
    print(continue_finetune)


def get_result_path(
        results_dir: str,
        pretrained_model: str,
        mixture: str,
        learning_rate: float,
        batch_size: int
) -> str:
    """
    Get a result path given arguments
    """
    result_path = results_dir + \
                  "/" + pretrained_model + \
                  "/" + mixture + \
                  f"/lr-{learning_rate}_bs-{batch_size}"
    return result_path

