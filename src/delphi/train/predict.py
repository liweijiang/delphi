#! /usr/bin/env python

"""Evaluate the model on the rainbow datasets."""

import t5
import os
import sys
import seqio
import logging
import click
import util
import pandas as pd
import tensorflow.compat.v1 as tf

# Improve logging.
from contextlib import contextmanager

# print("python", sys.version)
# print("t5", t5.__version__)
# print("tf", tf.__version__)
# print("seqio", seqio.__version__)

tf.disable_v2_behavior()

logger = logging.getLogger(__name__)

def getSubstringBetweenMarkers(source_string, start_marker, end_marker):
	start = source_string.find(start_marker) + len(start_marker)
	end = source_string.find(end_marker)
	return source_string[start: end]


@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)


@click.command()
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help=(
        "The batch size to use for prediction. For efficient prediction on the"
        " TPU, choose a multiple of either 8 or 128. Defaults to 64."
    ),
)
@click.option(
    "--model-parallelism",
    type=int,
    default=8,
    help="The degree of model parallelism to use. Defaults to 8.",
)
@click.option(
    "--tpu-name",
    type=str,
    default="de-tpu-5",
    required=True,
    help="The name of the TPU. Defaults to the TPU_NAME environment variable.",
)
@click.option(
    "--tpu-topology",
    type=str,
    default="v3-32",
    required=True,
    help=(
        "The topology of the TPU. Defaults to the TPU_TOPOLOGY environment"
        " variable."
    ),
)
def predict(
    batch_size: int,
    model_parallelism: int,
    tpu_name: str,
    tpu_topology: str,
) -> None:
    """Evaluate the model located at RESULTS_DIR on MIXTURE."""

    # eval_data = "race_topk_batch6to10"

    # eval_data = "acceptability_subset"
    # eval_data = "agreement_subset"

    # eval_data = "long_maarten_9"
    # eval_data = "divTopics.3ids.7"
    eval_data = "UNDHR.idty.0"
    # eval_data = "defeasible"
    # eval_data = "compositional_nature"
    # eval_data = "nature_paper"


    # data_version = "v9"
    # model_type = "sbic_commonsense_morality_joint_all_proportional"
    # # check_point = 1264700
    # check_point = 1239200
    # "ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/v9/unicorn-pt/sbic_commonsense_morality_joint_all_proportional/lr-0.0001_bs-16"

    data_version = "v11"
    model_type = "distribution"
    check_point = 1249400

    lr = 0.0001
    bs = 16
    bucket_name = "ai2-tpu-europe-west4"
    models_dir = f"gs://{bucket_name}/projects/liweij/mosaic-commonsense-morality/model/{data_version}/" \
                  f"unicorn-pt/{model_type}/lr-{lr}_bs-{bs}"
    training_type = "joint"
        # model_type.split("_")[-3]

    # Run evaluation.
    model = t5.models.MtfModel(
        model_dir=models_dir,
        tpu=tpu_name,
        tpu_topology=tpu_topology,
        model_parallelism=model_parallelism,
        batch_size=batch_size,
        sequence_length={"inputs": 512, "targets": 128},
        learning_rate_schedule=None,
        save_checkpoints_steps=5000,
        keep_checkpoint_max=None,
        iterations_per_loop=100,
    )

    predict_joint_inputs_paths = ["gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/" \
                                  f"data/qualitative_eval/{training_type}/" + eval_data + "_qualitative_eval.tsv"]
    predict_joint_outputs_paths = [
        models_dir.replace("model", "preds") + "/raw/" + eval_data + "_qualitative_eval.tsv"]

    for i in range(len(predict_joint_inputs_paths)):
        predict_joint_inputs_path = predict_joint_inputs_paths[i]
        predict_joint_outputs_path = predict_joint_outputs_paths[i]

        # Ignore any logging so that we only see the model's answers to the questions.
        with tf_verbosity_level('ERROR'):
            model.batch_size = 8  # Min size for small model on v2-8 with parallelism 1.
            model.predict(
                input_file=predict_joint_inputs_path,
                output_file=predict_joint_outputs_path,
                # Select the most probable output token at each step.
                temperature=0,
                checkpoint_steps=check_point,
            )


if __name__ == "__main__":
    predict()
