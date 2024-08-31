#! /usr/bin/env python

"""Fine-tune T5 based models."""

import t5
import sys
import seqio
import click
import logging
import tensorflow.compat.v1 as tf

print("python", sys.version)
print("t5", t5.__version__)
print("tf", tf.__version__)
print("seqio", seqio.__version__)

import util
import warnings
import tasks, mixtures
# We must import tasks and mixtures here so that the tasks and mixtures are registered and available for training.

logger = logging.getLogger(__name__)

v=tf.compat.v1.logging.FATAL
tf.compat.v1.logging.set_verbosity(v)
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

warnings.filterwarnings("ignore", category=DeprecationWarning)

PRETRAINED_MODELS = {
    "small": ("gs://t5-data/pretrained_models/small/", -1),
    "base": ("gs://t5-data/pretrained_models/base/", -1),
    "large": ("gs://t5-data/pretrained_models/large/", -1),
    "3B": ("gs://t5-data/pretrained_models/3B/", -1),
    "11B": ("gs://t5-data/pretrained_models/11B/", -1),
    "unicorn-pt": ("gs://ai2-mosaic-public/projects/rainbow/v1.0/unicorns/lr-2e-3_batch-size-32/", -1),
    "v9-delphi": ("gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/v9/unicorn-pt/sbic_commonsense_morality_joint_all_proportional/lr-0.0001_bs-16/", 1264700),
    "v9-delphi-new": ("gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/v9/unicorn-pt/sbic_commonsense_morality_joint_all_proportional/lr-0.0001_bs-16/", 1239200),
    "v9-delphi-large": ("gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/model/v9/large/sbic_commonsense_morality_joint_all_proportional/lr-0.0001_bs-8/", 1643100),
}

@click.command()
@click.argument("mixture", type=str)
@click.argument("results_dir", type=str)
@click.argument("tpu-name", type=str) # The name of the TPU. Defaults to the TPU_NAME environment variable.
@click.argument("tpu-topology", type=str) # The topology of the TPU. Defaults to the TPU_TOPOLOGY environment variable.
@click.argument("pretrained-model", type=str)
@click.option(
    "--split",
    type=str,
    default="train",
    help="The split on which to train. Defaults to 'train'.",
)
@click.option(
    "--n-steps",
    type=int,
    default=600000,
    help="The number of gradient updates. Defaults to 25,000.",
)
@click.option(
    "--save-checkpoints-steps",
    type=int,
    default=5000,
    help=(
        "The number of steps to take before saving a checkpoint. Defaults to"
        " 5000."
    ),
)
@click.option(
    "--n-checkpoints-to-keep",
    type=int,
    default=300,
    help=(
        "The number of checkpoints to keep during fine-tuning. Defaults"
        " to 4."
    ),
)
@click.option(
    "--learning-rate",
    type=float,
    default=2e-4,
    help="The learning rate to use for training. Defaults to 3e-3.",
)
@click.option(
    "--continue_finetune",
    type=bool,
    default=False,
    help="Whether to continue training from an existing checkpoint.",
)

def fine_tune(
    mixture: str,
    results_dir: str,
    split: str,
    pretrained_model: str,
    n_steps: int,
    learning_rate: float,
    save_checkpoints_steps: int,
    n_checkpoints_to_keep: int,
    tpu_name: str,
    tpu_topology: str,
    continue_finetune: bool,
) -> None:
    """
    Fine-tune the model on MIXTURE, writing results to RESULTS_DIR.
    """

    # Initialize arguments
    if tpu_topology == "v3-32":
        batch_size = 16
        model_parallelism = 32
    elif tpu_topology == "v3-8":
        batch_size = 8
        model_parallelism = 8
    else:
        print("ERROR: tpu_topology invalid")
        return

    pretrained_checkpoint_step = -1

    # Get result path given arguments
    result_path = util.get_result_path(results_dir, pretrained_model, mixture, learning_rate, batch_size)

    # Validate path
    util.validate_path(results_dir, pretrained_model, PRETRAINED_MODELS)

    # Process arguments
    if pretrained_model in PRETRAINED_MODELS:
        pretrained_model, pretrained_checkpoint_step = PRETRAINED_MODELS[pretrained_model]

    # If the training stops before finishing and we want to continue from the last checkpoint
    if continue_finetune:
        pretrained_model = result_path

    # Print arguments
    util.print_arguments(result_path, results_dir, mixture, split, pretrained_model,
                    pretrained_checkpoint_step, n_steps, batch_size, model_parallelism,
                    save_checkpoints_steps, n_checkpoints_to_keep, learning_rate,
                    tpu_name, tpu_topology, tasks, continue_finetune)

    # Run fine-tuning
    model = t5.models.MtfModel(
        model_dir=result_path,
        tpu=tpu_name,
        tpu_topology=tpu_topology,
        model_parallelism=model_parallelism,
        batch_size=batch_size,
        sequence_length={"inputs": 512, "targets": 128},
        learning_rate_schedule=learning_rate,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=n_checkpoints_to_keep,
        iterations_per_loop=100,
    )

    model.finetune(
        mixture_or_task_name=mixture,
        pretrained_model_dir=pretrained_model,
        pretrained_checkpoint_step=pretrained_checkpoint_step,
        finetune_steps=n_steps,
        split=split,
    )


if __name__ == "__main__":
    fine_tune()
