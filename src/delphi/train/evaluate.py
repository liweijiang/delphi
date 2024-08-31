#! /usr/bin/env python

"""
Evaluate the model checkpoint
"""

import mixtures
import tasks
import t5
import os
import sys
import util
import seqio
import click
import logging
import tensorflow.compat.v1 as tf

print("python", sys.version)
print("t5", t5.__version__)
print("tf", tf.__version__)
print("seqio", seqio.__version__)

tf.disable_v2_behavior()

# N.B. We must import tasks and mixtures here so that they are registered and available for evaluation.

logger = logging.getLogger(__name__)


@click.command()
@click.argument("mixture", type=str)
@click.argument("results_dir", type=str)
# The name of the TPU. Defaults to the TPU_NAME environment variable.
@click.argument("tpu-name", type=str)
# The topology of the TPU. Defaults to the TPU_TOPOLOGY environment variable.
@click.argument("tpu-topology", type=str)
@click.argument("split", type=str)
@click.argument("checkpoint", type=int)
@click.option(
    "--model-parallelism",
    type=int,
    default=8,
    help="The degree of model parallelism to use. Defaults to 8.",
)
def evaluate(
    mixture: str,
    results_dir: str,
    split: str,
    checkpoint: int,
    model_parallelism: int,
    tpu_name: str,
    tpu_topology: str,
) -> None:
    """
    Evaluate the model located at RESULTS_DIR on MIXTURE.
    """

    print(tpu_name)
    print(tpu_topology)

    # Initialize arguments
    if tpu_topology == "v3-32":
        batch_size = 16
        model_parallelism = 8
    elif tpu_topology == "v3-8":
        batch_size = 8
        model_parallelism = 8
    else:
        print("ERROR: tpu_topology invalid")
        return

    # Validate arguments
    util.validate_path(results_dir)

    checkpoints = util.get_result_check_points(
        results_dir, split, "ethics_virtue")
    checkpoints = util.get_result_check_points(
        results_dir, split, "latenthatred")

    print("-" * 10, "checkpoints todo", "-" * 10)

    if checkpoint == 100:
        checkpoints_to_eval = None
    elif checkpoint == 0:
        checkpoints_to_eval = checkpoints
    else:
        checkpoints_to_eval = [checkpoint]

    print(checkpoints_to_eval)

    # Run evaluation
    model = t5.models.MtfModel(
        model_dir=results_dir,
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

    model.eval(
        mixture_or_task_name=mixture,
        checkpoint_steps=checkpoints_to_eval,
        split=split,
    )


if __name__ == "__main__":
    evaluate()
