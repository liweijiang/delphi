#! /usr/bin/env python

"""
Evaluate the model checkpoint
"""

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

import tasks, mixtures
# N.B. We must import tasks and mixtures here so that they are registered and available for evaluation.

logger = logging.getLogger(__name__)

@click.command()
@click.argument("mixture", type=str)
@click.argument("results_dir", type=str)
@click.argument("tpu-name", type=str) # The name of the TPU. Defaults to the TPU_NAME environment variable.
@click.argument("tpu-topology", type=str) # The topology of the TPU. Defaults to the TPU_TOPOLOGY environment variable.
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

    # wild_train_100 dynahate_round_1 race_test
    # Get checkpoints to be evaluated
    # checkpoints = util.get_result_check_points(results_dir, split, "ethics_deontology")
    # checkpoints = util.get_result_check_points(results_dir, split, "ethics_justice")
    # checkpoints = util.get_result_check_points(results_dir, split, "ethics_util")
    checkpoints = util.get_result_check_points(results_dir, split, "ethics_virtue")
    # checkpoints = util.get_result_check_points(results_dir, split, "ethics_cm")

    checkpoints = util.get_result_check_points(results_dir, split, "latenthatred")






    print("-" * 10, "checkpoints todo", "-" * 10)
    # checkpoints.sort(reverse=True)

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
        # checkpoint_steps=checkpoints,
        # checkpoint_steps=None,

        # checkpoint_steps=[1320800],  # R1 v9-delphi full
        # checkpoint_steps=[1086200],  # R1 unicorn-pt full
        # checkpoint_steps=[1066300],  # R1 11B full

        # checkpoint_steps=[1295300],  # R1 v9-delphi 100-shot
        # checkpoint_steps=[1142300],  # R1 unicorn-pt 100-shot
        # checkpoint_steps=[1076500],  # R1 11B 100-shot

        # checkpoint_steps=[1438100],  # R1234 v9-delphi full
        # checkpoint_steps=[1213700],  # R1234 unicorn-pt full
        # checkpoint_steps=[1234600],  # R1234 11B full

        # checkpoint_steps=[1341200],  # R1234 v9-delphi 100-shot
        # checkpoint_steps=[1086700],  # R1234 11B 100-shot
        # checkpoint_steps=[1111700],  # R1234 unicorn-pt 100-shot

        # checkpoint_steps=[1417700],  # (nat) R1 v9-delphi full
        # checkpoint_steps=[1188200],  # (nat) R1 unicorn-pt full
        # checkpoint_steps=[1147900],  # (nat) R1 11B full

        # checkpoint_steps=[1397300],  # (nat) R1 v9-delphi 100-shot
        # checkpoint_steps=[1076000],  # (nat) R1 unicorn-pt 100-shot
        # checkpoint_steps=[1107100],  # (nat) R1 11B 100-shot

        # checkpoint_steps=[1331000],  # (bc) R1 v9-delphi full
        # checkpoint_steps=[1116800],  # (bc) R1 unicorn-pt full
        # checkpoint_steps=[1051000],  # (bc) R1 11B full

        # checkpoint_steps=[1351400],  # (st_clean) R1 v9-delphi full
        # checkpoint_steps=[1106600],  # (st_clean) R1 unicorn-pt full
        # checkpoint_steps=[1102000],  # (st_clean) R1 11B full


        # checkpoint_steps=[1341200],  # (st) R1 v9-delphi full
        # checkpoint_steps=[1127000],  # (st) R1 unicorn-pt full
        # checkpoint_steps=[1102000],  # (st) R1 11B full

        # checkpoint_steps=[1402400],  # (st) R1234 v9-delphi full
        # checkpoint_steps=[1157600],  # (st) R1234 unicorn-pt full
        # checkpoint_steps=[1132600],  # (st) R1234 11B full

        # checkpoint_steps=[1325900],  # (st) R1 v9-delphi 100-shot
        # checkpoint_steps=[1106600],  # (st) R1 unicorn-pt 100-shot
        # checkpoint_steps=[1076500],  # (st) R1 11B 100-shot

        # checkpoint_steps=[1392200],  # (st) R1234 v9-delphi 100-shot
        # checkpoint_steps=[1147400],  # (st) R1234 unicorn-pt 100-shot
        # checkpoint_steps=[1076500],  # (st) R1234 11B 100-shot


        # checkpoint_steps=[1397300],  # latenthatred v9-delphi full
        # checkpoint_steps=[1157600],  # latenthatred unicorn-pt full

        # checkpoint_steps=[],  # ethics cm v9-delphi full
        # checkpoint_steps=[],  # ethics cm unicorn-pt full

        # checkpoint_steps=[],  # ethics deontology v9-delphi full
        # checkpoint_steps=[],  # ethics deontology unicorn-pt full

        # checkpoint_steps=[1356500],  # ethics justice v9-delphi full
        # checkpoint_steps=[1137200],  # ethics justice unicorn-pt full

        # checkpoint_steps=[1356500],  # ethics virtue v9-delphi full
        # checkpoint_steps=[1050500],  # ethics virtue unicorn-pt full

        # checkpoint_steps=[],  # ethics util v9-delphi full
        # checkpoint_steps=[],  # ethics util unicorn-pt full

        # checkpoint_steps=[1224400], # 11B all norm bank

        # checkpoint_steps=[1081600], # 11b 0.1%
        # checkpoint_steps=[1040800], # 11b 0.01%

        # checkpoint_steps = [1643100], # wild 0
        # checkpoint_steps = [1641900], # wild 10
        # checkpoint_steps=[1702200],  # wild 20
        # checkpoint_steps = [1561500], # wild 40
        # checkpoint_steps = [1662000], # wild 60
        # checkpoint_steps = [1702200],  # wild 80
        # checkpoint_steps = [1662000],  # wild 100
        # checkpoint_steps = [1702200], # wild woz 100

        # checkpoint_steps = [1264700], # wild finetune 11b 0
        # checkpoint_steps = [1284800], # wild finetune 11b 10
        # checkpoint_steps = [1284800], # wild finetune 11b 20
        # checkpoint_steps = [1284800], # wild finetune 11b 40
        # checkpoint_steps = [1304900], # wild finetune 11b 60
        # checkpoint_steps = [1304900],  # wild finetune 11b 80
        # checkpoint_steps = [1284800],  # wild finetune 11b 100
        # checkpoint_steps = [1284800],  # wild finetune 11b woz 100
        split=split,
    )


if __name__ == "__main__":
    evaluate()
