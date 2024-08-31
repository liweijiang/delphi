"""
Data tasks
"""
import util
import os
import t5
import tensorflow.compat.v1 as tf
import sys
import seqio
import functools

print("python", sys.version)
print("t5", t5.__version__)
print("tf", tf.__version__)
print("seqio", seqio.__version__)


# DEFAULT_OUTPUT_FEATURES = {
#     "inputs":
#         seqio.Feature(
#             vocabulary=t5.data.get_default_vocabulary(), add_eos=True),
#     "targets":
#         seqio.Feature(
#             vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
# }
#
# ############################################# {data_version} sbic #############################################
# data_version = "v10"
# MORALITY_SBIC_JOINT_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_sbic_joint/"
#
# sbic_moral_acceptability_tsv_path = {
#     "test": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_acceptability/test.moral_acceptability.tsv"),
#     "train": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_acceptability/train.moral_acceptability.tsv"),
#     "validation": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_acceptability/validation.moral_acceptability.tsv")
# }
#
# sbic_moral_agreement_tsv_path = {
#     "test": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_agreement/test.moral_agreement.tsv"),
#     "train": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_agreement/train.moral_agreement.tsv"),
#     "validation": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_agreement/validation.moral_agreement.tsv")
# }
#
# sbic_moral_comparison_tsv_path = {
#     "test": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/test.moral_comparison.tsv"),
#     "train": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/train.moral_comparison.tsv"),
#     "validation": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/validation.moral_comparison.tsv")
# }
#
# ################## commonsense morality {data_version} sbic joint ##################
# seqio.TaskRegistry.add(
#     "sbic_moral_acceptability",
#     # Specify the task source.
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=sbic_moral_acceptability_tsv_path,
#         # Not required, but helps for mixing and auto-caching.
#         num_input_examples=util.get_num_elements_split(sbic_moral_acceptability_tsv_path)
#     ),
#     # Supply a list of functions that preprocess the input tf.data.Dataset.
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     # Lowercase targets before computing metrics.
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     # We'll use accuracy as our evaluation metric.
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "sbic_moral_agreement",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=sbic_moral_agreement_tsv_path,
#         num_input_examples=util.get_num_elements_split(sbic_moral_agreement_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "sbic_moral_comparison",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=sbic_moral_comparison_tsv_path,
#         num_input_examples=util.get_num_elements_split(sbic_moral_comparison_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# util.print_task_examples("sbic_moral_acceptability")
# util.print_task_examples("sbic_moral_agreement")
# util.print_task_examples("sbic_moral_comparison")
#
#
# ################## in the wild data ##################
#
# BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_wild/"
#
# gender_test_tsv_path = {
#     "test": os.path.join(BASE_DATA_DIR, f"{data_version}_gender_test.tsv"),
#     "train": os.path.join(BASE_DATA_DIR, f"{data_version}_gender_test.tsv"),
#     "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_gender_test.tsv")
# }
#
# race_test_tsv_path = {
#     "test": os.path.join(BASE_DATA_DIR, f"{data_version}_race_test.tsv"),
#     "train": os.path.join(BASE_DATA_DIR, f"{data_version}_race_test.tsv"),
#     "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_race_test.tsv")
# }
#
# general_test_tsv_path = {
#     "test": os.path.join(BASE_DATA_DIR, f"{data_version}_general_test.tsv"),
#     "train": os.path.join(BASE_DATA_DIR, f"{data_version}_general_test.tsv"),
#     "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_general_test.tsv")
# }
#
# proportions = [20, 40, 60, 80, 100]
# tasks = ["", "woz"]
# for p in proportions:
#     for t in tasks:
#         if t == "woz":
#             wild_train_tsv_path = {
#                 "test": os.path.join(BASE_DATA_DIR, f"{data_version}_train_woz_{p}.tsv"),
#                 "train": os.path.join(BASE_DATA_DIR, f"{data_version}_train_woz_{p}.tsv"),
#                 "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_train_woz_{p}.tsv")
#             }
#             task_name = f"wild_train_woz_{p}"
#         else:
#             wild_train_tsv_path = {
#                 "test": os.path.join(BASE_DATA_DIR, f"{data_version}_train_{p}.tsv"),
#                 "train": os.path.join(BASE_DATA_DIR, f"{data_version}_train_{p}.tsv"),
#                 "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_train_{p}.tsv")
#             }
#             task_name = f"wild_train_{p}"
#
#
#         seqio.TaskRegistry.add(
#             task_name,
#             # Specify the task source.
#             source=seqio.TextLineDataSource(
#                 split_to_filepattern=wild_train_tsv_path,
#                 # Not required, but helps for mixing and auto-caching.
#                 num_input_examples=util.get_num_elements_split(wild_train_tsv_path)
#             ),
#             # Supply a list of functions that preprocess the input tf.data.Dataset.
#             preprocessors=[
#                 functools.partial(
#                     t5.data.preprocessors.parse_tsv,
#                     field_names=["inputs", "targets"]),
#                 seqio.preprocessors.tokenize_and_append_eos,
#             ],
#             # Lowercase targets before computing metrics.
#             postprocess_fn=t5.data.postprocessors.lower_text,
#             # We'll use accuracy as our evaluation metric.
#             metric_fns=[t5.evaluation.metrics.accuracy],
#             output_features=DEFAULT_OUTPUT_FEATURES,
#         )
#
#         util.print_task_examples(task_name)


DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}

############################################# {data_version} #############################################
data_version = "v9"

MORALITY_JOINT_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_joint/"
MORALITY_SEPARATE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_separate/"

moral_acceptability_tsv_path = {
    "test": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_acceptability/test.moral_acceptability.tsv"),
    "train": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_acceptability/train.moral_acceptability.tsv"),
    "validation": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_acceptability/validation.moral_acceptability.tsv")
}

moral_agreement_tsv_path = {
    "test": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_agreement/test.moral_agreement.tsv"),
    "train": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_agreement/train.moral_agreement.tsv"),
    "validation": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_agreement/validation.moral_agreement.tsv")
}

moral_comparison_tsv_path = {
    "test": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_comparison/test.moral_comparison.tsv"),
    "train": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_comparison/train.moral_comparison.tsv"),
    "validation": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_comparison/validation.moral_comparison.tsv")
}

moral_acceptability_class_tsv_path = {
    "test": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_acceptability_class/test.moral_acceptability_class.tsv"),
    "train": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_acceptability_class/train.moral_acceptability_class.tsv"),
    "validation": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_acceptability_class/validation.moral_acceptability_class.tsv")
}

moral_acceptability_text_tsv_path = {
    "test": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_acceptability_text/test.moral_acceptability_text.tsv"),
    "train": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_acceptability_text/train.moral_acceptability_text.tsv"),
    "validation": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_acceptability_text/validation.moral_acceptability_text.tsv")
}

moral_agreement_class_tsv_path = {
    "test": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_agreement_class/test.moral_agreement_class.tsv"),
    "train": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_agreement_class/train.moral_agreement_class.tsv"),
    "validation": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_agreement_class/validation.moral_agreement_class.tsv")
}

moral_agreement_text_tsv_path = {
    "test": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_agreement_text/test.moral_agreement_text.tsv"),
    "train": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_agreement_text/train.moral_agreement_text.tsv"),
    "validation": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_agreement_text/validation.moral_agreement_text.tsv")
}

moral_comparison_class_tsv_path = {
    "test": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_comparison_class/test.moral_comparison_class.tsv"),
    "train": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_comparison_class/train.moral_comparison_class.tsv"),
    "validation": os.path.join(MORALITY_SEPARATE_DATA_DIR, "moral_comparison_class/validation.moral_comparison_class.tsv")
}


############################################# {data_version} sbic #############################################
MORALITY_SBIC_JOINT_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_sbic_joint/"
MORALITY_SBIC_SEPARATE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_sbic_separate/"

sbic_moral_acceptability_tsv_path = {
    "test": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_acceptability/test.moral_acceptability.tsv"),
    "train": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_acceptability/train.moral_acceptability.tsv"),
    "validation": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_acceptability/validation.moral_acceptability.tsv")
}

sbic_moral_agreement_tsv_path = {
    "test": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_agreement/test.moral_agreement.tsv"),
    "train": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_agreement/train.moral_agreement.tsv"),
    "validation": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_agreement/validation.moral_agreement.tsv")
}

sbic_moral_comparison_tsv_path = {
    "test": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/test.moral_comparison.tsv"),
    "train": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/train.moral_comparison.tsv"),
    "validation": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/validation.moral_comparison.tsv")
}

sbic_moral_acceptability_class_tsv_path = {
    "test": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_class/test.moral_acceptability_class.tsv"),
    "train": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_class/train.moral_acceptability_class.tsv"),
    "validation": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_class/validation.moral_acceptability_class.tsv")
}

sbic_moral_acceptability_text_tsv_path = {
    "test": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_text/test.moral_acceptability_text.tsv"),
    "train": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_text/train.moral_acceptability_text.tsv"),
    "validation": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_text/validation.moral_acceptability_text.tsv")
}

sbic_moral_agreement_class_tsv_path = {
    "test": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_class/test.moral_agreement_class.tsv"),
    "train": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_class/train.moral_agreement_class.tsv"),
    "validation": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_class/validation.moral_agreement_class.tsv")
}

sbic_moral_agreement_text_tsv_path = {
    "test": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_text/test.moral_agreement_text.tsv"),
    "train": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_text/train.moral_agreement_text.tsv"),
    "validation": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_text/validation.moral_agreement_text.tsv")
}

sbic_moral_comparison_class_tsv_path = {
    "test": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_comparison_class/test.moral_comparison_class.tsv"),
    "train": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_comparison_class/train.moral_comparison_class.tsv"),
    "validation": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_comparison_class/validation.moral_comparison_class.tsv")
}


sbic_moral_comparison_double_tsv_path = {
    "test": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/test.moral_comparison.tsv"),
    "train": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/train.moral_comparison_double.tsv"),
    "validation": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/validation.moral_comparison.tsv")
}

# ################## commonsense morality {data_version} joint ##################
# seqio.TaskRegistry.add(
#     "moral_acceptability",

#     # Specify the task source.
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=moral_acceptability_tsv_path,
#         # Not required, but helps for mixing and auto-caching.
#         num_input_examples=util.get_num_elements_split(moral_acceptability_tsv_path)
#     ),
#     # Supply a list of functions that preprocess the input tf.data.Dataset.
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     # Lowercase targets before computing metrics.
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     # We'll use accuracy as our evaluation metric.
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "moral_agreement",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=moral_agreement_tsv_path,
#         num_input_examples=util.get_num_elements_split(moral_agreement_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "moral_comparison",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=moral_comparison_tsv_path,
#         num_input_examples=util.get_num_elements_split(moral_comparison_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# util.print_task_examples("moral_acceptability")
# util.print_task_examples("moral_agreement")
# util.print_task_examples("moral_comparison")


# ################## commonsense morality {data_version} separate ##################
# seqio.TaskRegistry.add(
#     "moral_acceptability_class",
#     # Specify the task source.
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=moral_acceptability_class_tsv_path,
#         # Not required, but helps for mixing and auto-caching.
#         num_input_examples=util.get_num_elements_split(moral_acceptability_class_tsv_path)
#     ),
#     # Supply a list of functions that preprocess the input tf.data.Dataset.
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     # Lowercase targets before computing metrics.
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     # We'll use accuracy as our evaluation metric.
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "moral_acceptability_text",
#     # Specify the task source.
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=moral_acceptability_text_tsv_path,
#         # Not required, but helps for mixing and auto-caching.
#         num_input_examples=util.get_num_elements_split(moral_acceptability_text_tsv_path)
#     ),
#     # Supply a list of functions that preprocess the input tf.data.Dataset.
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     # Lowercase targets before computing metrics.
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     # We'll use accuracy as our evaluation metric.
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "moral_agreement_class",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=moral_agreement_class_tsv_path,
#         num_input_examples=util.get_num_elements_split(moral_agreement_class_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "moral_agreement_text",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=moral_agreement_text_tsv_path,
#         num_input_examples=util.get_num_elements_split(moral_agreement_text_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "moral_comparison_class",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=moral_comparison_class_tsv_path,
#         num_input_examples=util.get_num_elements_split(moral_comparison_class_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# util.print_task_examples("moral_acceptability_class")
# util.print_task_examples("moral_acceptability_text")
# util.print_task_examples("moral_agreement_class")
# util.print_task_examples("moral_agreement_text")
# util.print_task_examples("moral_comparison_class")


################## commonsense morality {data_version} sbic joint ##################
seqio.TaskRegistry.add(
    "sbic_moral_acceptability",
    # Specify the task source.
    source=seqio.TextLineDataSource(
        split_to_filepattern=sbic_moral_acceptability_tsv_path,
        # Not required, but helps for mixing and auto-caching.
        num_input_examples=util.get_num_elements_split(
            sbic_moral_acceptability_tsv_path)
    ),
    # Supply a list of functions that preprocess the input tf.data.Dataset.
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["inputs", "targets"]),
        seqio.preprocessors.tokenize_and_append_eos,
    ],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES,
)

seqio.TaskRegistry.add(
    "sbic_moral_agreement",
    source=seqio.TextLineDataSource(
        split_to_filepattern=sbic_moral_agreement_tsv_path,
        num_input_examples=util.get_num_elements_split(
            sbic_moral_agreement_tsv_path)
    ),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["inputs", "targets"]),
        seqio.preprocessors.tokenize_and_append_eos,
    ],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES,
)

seqio.TaskRegistry.add(
    "sbic_moral_comparison",
    source=seqio.TextLineDataSource(
        split_to_filepattern=sbic_moral_comparison_tsv_path,
        num_input_examples=util.get_num_elements_split(
            sbic_moral_comparison_tsv_path)
    ),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["inputs", "targets"]),
        seqio.preprocessors.tokenize_and_append_eos,
    ],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES,
)
#
# seqio.TaskRegistry.add(
#     "sbic_moral_comparison_double",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=sbic_moral_comparison_double_tsv_path,
#         num_input_examples=util.get_num_elements_split(sbic_moral_comparison_double_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )

util.print_task_examples("sbic_moral_acceptability")
util.print_task_examples("sbic_moral_agreement")
util.print_task_examples("sbic_moral_comparison")
# util.print_task_examples("sbic_moral_comparison_double")


# ################## commonsense morality {data_version} separate ##################
# seqio.TaskRegistry.add(
#     "sbic_moral_acceptability_class",
#     # Specify the task source.
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=sbic_moral_acceptability_class_tsv_path,
#         # Not required, but helps for mixing and auto-caching.
#         num_input_examples=util.get_num_elements_split(sbic_moral_acceptability_class_tsv_path)
#     ),
#     # Supply a list of functions that preprocess the input tf.data.Dataset.
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     # Lowercase targets before computing metrics.
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     # We'll use accuracy as our evaluation metric.
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "sbic_moral_acceptability_text",
#     # Specify the task source.
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=sbic_moral_acceptability_text_tsv_path,
#         # Not required, but helps for mixing and auto-caching.
#         num_input_examples=util.get_num_elements_split(sbic_moral_acceptability_text_tsv_path)
#     ),
#     # Supply a list of functions that preprocess the input tf.data.Dataset.
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     # Lowercase targets before computing metrics.
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     # We'll use accuracy as our evaluation metric.
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "sbic_moral_agreement_class",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=sbic_moral_agreement_class_tsv_path,
#         num_input_examples=util.get_num_elements_split(sbic_moral_agreement_class_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "sbic_moral_agreement_text",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=sbic_moral_agreement_text_tsv_path,
#         num_input_examples=util.get_num_elements_split(sbic_moral_agreement_text_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# seqio.TaskRegistry.add(
#     "sbic_moral_comparison_class",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=sbic_moral_comparison_class_tsv_path,
#         num_input_examples=util.get_num_elements_split(sbic_moral_comparison_class_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# util.print_task_examples("sbic_moral_acceptability_class")
# util.print_task_examples("sbic_moral_acceptability_text")
# util.print_task_examples("sbic_moral_agreement_class")
# util.print_task_examples("sbic_moral_agreement_text")
# util.print_task_examples("sbic_moral_comparison_class")


#     seqio.TaskRegistry.add(
#         f"sbic_moral_acceptability_{proportion}",
#         # Specify the task source.
#         source=seqio.TextLineDataSource(
#             split_to_filepattern=sbic_moral_acceptability_tsv_path,
#             # Not required, but helps for mixing and auto-caching.
#             num_input_examples=util.get_dataset_proportion(util.get_num_elements_split(sbic_moral_acceptability_tsv_path), proportion)
#         ),
#         # Supply a list of functions that preprocess the input tf.data.Dataset.
#         preprocessors=[
#           functools.partial(
#               t5.data.preprocessors.parse_tsv,
#               field_names=["inputs", "targets"]),
#           seqio.preprocessors.tokenize_and_append_eos,
#         ],
#         # Lowercase targets before computing metrics.
#         postprocess_fn=t5.data.postprocessors.lower_text,
#         # We'll use accuracy as our evaluation metric.
#         metric_fns=[t5.evaluation.metrics.accuracy],
#         output_features=DEFAULT_OUTPUT_FEATURES,
#     )
#
#     seqio.TaskRegistry.add(
#         f"sbic_moral_agreement_{proportion}",
#         source=seqio.TextLineDataSource(
#             split_to_filepattern=sbic_moral_agreement_tsv_path,
#             num_input_examples=util.get_dataset_proportion(util.get_num_elements_split(sbic_moral_agreement_tsv_path), proportion)
#         ),
#         preprocessors=[
#           functools.partial(
#               t5.data.preprocessors.parse_tsv,
#               field_names=["inputs", "targets"]),
#           seqio.preprocessors.tokenize_and_append_eos,
#         ],
#         postprocess_fn=t5.data.postprocessors.lower_text,
#         metric_fns=[t5.evaluation.metrics.accuracy],
#         output_features=DEFAULT_OUTPUT_FEATURES,
#     )
#
#     seqio.TaskRegistry.add(
#         f"sbic_moral_comparison_{proportion}",
#         source=seqio.TextLineDataSource(
#             split_to_filepattern=sbic_moral_comparison_tsv_path,
#             num_input_examples=util.get_dataset_proportion(util.get_num_elements_split(sbic_moral_comparison_tsv_path), proportion)
#         ),
#         preprocessors=[
#           functools.partial(
#               t5.data.preprocessors.parse_tsv,
#               field_names=["inputs", "targets"]),
#           seqio.preprocessors.tokenize_and_append_eos,
#         ],
#         postprocess_fn=t5.data.postprocessors.lower_text,
#         metric_fns=[t5.evaluation.metrics.accuracy],
#         output_features=DEFAULT_OUTPUT_FEATURES,
#     )
#


# ################## ethics values raw (for fine-tuning) ##################
# ETHICS_VALUES_RAW_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/ethics_values_raw/"
#
# ethics_cm_long_raw_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_cm_long_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_cm_long_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_cm_long_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_cm_long_raw",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_cm_long_raw_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_cm_long_raw_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# ethics_cm_raw_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_cm_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_cm_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_cm_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_cm_raw",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_cm_raw_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_cm_raw_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# ethics_deontology_raw_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_deontology_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_deontology_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_deontology_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_deontology_raw",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_deontology_raw_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_deontology_raw_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# ethics_justice_raw_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_justice_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_justice_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_justice_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_justice_raw",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_justice_raw_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_justice_raw_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# ethics_util_raw_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_util_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_util_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_util_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_util_raw",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_util_raw_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_util_raw_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# ethics_virtue_raw_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_virtue_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_virtue_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_RAW_DATA_DIR, "ethics_virtue_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_virtue_raw",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_virtue_raw_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_virtue_raw_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# ################## ethics values (for pre-train on ethics) ##################
# ETHICS_VALUES_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/ethics_values/"
#
# ethics_cm_long_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_cm_long_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_cm_long_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_cm_long_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_cm_long",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_cm_long_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_cm_long_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
# # util.print_task_examples("ethics_cm_long", split="train", num_ex=1)
#
# ethics_cm_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_cm_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_cm_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_cm_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_cm",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_cm_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_cm_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# ethics_deontology_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_deontology_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_deontology_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_deontology_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_deontology",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_deontology_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_deontology_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# ethics_justice_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_justice_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_justice_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_justice_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_justice",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_justice_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_justice_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# ethics_util_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_util_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_util_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_util_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_util",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_util_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_util_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# ethics_virtue_tsv_path = {
#     "test": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_virtue_test.tsv"),
#     "test_hard": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_virtue_test_hard.tsv"),
#     "train": os.path.join(ETHICS_VALUES_DATA_DIR, "ethics_virtue_train.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "ethics_virtue",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=ethics_virtue_tsv_path,
#         num_input_examples=util.get_num_elements_split(ethics_virtue_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#

# ################## demo examples ##################
# DEMO_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/demo/"
#
# demo_v0_tsv_path = {
#     "train": os.path.join(DEMO_DATA_DIR, "demo_v0.tsv"),
#     "validation": os.path.join(DEMO_DATA_DIR, "demo_v0.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "demo_v0",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=demo_v0_tsv_path,
#         num_input_examples=util.get_num_elements_split(demo_v0_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# demo_v1_tsv_path = {
#     "train": os.path.join(DEMO_DATA_DIR, "demo_v1.tsv"),
#     "validation": os.path.join(DEMO_DATA_DIR, "demo_v1.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "demo_v1",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=demo_v1_tsv_path,
#         num_input_examples=util.get_num_elements_split(demo_v1_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# demo_v2_tsv_path = {
#     "train": os.path.join(DEMO_DATA_DIR, "demo_v2.tsv"),
#     "validation": os.path.join(DEMO_DATA_DIR, "demo_v2.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "demo_v2",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=demo_v2_tsv_path,
#         num_input_examples=util.get_num_elements_split(demo_v2_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# demo_v3_tsv_path = {
#     "train": os.path.join(DEMO_DATA_DIR, "demo_v3.tsv"),
#     "validation": os.path.join(DEMO_DATA_DIR, "demo_v3.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "demo_v3",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=demo_v3_tsv_path,
#         num_input_examples=util.get_num_elements_split(demo_v3_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# demo_v4_tsv_path = {
#     "train": os.path.join(DEMO_DATA_DIR, "demo_v4.tsv"),
#     "validation": os.path.join(DEMO_DATA_DIR, "demo_v4.tsv"),
#     "test": os.path.join(DEMO_DATA_DIR, "demo_v4.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "demo_v4",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=demo_v4_tsv_path,
#         num_input_examples=util.get_num_elements_split(demo_v4_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# demo_v5_tsv_path = {
#     "train": os.path.join(DEMO_DATA_DIR, "demo_v5.tsv"),
#     "validation": os.path.join(DEMO_DATA_DIR, "demo_v5.tsv"),
#     "test": os.path.join(DEMO_DATA_DIR, "demo_v5.tsv"),
# }
#
# seqio.TaskRegistry.add(
#     "demo_v5",
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=demo_v5_tsv_path,
#         num_input_examples=util.get_num_elements_split(demo_v5_tsv_path)
#     ),
#     preprocessors=[
#       functools.partial(
#           t5.data.preprocessors.parse_tsv,
#           field_names=["inputs", "targets"]),
#       seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )


################## commonsense morality {data_version} sbic joint (10, 30, 60 percent) ##################
proportions = [0.01]  # , 1, 10, 30, 60, 90, "base"
for proportion in proportions:
    if proportion == "base":
        MORALITY_JOINT_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_ablation/compositionality/{proportion}/"
    else:
        MORALITY_JOINT_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_ablation/scale/{proportion}/"

    sbic_moral_acceptability_tsv_path = {
        "test": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_acceptability/test.moral_acceptability.tsv"),
        "train": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_acceptability/train.moral_acceptability.tsv"),
        "validation": os.path.join(MORALITY_JOINT_DATA_DIR,
                                   "moral_acceptability/validation.moral_acceptability.tsv")
    }

    sbic_moral_agreement_tsv_path = {
        "test": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_agreement/test.moral_agreement.tsv"),
        "train": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_agreement/train.moral_agreement.tsv"),
        "validation": os.path.join(MORALITY_JOINT_DATA_DIR,
                                   "moral_agreement/validation.moral_agreement.tsv")
    }

    sbic_moral_comparison_tsv_path = {
        "test": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_comparison/test.moral_comparison.tsv"),
        "train": os.path.join(MORALITY_JOINT_DATA_DIR, "moral_comparison/train.moral_comparison.tsv"),
        "validation": os.path.join(MORALITY_JOINT_DATA_DIR,
                                   "moral_comparison/validation.moral_comparison.tsv")
    }

    seqio.TaskRegistry.add(
        f"sbic_moral_acceptability_{proportion}",
        # Specify the task source.
        source=seqio.TextLineDataSource(
            split_to_filepattern=sbic_moral_acceptability_tsv_path,
            # Not required, but helps for mixing and auto-caching.
            num_input_examples=util.get_num_elements_split(
                sbic_moral_acceptability_tsv_path)
        ),
        # Supply a list of functions that preprocess the input tf.data.Dataset.
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.parse_tsv,
                field_names=["inputs", "targets"]),
            seqio.preprocessors.tokenize_and_append_eos,
        ],
        # Lowercase targets before computing metrics.
        postprocess_fn=t5.data.postprocessors.lower_text,
        # We'll use accuracy as our evaluation metric.
        metric_fns=[t5.evaluation.metrics.accuracy],
        output_features=DEFAULT_OUTPUT_FEATURES,
    )

    seqio.TaskRegistry.add(
        f"sbic_moral_agreement_{proportion}",
        source=seqio.TextLineDataSource(
            split_to_filepattern=sbic_moral_agreement_tsv_path,
            num_input_examples=util.get_num_elements_split(
                sbic_moral_agreement_tsv_path)
        ),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.parse_tsv,
                field_names=["inputs", "targets"]),
            seqio.preprocessors.tokenize_and_append_eos,
        ],
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy],
        output_features=DEFAULT_OUTPUT_FEATURES,
    )

    seqio.TaskRegistry.add(
        f"sbic_moral_comparison_{proportion}",
        source=seqio.TextLineDataSource(
            split_to_filepattern=sbic_moral_comparison_tsv_path,
            num_input_examples=util.get_num_elements_split(
                sbic_moral_comparison_tsv_path)
        ),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.parse_tsv,
                field_names=["inputs", "targets"]),
            seqio.preprocessors.tokenize_and_append_eos,
        ],
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy],
        output_features=DEFAULT_OUTPUT_FEATURES,
    )

    util.print_task_examples(f"sbic_moral_acceptability_{proportion}")
    util.print_task_examples(f"sbic_moral_agreement_{proportion}")
    util.print_task_examples(f"sbic_moral_comparison_{proportion}")


# ================================================== wild ==================================================

BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_wild/"

race_test_tsv_path = {
    "test": os.path.join(BASE_DATA_DIR, f"{data_version}_race_test.tsv"),
    "train": os.path.join(BASE_DATA_DIR, f"{data_version}_race_test.tsv"),
    "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_race_test.tsv")
}

seqio.TaskRegistry.add(
    f"race_test",
    # Specify the task source.
    source=seqio.TextLineDataSource(
        split_to_filepattern=race_test_tsv_path,
        # Not required, but helps for mixing and auto-caching.
        num_input_examples=util.get_num_elements_split(race_test_tsv_path)
    ),
    # Supply a list of functions that preprocess the input tf.data.Dataset.
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["inputs", "targets"]),
        seqio.preprocessors.tokenize_and_append_eos,
    ],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES,
)


gender_test_tsv_path = {
    "test": os.path.join(BASE_DATA_DIR, f"{data_version}_gender_test.tsv"),
    "train": os.path.join(BASE_DATA_DIR, f"{data_version}_gender_test.tsv"),
    "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_gender_test.tsv")
}

seqio.TaskRegistry.add(
    f"gender_test",
    # Specify the task source.
    source=seqio.TextLineDataSource(
        split_to_filepattern=gender_test_tsv_path,
        # Not required, but helps for mixing and auto-caching.
        num_input_examples=util.get_num_elements_split(gender_test_tsv_path)
    ),
    # Supply a list of functions that preprocess the input tf.data.Dataset.
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["inputs", "targets"]),
        seqio.preprocessors.tokenize_and_append_eos,
    ],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES,
)


BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_wild/"
#
proportions = [100]  # 10, 20, 40, 60, 80,
tasks = ["", "woz"]
for p in proportions:
    for t in tasks:
        if t == "woz":
            wild_train_tsv_path = {
                "test": os.path.join(BASE_DATA_DIR, f"{data_version}_general_test.tsv"),
                "train": os.path.join(BASE_DATA_DIR, f"{data_version}_train_woz_{p}.tsv"),
                "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_dev.tsv")
            }
            task_name = f"wild_train_woz_{p}"
        else:
            wild_train_tsv_path = {
                "test": os.path.join(BASE_DATA_DIR, f"{data_version}_general_test.tsv"),
                "train": os.path.join(BASE_DATA_DIR, f"{data_version}_train_{p}.tsv"),
                "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_dev.tsv"),
            }
            task_name = f"wild_train_{p}"

        seqio.TaskRegistry.add(
            task_name,
            # Specify the task source.
            source=seqio.TextLineDataSource(
                split_to_filepattern=wild_train_tsv_path,
                # Not required, but helps for mixing and auto-caching.
                num_input_examples=util.get_num_elements_split(
                    wild_train_tsv_path)
            ),
            # Supply a list of functions that preprocess the input tf.data.Dataset.
            preprocessors=[
                functools.partial(
                    t5.data.preprocessors.parse_tsv,
                    field_names=["inputs", "targets"]),
                seqio.preprocessors.tokenize_and_append_eos,
            ],
            # Lowercase targets before computing metrics.
            postprocess_fn=t5.data.postprocessors.lower_text,
            # We'll use accuracy as our evaluation metric.
            metric_fns=[t5.evaluation.metrics.accuracy],
            output_features=DEFAULT_OUTPUT_FEATURES,
        )

        util.print_task_examples(task_name)


# # ================================================== wild (zero-shot) ==================================================
#
# BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v10_wild/zero_shot/"
#
# options = {
#     "good": [
#                 "good.",
#                 "nice.",
#                 "ethical.",
#                 "moral.",
#                 "correct.",
#                 "acceptable."
#             ],
#
#     "bad": [
#                 "unacceptable.",
#                 "wrong.",
#                 "rude.",
#                 "bad.",
#                 "unethical.",
#                 "immoral."
#             ],
#
#     "ok": [
#                 "okay.",
#                 "ok.",
#                 "understandable.",
#                 "expected.",
#                 "fine.",
#                 "normal."
#             ]
# }
#
# # proportions = [10, 20, 40, 60, 80, 100]
# # tasks = ["", "woz"]
# test_tasks = ["race", "general", "gender"]
# for t in test_tasks:
#     for k in options:
#         for _o in options[k]:
#             data_path = BASE_DATA_DIR + f"v10_{t}_test_{k}_{_o}tsv"
#
#             # print(data_path)
#
#             data_tsv_path = {
#                 "test": data_path,
#                 "train": data_path,
#                 "validation": data_path
#             }
#             task_name = f"{t}_{k}_{_o}"[:-1]
#             print(task_name)
#
#             seqio.TaskRegistry.add(
#                 task_name,
#                 # Specify the task source.
#                 source=seqio.TextLineDataSource(
#                     split_to_filepattern=data_tsv_path,
#                     # Not required, but helps for mixing and auto-caching.
#                     num_input_examples=util.get_num_elements_split(data_tsv_path)
#                 ),
#                 # Supply a list of functions that preprocess the input tf.data.Dataset.
#                 preprocessors=[
#                     functools.partial(
#                         t5.data.preprocessors.parse_tsv,
#                         field_names=["inputs", "targets"]),
#                     seqio.preprocessors.tokenize_and_append_eos,
#                 ],
#                 # Lowercase targets before computing metrics.
#                 postprocess_fn=t5.data.postprocessors.lower_text,
#                 # We'll use accuracy as our evaluation metric.
#                 metric_fns=[t5.evaluation.metrics.accuracy],
#                 output_features=DEFAULT_OUTPUT_FEATURES,
#             )
#
#             # util.print_task_examples(task_name)


# ================================================== dynahate ==================================================
dynahate = False
if dynahate:
    for round_id in [1, 2, 3, 4]:
        # BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/{round_id}/"
        #
        # dynahate_tsv_path = {
        #     "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
        #     "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
        #     "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
        # }
        #
        # seqio.TaskRegistry.add(
        #     f"dynahate_round_{round_id}",
        #     # Specify the task source.
        #     source=seqio.TextLineDataSource(
        #         split_to_filepattern=dynahate_tsv_path,
        #         # Not required, but helps for mixing and auto-caching.
        #         num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
        #     ),
        #     # Supply a list of functions that preprocess the input tf.data.Dataset.
        #     preprocessors=[
        #       functools.partial(
        #           t5.data.preprocessors.parse_tsv,
        #           field_names=["inputs", "targets"]),
        #       seqio.preprocessors.tokenize_and_append_eos,
        #     ],
        #     # Lowercase targets before computing metrics.
        #     postprocess_fn=t5.data.postprocessors.lower_text,
        #     # We'll use accuracy as our evaluation metric.
        #     metric_fns=[t5.evaluation.metrics.accuracy],
        #     output_features=DEFAULT_OUTPUT_FEATURES,
        # )
        #
        #
        # BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/100-shot/{round_id}/"
        # dynahate_tsv_path = {
        #     "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
        #     "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
        #     "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
        # }
        #
        # seqio.TaskRegistry.add(
        #     f"dynahate_round_{round_id}_100_shot",
        #     # Specify the task source.
        #     source=seqio.TextLineDataSource(
        #         split_to_filepattern=dynahate_tsv_path,
        #         # Not required, but helps for mixing and auto-caching.
        #         num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
        #     ),
        #     # Supply a list of functions that preprocess the input tf.data.Dataset.
        #     preprocessors=[
        #         functools.partial(
        #             t5.data.preprocessors.parse_tsv,
        #             field_names=["inputs", "targets"]),
        #         seqio.preprocessors.tokenize_and_append_eos,
        #     ],
        #     # Lowercase targets before computing metrics.
        #     postprocess_fn=t5.data.postprocessors.lower_text,
        #     # We'll use accuracy as our evaluation metric.
        #     metric_fns=[t5.evaluation.metrics.accuracy],
        #     output_features=DEFAULT_OUTPUT_FEATURES,
        # )

        # BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/{round_id}_nat/"
        #
        # dynahate_tsv_path = {
        #     "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
        #     "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
        #     "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
        # }
        #
        # seqio.TaskRegistry.add(
        #     f"dynahate_round_{round_id}_nat",
        #     # Specify the task source.
        #     source=seqio.TextLineDataSource(
        #         split_to_filepattern=dynahate_tsv_path,
        #         # Not required, but helps for mixing and auto-caching.
        #         num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
        #     ),
        #     # Supply a list of functions that preprocess the input tf.data.Dataset.
        #     preprocessors=[
        #       functools.partial(
        #           t5.data.preprocessors.parse_tsv,
        #           field_names=["inputs", "targets"]),
        #       seqio.preprocessors.tokenize_and_append_eos,
        #     ],
        #     # Lowercase targets before computing metrics.
        #     postprocess_fn=t5.data.postprocessors.lower_text,
        #     # We'll use accuracy as our evaluation metric.
        #     metric_fns=[t5.evaluation.metrics.accuracy],
        #     output_features=DEFAULT_OUTPUT_FEATURES,
        # )
        #
        #
        # BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/100-shot/{round_id}_nat/"
        # dynahate_tsv_path = {
        #     "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
        #     "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
        #     "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
        # }
        #
        # seqio.TaskRegistry.add(
        #     f"dynahate_round_{round_id}_nat_100_shot",
        #     # Specify the task source.
        #     source=seqio.TextLineDataSource(
        #         split_to_filepattern=dynahate_tsv_path,
        #         # Not required, but helps for mixing and auto-caching.
        #         num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
        #     ),
        #     # Supply a list of functions that preprocess the input tf.data.Dataset.
        #     preprocessors=[
        #         functools.partial(
        #             t5.data.preprocessors.parse_tsv,
        #             field_names=["inputs", "targets"]),
        #         seqio.preprocessors.tokenize_and_append_eos,
        #     ],
        #     # Lowercase targets before computing metrics.
        #     postprocess_fn=t5.data.postprocessors.lower_text,
        #     # We'll use accuracy as our evaluation metric.
        #     metric_fns=[t5.evaluation.metrics.accuracy],
        #     output_features=DEFAULT_OUTPUT_FEATURES,
        # )

        BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/{round_id}_st/"

        dynahate_tsv_path = {
            "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
            "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
            "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
        }

        seqio.TaskRegistry.add(
            f"dynahate_round_{round_id}_st",
            # Specify the task source.
            source=seqio.TextLineDataSource(
                split_to_filepattern=dynahate_tsv_path,
                # Not required, but helps for mixing and auto-caching.
                num_input_examples=util.get_num_elements_split(
                    dynahate_tsv_path)
            ),
            # Supply a list of functions that preprocess the input tf.data.Dataset.
            preprocessors=[
                functools.partial(
                    t5.data.preprocessors.parse_tsv,
                    field_names=["inputs", "targets"]),
                seqio.preprocessors.tokenize_and_append_eos,
            ],
            # Lowercase targets before computing metrics.
            postprocess_fn=t5.data.postprocessors.lower_text,
            # We'll use accuracy as our evaluation metric.
            metric_fns=[t5.evaluation.metrics.accuracy],
            output_features=DEFAULT_OUTPUT_FEATURES,
        )

        BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/100-shot/{round_id}_st/"
        dynahate_tsv_path = {
            "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
            "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
            "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
        }

        seqio.TaskRegistry.add(
            f"dynahate_round_{round_id}_st_100_shot",
            # Specify the task source.
            source=seqio.TextLineDataSource(
                split_to_filepattern=dynahate_tsv_path,
                # Not required, but helps for mixing and auto-caching.
                num_input_examples=util.get_num_elements_split(
                    dynahate_tsv_path)
            ),
            # Supply a list of functions that preprocess the input tf.data.Dataset.
            preprocessors=[
                functools.partial(
                    t5.data.preprocessors.parse_tsv,
                    field_names=["inputs", "targets"]),
                seqio.preprocessors.tokenize_and_append_eos,
            ],
            # Lowercase targets before computing metrics.
            postprocess_fn=t5.data.postprocessors.lower_text,
            # We'll use accuracy as our evaluation metric.
            metric_fns=[t5.evaluation.metrics.accuracy],
            output_features=DEFAULT_OUTPUT_FEATURES,
        )

        # BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/{round_id}_st_clean/"
        #
        # dynahate_tsv_path = {
        #     "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
        #     "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
        #     "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
        # }
        #
        # seqio.TaskRegistry.add(
        #     f"dynahate_round_{round_id}_st_clean",
        #     # Specify the task source.
        #     source=seqio.TextLineDataSource(
        #         split_to_filepattern=dynahate_tsv_path,
        #         # Not required, but helps for mixing and auto-caching.
        #         num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
        #     ),
        #     # Supply a list of functions that preprocess the input tf.data.Dataset.
        #     preprocessors=[
        #         functools.partial(
        #             t5.data.preprocessors.parse_tsv,
        #             field_names=["inputs", "targets"]),
        #         seqio.preprocessors.tokenize_and_append_eos,
        #     ],
        #     # Lowercase targets before computing metrics.
        #     postprocess_fn=t5.data.postprocessors.lower_text,
        #     # We'll use accuracy as our evaluation metric.
        #     metric_fns=[t5.evaluation.metrics.accuracy],
        #     output_features=DEFAULT_OUTPUT_FEATURES,
        # )
        #
        # BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/100-shot/{round_id}_st_clean/"
        # dynahate_tsv_path = {
        #     "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
        #     "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
        #     "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
        # }
        #
        # seqio.TaskRegistry.add(
        #     f"dynahate_round_{round_id}_st_clean_100_shot",
        #     # Specify the task source.
        #     source=seqio.TextLineDataSource(
        #         split_to_filepattern=dynahate_tsv_path,
        #         # Not required, but helps for mixing and auto-caching.
        #         num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
        #     ),
        #     # Supply a list of functions that preprocess the input tf.data.Dataset.
        #     preprocessors=[
        #         functools.partial(
        #             t5.data.preprocessors.parse_tsv,
        #             field_names=["inputs", "targets"]),
        #         seqio.preprocessors.tokenize_and_append_eos,
        #     ],
        #     # Lowercase targets before computing metrics.
        #     postprocess_fn=t5.data.postprocessors.lower_text,
        #     # We'll use accuracy as our evaluation metric.
        #     metric_fns=[t5.evaluation.metrics.accuracy],
        #     output_features=DEFAULT_OUTPUT_FEATURES,
        # )
        #
        # BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/{round_id}_bc/"
        #
        # dynahate_tsv_path = {
        #     "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
        #     "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
        #     "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
        # }
        #
        # seqio.TaskRegistry.add(
        #     f"dynahate_round_{round_id}_bc",
        #     # Specify the task source.
        #     source=seqio.TextLineDataSource(
        #         split_to_filepattern=dynahate_tsv_path,
        #         # Not required, but helps for mixing and auto-caching.
        #         num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
        #     ),
        #     # Supply a list of functions that preprocess the input tf.data.Dataset.
        #     preprocessors=[
        #         functools.partial(
        #             t5.data.preprocessors.parse_tsv,
        #             field_names=["inputs", "targets"]),
        #         seqio.preprocessors.tokenize_and_append_eos,
        #     ],
        #     # Lowercase targets before computing metrics.
        #     postprocess_fn=t5.data.postprocessors.lower_text,
        #     # We'll use accuracy as our evaluation metric.
        #     metric_fns=[t5.evaluation.metrics.accuracy],
        #     output_features=DEFAULT_OUTPUT_FEATURES,
        # )
        #
        # BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/dynahate/100-shot/{round_id}_bc/"
        # dynahate_tsv_path = {
        #     "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
        #     "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
        #     "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
        # }
        #
        # seqio.TaskRegistry.add(
        #     f"dynahate_round_{round_id}_bc_100_shot",
        #     # Specify the task source.
        #     source=seqio.TextLineDataSource(
        #         split_to_filepattern=dynahate_tsv_path,
        #         # Not required, but helps for mixing and auto-caching.
        #         num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
        #     ),
        #     # Supply a list of functions that preprocess the input tf.data.Dataset.
        #     preprocessors=[
        #         functools.partial(
        #             t5.data.preprocessors.parse_tsv,
        #             field_names=["inputs", "targets"]),
        #         seqio.preprocessors.tokenize_and_append_eos,
        #     ],
        #     # Lowercase targets before computing metrics.
        #     postprocess_fn=t5.data.postprocessors.lower_text,
        #     # We'll use accuracy as our evaluation metric.
        #     metric_fns=[t5.evaluation.metrics.accuracy],
        #     output_features=DEFAULT_OUTPUT_FEATURES,
        # )


################## ethics values ##################
ethics = False
if ethics:
    tasks = ["cm", "deontology", "justice", "util", "virtue"]

    for task in tasks:
        ETHICS_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/ethics/ethics_st/"
        # print("-" * 100)
        # print(os.path.join(ETHICS_DATA_DIR, f"{task}/test.tsv"))
        ethics_tsv_path = {
            "validation": os.path.join(ETHICS_DATA_DIR, f"{task}/test.tsv"),
            "test": os.path.join(ETHICS_DATA_DIR, f"{task}/test_hard.tsv"),
            "train": os.path.join(ETHICS_DATA_DIR, f"{task}/train.tsv"),
        }

        seqio.TaskRegistry.add(
            f"ethics_{task}",
            source=seqio.TextLineDataSource(
                split_to_filepattern=ethics_tsv_path,
                num_input_examples=util.get_num_elements_split(ethics_tsv_path)
            ),
            preprocessors=[
                functools.partial(
                    t5.data.preprocessors.parse_tsv,
                    field_names=["inputs", "targets"]),
                seqio.preprocessors.tokenize_and_append_eos,
            ],
            postprocess_fn=t5.data.postprocessors.lower_text,
            metric_fns=[t5.evaluation.metrics.accuracy],
            output_features=DEFAULT_OUTPUT_FEATURES,
        )

        ETHICS_DATA_DIR += "100-shot/"
        ethics_tsv_path = {
            "validation": os.path.join(ETHICS_DATA_DIR, f"{task}/test.tsv"),
            "test": os.path.join(ETHICS_DATA_DIR, f"{task}/test_hard.tsv"),
            "train": os.path.join(ETHICS_DATA_DIR, f"{task}/train.tsv"),
        }

        seqio.TaskRegistry.add(
            f"ethics_{task}_100_shot",
            source=seqio.TextLineDataSource(
                split_to_filepattern=ethics_tsv_path,
                num_input_examples=util.get_num_elements_split(ethics_tsv_path)
            ),
            preprocessors=[
                functools.partial(
                    t5.data.preprocessors.parse_tsv,
                    field_names=["inputs", "targets"]),
                seqio.preprocessors.tokenize_and_append_eos,
            ],
            postprocess_fn=t5.data.postprocessors.lower_text,
            metric_fns=[t5.evaluation.metrics.accuracy],
            output_features=DEFAULT_OUTPUT_FEATURES,
        )


################## latent hatred ##################
latenthatred = False
if latenthatred:
    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/latenthatred/latenthatred_st/"
    tsv_path = {
        "validation": os.path.join(DATA_DIR, f"dev.tsv"),
        "test": os.path.join(DATA_DIR, f"test.tsv"),
        "train": os.path.join(DATA_DIR, f"train.tsv"),
    }

    seqio.TaskRegistry.add(
        f"latenthatred",
        source=seqio.TextLineDataSource(
            split_to_filepattern=tsv_path,
            num_input_examples=util.get_num_elements_split(tsv_path)
        ),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.parse_tsv,
                field_names=["inputs", "targets"]),
            seqio.preprocessors.tokenize_and_append_eos,
        ],
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy],
        output_features=DEFAULT_OUTPUT_FEATURES,
    )

    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v9_downstream/latenthatred/latenthatred_st/100-shot/"
    tsv_path = {
        "validation": os.path.join(DATA_DIR, f"dev.tsv"),
        "test": os.path.join(DATA_DIR, f"test.tsv"),
        "train": os.path.join(DATA_DIR, f"train.tsv"),
    }

    seqio.TaskRegistry.add(
        f"latenthatred_100_shot",
        source=seqio.TextLineDataSource(
            split_to_filepattern=tsv_path,
            num_input_examples=util.get_num_elements_split(tsv_path)
        ),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.parse_tsv,
                field_names=["inputs", "targets"]),
            seqio.preprocessors.tokenize_and_append_eos,
        ],
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy],
        output_features=DEFAULT_OUTPUT_FEATURES,
    )


sciworld = False

if sciworld:
    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/data/"
    tsv_path = {
        "validation": os.path.join(DATA_DIR, f"sciworld_formatted_v2_val.csv"),
        "test": os.path.join(DATA_DIR, f"sciworld_formatted_v2_test.csv"),
        "train": os.path.join(DATA_DIR, f"sciworld_formatted_v2_train.csv"),
    }
    print(tsv_path)

    seqio.TaskRegistry.add(
        f"sciworld",
        source=seqio.TextLineDataSource(
            split_to_filepattern=tsv_path,
            num_input_examples=util.get_num_elements_split(tsv_path)
        ),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.parse_tsv,
                field_names=["inputs", "targets"]),
            seqio.preprocessors.tokenize_and_append_eos,
        ],
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy],
        output_features=DEFAULT_OUTPUT_FEATURES,
    )
