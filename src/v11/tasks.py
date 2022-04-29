import os
import t5
import seqio
import functools

import util

DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}

data_version = "v11"

############################################# {data_version} sbic #############################################
# MORALITY_SBIC_JOINT_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_sbic_joint/"
# MORALITY_SBIC_SEPARATE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_sbic_separate/"
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

# sbic_moral_acceptability_class_tsv_path = {
#     "test": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_class/test.moral_acceptability_class.tsv"),
#     "train": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_class/train.moral_acceptability_class.tsv"),
#     "validation": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_class/validation.moral_acceptability_class.tsv")
# }
#
# sbic_moral_acceptability_text_tsv_path = {
#     "test": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_text/test.moral_acceptability_text.tsv"),
#     "train": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_text/train.moral_acceptability_text.tsv"),
#     "validation": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_acceptability_text/validation.moral_acceptability_text.tsv")
# }
#
# sbic_moral_agreement_class_tsv_path = {
#     "test": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_class/test.moral_agreement_class.tsv"),
#     "train": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_class/train.moral_agreement_class.tsv"),
#     "validation": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_class/validation.moral_agreement_class.tsv")
# }
#
# sbic_moral_agreement_text_tsv_path = {
#     "test": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_text/test.moral_agreement_text.tsv"),
#     "train": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_text/train.moral_agreement_text.tsv"),
#     "validation": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_agreement_text/validation.moral_agreement_text.tsv")
# }
#
# sbic_moral_comparison_class_tsv_path = {
#     "test": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_comparison_class/test.moral_comparison_class.tsv"),
#     "train": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_comparison_class/train.moral_comparison_class.tsv"),
#     "validation": os.path.join(MORALITY_SBIC_SEPARATE_DATA_DIR, "moral_comparison_class/validation.moral_comparison_class.tsv")
# }
#
#
# sbic_moral_comparison_double_tsv_path = {
#     "test": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/test.moral_comparison.tsv"),
#     "train": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/train.moral_comparison_double.tsv"),
#     "validation": os.path.join(MORALITY_SBIC_JOINT_DATA_DIR, "moral_comparison/validation.moral_comparison.tsv")
# }


################## commonsense morality {data_version} sbic joint ##################
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
#
# util.print_task_examples("sbic_moral_acceptability")
# util.print_task_examples("sbic_moral_agreement")
# util.print_task_examples("sbic_moral_comparison")
#
#
# BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_wild/"
#
# race_test_tsv_path = {
#     "test": os.path.join(BASE_DATA_DIR, f"{data_version}_race_test.tsv"),
#     "train": os.path.join(BASE_DATA_DIR, f"{data_version}_race_test.tsv"),
#     "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_race_test.tsv")
# }
#
# seqio.TaskRegistry.add(
#     f"race_test",
#     # Specify the task source.
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=race_test_tsv_path,
#         # Not required, but helps for mixing and auto-caching.
#         num_input_examples=util.get_num_elements_split(race_test_tsv_path)
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
# gender_test_tsv_path = {
#     "test": os.path.join(BASE_DATA_DIR, f"{data_version}_gender_test.tsv"),
#     "train": os.path.join(BASE_DATA_DIR, f"{data_version}_gender_test.tsv"),
#     "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_gender_test.tsv")
# }
#
# seqio.TaskRegistry.add(
#     f"gender_test",
#     # Specify the task source.
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=gender_test_tsv_path,
#         # Not required, but helps for mixing and auto-caching.
#         num_input_examples=util.get_num_elements_split(gender_test_tsv_path)
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
# BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_wild/"
# proportions = [10, 20, 40, 60, 80, 100]
# tasks = ["", "woz"]
# for p in proportions:
#     for t in tasks:
#         if t == "woz":
#             wild_train_tsv_path = {
#                 "test": os.path.join(BASE_DATA_DIR, f"{data_version}_general_test.tsv"),
#                 "train": os.path.join(BASE_DATA_DIR, f"{data_version}_train_woz_{p}.tsv"),
#                 "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_dev.tsv")
#             }
#             task_name = f"wild_train_woz_{p}"
#         else:
#             wild_train_tsv_path = {
#                 "test": os.path.join(BASE_DATA_DIR, f"{data_version}_general_test.tsv"),
#                 "train": os.path.join(BASE_DATA_DIR, f"{data_version}_train_{p}.tsv"),
#                 "validation": os.path.join(BASE_DATA_DIR, f"{data_version}_dev.tsv"),
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



# BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/{data_version}_sbic_single_class_only/"
#
# sbic_moral_acceptability_tsv_path = {
#     "test": os.path.join(BASE_DATA_DIR, "moral_acceptability/test.moral_acceptability.tsv"),
#     "train": os.path.join(BASE_DATA_DIR, "moral_acceptability/train.moral_acceptability.tsv"),
#     "validation": os.path.join(BASE_DATA_DIR, "moral_acceptability/validation.moral_acceptability.tsv")
# }
#
# seqio.TaskRegistry.add(
#     f"sbic_moral_acceptability_single_class_only",
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
#
# sbic_moral_agreement_tsv_path = {
#     "test": os.path.join(BASE_DATA_DIR, "moral_agreement/test.moral_agreement.tsv"),
#     "train": os.path.join(BASE_DATA_DIR, "moral_agreement/train.moral_agreement.tsv"),
#     "validation": os.path.join(BASE_DATA_DIR, "moral_agreement/validation.moral_agreement.tsv")
# }
#
# seqio.TaskRegistry.add(
#     f"sbic_moral_agreement_single_class_only",
#     # Specify the task source.
#     source=seqio.TextLineDataSource(
#         split_to_filepattern=sbic_moral_agreement_tsv_path,
#         # Not required, but helps for mixing and auto-caching.
#         num_input_examples=util.get_num_elements_split(sbic_moral_agreement_tsv_path)
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


# # jiminy action distillation
#
# BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/jiminy_cricket/"
#
# for i in [0, 1, 2]:
#     jiminy_tsv_path = {
#         "test": os.path.join(BASE_DATA_DIR, f"floyd_jericho_undistilled_{i}.tsv"),
#         "train": os.path.join(BASE_DATA_DIR, f"floyd_jericho_undistilled_{i}.tsv"),
#         "validation": os.path.join(BASE_DATA_DIR, f"floyd_jericho_undistilled_{i}.tsv")
#     }
#
#     seqio.TaskRegistry.add(
#         f"jiminy_action_distill_{i}",
#         # Specify the task source.
#         source=seqio.TextLineDataSource(
#             split_to_filepattern=jiminy_tsv_path,
#             # Not required, but helps for mixing and auto-caching.
#             num_input_examples=util.get_num_elements_split(jiminy_tsv_path)
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
#
#


# # ================================================== dynahate ==================================================
# dynahate = True
# if dynahate:
#     for round_id in [1, 2, 3, 4]: # , 2, 3, 4
#         BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/dynahate/{round_id}_st/"
#
#         dynahate_tsv_path = {
#             "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
#             "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
#             "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
#         }
#
#         seqio.TaskRegistry.add(
#             f"new_dynahate_round_{round_id}_st",
#             # Specify the task source.
#             source=seqio.TextLineDataSource(
#                 split_to_filepattern=dynahate_tsv_path,
#                 # Not required, but helps for mixing and auto-caching.
#                 num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
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
#         BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/dynahate/100-shot/{round_id}_st/"
#         dynahate_tsv_path = {
#             "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
#             "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
#             "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
#         }
#
#         seqio.TaskRegistry.add(
#             f"new_dynahate_round_{round_id}_st_100_shot",
#             # Specify the task source.
#             source=seqio.TextLineDataSource(
#                 split_to_filepattern=dynahate_tsv_path,
#                 # Not required, but helps for mixing and auto-caching.
#                 num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
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
#
#
# ################## ethics values raw (for fine-tuning) ##################
# ethics = True
# if ethics:
#     ETHICS_VALUES_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/ethics/ethics_st/"
#     tasks = ["cm", "deontology", " justice", "util", " virtue"]
#     for task in tasks:
#         ethics_tsv_path = {
#             "validation": os.path.join(ETHICS_VALUES_DATA_DIR, f"{task}/test.tsv"),
#             "test": os.path.join(ETHICS_VALUES_DATA_DIR, f"{task}/test_hard.tsv"),
#             "train": os.path.join(ETHICS_VALUES_DATA_DIR, f"{task}/train.tsv"),
#         }
#
#         seqio.TaskRegistry.add(
#             f"new_ethics_{task}",
#             source=seqio.TextLineDataSource(
#                 split_to_filepattern=ethics_tsv_path,
#                 num_input_examples=util.get_num_elements_split(ethics_tsv_path)
#             ),
#             preprocessors=[
#               functools.partial(
#                   t5.data.preprocessors.parse_tsv,
#                   field_names=["inputs", "targets"]),
#               seqio.preprocessors.tokenize_and_append_eos,
#             ],
#             postprocess_fn=t5.data.postprocessors.lower_text,
#             metric_fns=[t5.evaluation.metrics.accuracy],
#             output_features=DEFAULT_OUTPUT_FEATURES,
#         )
#
#
# ################## latent hatred ##################
# latenthatred = True
# if latenthatred:
#     DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/latenthatred/latenthatred_st/"
#     tsv_path = {
#         "validation": os.path.join(DATA_DIR, f"dev.tsv"),
#         "test": os.path.join(DATA_DIR, f"test.tsv"),
#         "train": os.path.join(DATA_DIR, f"train.tsv"),
#     }
#
#     seqio.TaskRegistry.add(
#         f"new_latenthatred",
#         source=seqio.TextLineDataSource(
#             split_to_filepattern=tsv_path,
#             num_input_examples=util.get_num_elements_split(tsv_path)
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
#     DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/latenthatred/latenthatred_st/100-shot/"
#     tsv_path = {
#         "validation": os.path.join(DATA_DIR, f"dev.tsv"),
#         "test": os.path.join(DATA_DIR, f"test.tsv"),
#         "train": os.path.join(DATA_DIR, f"train.tsv"),
#     }
#
#     seqio.TaskRegistry.add(
#         f"new_latenthatred_100_shot",
#         source=seqio.TextLineDataSource(
#             split_to_filepattern=tsv_path,
#             num_input_examples=util.get_num_elements_split(tsv_path)
#         ),
#         preprocessors=[
#             functools.partial(
#                 t5.data.preprocessors.parse_tsv,
#                 field_names=["inputs", "targets"]),
#             seqio.preprocessors.tokenize_and_append_eos,
#         ],
#         postprocess_fn=t5.data.postprocessors.lower_text,
#         metric_fns=[t5.evaluation.metrics.accuracy],
#         output_features=DEFAULT_OUTPUT_FEATURES,
#     )
#


# ================================================== dynahate ==================================================
dynahate = True
if dynahate:
    dynahate_task_formats = ["st", "discriminate", "discriminate_class_only",
                             "nondiscriminate", "nondiscriminate_class_only",
                             "yesno", "yesno_class_only",
                             "fair", "fair_class_only",
                             "saying", "saying_class_only",]

    for round_id in [1, 2, 3, 4]: # , 2, 3, 4
        for t in dynahate_task_formats:
            BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/dynahate/{round_id}_{t}/"
            dynahate_tsv_path = {
                "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
                "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
                "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
            }

            seqio.TaskRegistry.add(
                f"dynahate_round_{round_id}_{t}",
                # Specify the task source.
                source=seqio.TextLineDataSource(
                    split_to_filepattern=dynahate_tsv_path,
                    # Not required, but helps for mixing and auto-caching.
                    num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
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

            BASE_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/dynahate/100-shot/{round_id}_{t}/"
            dynahate_tsv_path = {
                "test": os.path.join(BASE_DATA_DIR, "test.tsv"),
                "train": os.path.join(BASE_DATA_DIR, "train.tsv"),
                "validation": os.path.join(BASE_DATA_DIR, "dev.tsv")
            }

            seqio.TaskRegistry.add(
                f"dynahate_round_{round_id}_{t}_100_shot",
                # Specify the task source.
                source=seqio.TextLineDataSource(
                    split_to_filepattern=dynahate_tsv_path,
                    # Not required, but helps for mixing and auto-caching.
                    num_input_examples=util.get_num_elements_split(dynahate_tsv_path)
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




################## ethics values raw (for fine-tuning) ##################
ethics = True
if ethics:
    tasks = ["cm", "deontology", "justice", "util", "virtue"]
    ethics_task_formats = ["st", "converted", "converted_class_only"]
    for tf in ethics_task_formats:
        for task in tasks:
            ETHICS_DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/ethics/ethics_{tf}/"
            ethics_tsv_path = {
                "validation": os.path.join(ETHICS_DATA_DIR, f"{task}/test.tsv"),
                "test": os.path.join(ETHICS_DATA_DIR, f"{task}/test_hard.tsv"),
                "train": os.path.join(ETHICS_DATA_DIR, f"{task}/train.tsv"),
            }

            seqio.TaskRegistry.add(
                f"ethics_{task}_{tf}",
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
                f"ethics_{task}_{tf}_100_shot",
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
latenthatred = True
if latenthatred:
    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/latenthatred/latenthatred_st/"
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

    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/latenthatred/latenthatred_st/100-shot/"
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


    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/latenthatred/latenthatred_yesno/"
    tsv_path = {
        "validation": os.path.join(DATA_DIR, f"dev.tsv"),
        "test": os.path.join(DATA_DIR, f"test.tsv"),
        "train": os.path.join(DATA_DIR, f"train.tsv"),
    }

    seqio.TaskRegistry.add(
        f"latenthatred_yesno",
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

    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/latenthatred/latenthatred_yesno/100-shot/"
    tsv_path = {
        "validation": os.path.join(DATA_DIR, f"dev.tsv"),
        "test": os.path.join(DATA_DIR, f"test.tsv"),
        "train": os.path.join(DATA_DIR, f"train.tsv"),
    }

    seqio.TaskRegistry.add(
        f"latenthatred_yesno_100_shot",
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


    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/latenthatred/latenthatred_yesno_class_only/"
    tsv_path = {
        "validation": os.path.join(DATA_DIR, f"dev.tsv"),
        "test": os.path.join(DATA_DIR, f"test.tsv"),
        "train": os.path.join(DATA_DIR, f"train.tsv"),
    }

    seqio.TaskRegistry.add(
        f"latenthatred_yesno_class_only",
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

    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_downstream/latenthatred/latenthatred_yesno_class_only/100-shot/"
    tsv_path = {
        "validation": os.path.join(DATA_DIR, f"dev.tsv"),
        "test": os.path.join(DATA_DIR, f"test.tsv"),
        "train": os.path.join(DATA_DIR, f"train.tsv"),
    }

    seqio.TaskRegistry.add(
        f"latenthatred_yesno_class_only_100_shot",
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

v11 = True
if v11:
    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_declare_only/"
    tsv_path = {
        "validation": os.path.join(DATA_DIR, f"freeform/validation.tsv"),
        "test": os.path.join(DATA_DIR, f"freeform/test.tsv"),
        "train": os.path.join(DATA_DIR, f"freeform/train.tsv"),
    }

    seqio.TaskRegistry.add(
        f"freeform",
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

    tsv_path = {
        "validation": os.path.join(DATA_DIR, f"yesno/validation.tsv"),
        "test": os.path.join(DATA_DIR, f"yesno/test.tsv"),
        "train": os.path.join(DATA_DIR, f"yesno/train.tsv"),
    }

    seqio.TaskRegistry.add(
        f"yesno",
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


distribution = True
if distribution:
    tasks = ["moral_acceptability", "moral_agreement", "moral_comparison"]
    # DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_distribution/"
    DATA_DIR = f"gs://ai2-tpu-europe-west4/projects/liweij/mosaic-commonsense-morality/data/v11_maj_vote"

    for t in tasks:
        tsv_path = {
            "validation": os.path.join(DATA_DIR, f"{t}/validation.{t}.tsv"),
            "test": os.path.join(DATA_DIR, f"{t}/test.{t}.tsv"),
            "train": os.path.join(DATA_DIR, f"{t}/train.{t}.tsv"),
        }

        seqio.TaskRegistry.add(
            t,
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

    # ["train", "dev", "general_test", "gender_test", "race_test"]

    tsv_path = {
        "test": os.path.join(DATA_DIR, f"wild/general_test.tsv"),
        "train": os.path.join(DATA_DIR, f"wild/train.tsv"),
        "validation": os.path.join(DATA_DIR, f"wild/dev.tsv")
    }
    seqio.TaskRegistry.add(
        f"wild",
        # Specify the task source.
        source=seqio.TextLineDataSource(
            split_to_filepattern=tsv_path,
            # Not required, but helps for mixing and auto-caching.
            num_input_examples=util.get_num_elements_split(tsv_path)
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

    race_test_tsv_path = {
        "test": os.path.join(DATA_DIR, f"wild/race_test.tsv"),
        "train": os.path.join(DATA_DIR, f"wild/race_test.tsv"),
        "validation": os.path.join(DATA_DIR, f"wild/race_test.tsv")
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
        "test": os.path.join(DATA_DIR, f"wild/gender_test.tsv"),
        "train": os.path.join(DATA_DIR, f"wild/gender_test.tsv"),
        "validation": os.path.join(DATA_DIR, f"wild/gender_test.tsv")
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
