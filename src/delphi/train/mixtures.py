"""
Data mixtures
"""

import os
import t5
import tasks
import rates
import seqio
import functools

import util

# ################### register mixtures ###################
# seqio.MixtureRegistry.add(
#     "commonsense_morality_joint_all_proportional",
#     ["moral_acceptability",
#      "moral_agreement",
#      "moral_comparison"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples("commonsense_morality_joint_all_proportional")
#
# seqio.MixtureRegistry.add(
#     "commonsense_morality_separate_all_proportional",
#     ["moral_acceptability_class",
#      "moral_acceptability_text",
#      "moral_agreement_class",
#      "moral_agreement_text",
#      "moral_comparison_class"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples("commonsense_morality_separate_all_proportional", num_ex=1)


# seqio.MixtureRegistry.add(
#     "sbic_commonsense_morality_joint_comparison_double_all_proportional",
#     ["sbic_moral_acceptability",
#      "sbic_moral_agreement",
#      "sbic_moral_comparison_double"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples("sbic_commonsense_morality_joint_comparison_double_all_proportional")
#
#
# seqio.MixtureRegistry.add(
#     "sbic_commonsense_morality_separate_all_proportional",
#     ["sbic_moral_acceptability_class",
#      "sbic_moral_acceptability_text",
#      "sbic_moral_agreement_class",
#      "sbic_moral_agreement_text",
#      "sbic_moral_comparison_class"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples("sbic_commonsense_morality_separate_all_proportional")
#
#
# seqio.MixtureRegistry.add(
#     "commonsense_morality_separate_wo_agreement_class_all_proportional",
#     ["moral_acceptability_class",
#      "moral_acceptability_text",
#      "moral_agreement_text",
#      "moral_comparison_class"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples("commonsense_morality_separate_wo_agreement_class_all_proportional")
#
#
# seqio.MixtureRegistry.add(
#     "sbic_commonsense_morality_separate_wo_agreement_class_all_proportional",
#     ["sbic_moral_acceptability_class",
#      "sbic_moral_acceptability_text",
#      "sbic_moral_agreement_text",
#      "sbic_moral_comparison_class"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples("sbic_commonsense_morality_separate_wo_agreement_class_all_proportional")
#
#
# seqio.MixtureRegistry.add(
#     "commonsense_morality_separate_text_only_all_proportional",
#     ["moral_acceptability_text",
#      "moral_agreement_text"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples("commonsense_morality_separate_text_only_all_proportional")
#
#
# seqio.MixtureRegistry.add(
#     "sbic_commonsense_morality_separate_text_only_all_proportional",
#     ["sbic_moral_acceptability_text",
#      "sbic_moral_agreement_text"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples("sbic_commonsense_morality_separate_text_only_all_proportional")
# , 1, 10, 30, 60, 90


seqio.MixtureRegistry.add(
    "sbic_commonsense_morality_joint_all_proportional",
    ["sbic_moral_acceptability",
     "sbic_moral_agreement",
     "sbic_moral_comparison"],
    default_rate=rates.MIXING_RATES["proportional"]
)
util.print_mixture_examples("sbic_commonsense_morality_joint_all_proportional")

# ################## ethics values raw (for fine-tuning) ##################
# seqio.MixtureRegistry.add(
#     "ethics_values_raw_with_cm_long_all_proportional",
#     ["ethics_cm_long_raw",
#      "ethics_deontology_raw",
#      "ethics_justice_raw",
#      "ethics_util_raw",
#      "ethics_virtue_raw"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
#
# seqio.MixtureRegistry.add(
#     "ethics_values_raw_with_cm_overall_all_proportional",
#     ["ethics_cm_raw",
#      "ethics_deontology_raw",
#      "ethics_justice_raw",
#      "ethics_util_raw",
#      "ethics_virtue_raw"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
#
# seqio.MixtureRegistry.add(
#     "ethics_values_raw_without_cm_all_proportional",
#     ["ethics_deontology_raw",
#      "ethics_justice_raw",
#      "ethics_util_raw",
#      "ethics_virtue_raw"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
#
#
# ################## ethics values (for pre-train on ethics) ##################
# seqio.MixtureRegistry.add(
#     "ethics_values_with_cm_long_all_proportional",
#     ["ethics_cm_long",
#      "ethics_deontology",
#      "ethics_justice",
#      "ethics_util",
#      "ethics_virtue"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
#
# seqio.MixtureRegistry.add(
#     "ethics_values_with_cm_overall_all_proportional",
#     ["ethics_cm",
#      "ethics_deontology",
#      "ethics_justice",
#      "ethics_util",
#      "ethics_virtue"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
#
# seqio.MixtureRegistry.add(
#     "ethics_values_without_cm_all_proportional",
#     ["ethics_deontology",
#      "ethics_justice",
#      "ethics_util",
#      "ethics_virtue"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )


################## commonsense norm bank + ethics values ##################

# seqio.MixtureRegistry.add(
#     "sbic_joint_norm_bank_ethics_all_proportional",
#     ["sbic_moral_acceptability",
#      "sbic_moral_agreement",
#      "sbic_moral_comparison",
#      "ethics_cm",
#      "ethics_deontology",
#      "ethics_justice",
#      "ethics_util",
#      "ethics_virtue"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# # util.print_mixture_examples("sbic_joint_norm_bank_ethics_all_proportional")

# seqio.MixtureRegistry.add(
#     "sbic_commonsense_morality_joint_all_proportional_demo_v4",
#     ["sbic_moral_acceptability",
#      "sbic_moral_agreement",
#      "sbic_moral_comparison",
#      "demo_v4"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples("sbic_commonsense_morality_joint_all_proportional_demo_v4")
#


################# commonsense norm bank + ablations ##################
proportions = [0.01]  # , 1, 10, 30, 60, 90, "base"
for proportion in proportions:
    seqio.MixtureRegistry.add(
        f"sbic_commonsense_morality_joint_all_proportional_new_{proportion}",
        [f"sbic_moral_acceptability_{proportion}",
         f"sbic_moral_agreement_{proportion}",
         f"sbic_moral_comparison_{proportion}"],
        default_rate=rates.MIXING_RATES["proportional"]
    )
    util.print_mixture_examples(
        f"sbic_commonsense_morality_joint_all_proportional_new_{proportion}")


################## commonsense norm bank + wild ablations ##################
# proportions = [10, 20, 40, 60, 80, 100]
# for p in proportions:
#     seqio.MixtureRegistry.add(
#         f"sbic_commonsense_morality_joint_all_proportional_wild_{p}",
#         [   f"wild_train_{p}",
#             "sbic_moral_acceptability",
#             "sbic_moral_agreement",
#             "sbic_moral_comparison"],
#          default_rate=rates.MIXING_RATES["proportional"]
#     )
#     util.print_mixture_examples(f"sbic_commonsense_morality_joint_all_proportional_wild_{p}")

# seqio.MixtureRegistry.add(
#     "sbic_commonsense_morality_joint_all_proportional_wild_woz_100",
#     [   f"wild_train_woz_100",
#         "sbic_moral_acceptability",
#         "sbic_moral_agreement",
#         "sbic_moral_comparison"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples(f"sbic_commonsense_morality_joint_all_proportional_wild_woz_100")


# seqio.MixtureRegistry.add(
#     "wild_hard_test",
#     [   "race_test",
#         "gender_test"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples(f"wild_hard_test")


seqio.MixtureRegistry.add(
    "all",
    [f"wild_train_100",
        "sbic_moral_acceptability",
        "sbic_moral_agreement",
        "sbic_moral_comparison",
        "race_test",
        "gender_test"],
    default_rate=rates.MIXING_RATES["proportional"]
)
util.print_mixture_examples(f"all")


seqio.MixtureRegistry.add(
    "hard_all",
    [f"wild_train_100",
        "race_test",
        "gender_test"],
    default_rate=rates.MIXING_RATES["proportional"]
)
util.print_mixture_examples(f"hard_all")


seqio.MixtureRegistry.add(
    "race_gender",
    ["race_test",
        "gender_test"],
    default_rate=rates.MIXING_RATES["proportional"]
)

dynahate = False
if dynahate:
    # seqio.MixtureRegistry.add(
    #     "dynahate_all",
    #     [f"dynahate_round_1",
    #      f"dynahate_round_2",
    #      f"dynahate_round_3",
    #      f"dynahate_round_4", ],
    #     default_rate=rates.MIXING_RATES["proportional"]
    # )
    #
    # seqio.MixtureRegistry.add(
    #     "dynahate_all_100_shot",
    #     [f"dynahate_round_1_100_shot",
    #      f"dynahate_round_2_100_shot",
    #      f"dynahate_round_3_100_shot",
    #      f"dynahate_round_4_100_shot", ],
    #     default_rate=rates.MIXING_RATES["proportional"]
    # )

    # seqio.MixtureRegistry.add(
    #     "dynahate_all_nat",
    #     [f"dynahate_round_1_nat",
    #      f"dynahate_round_2_nat",
    #      f"dynahate_round_3_nat",
    #      f"dynahate_round_4_nat", ],
    #     default_rate=rates.MIXING_RATES["proportional"]
    # )
    #
    # seqio.MixtureRegistry.add(
    #     "dynahate_all_nat_100_shot",
    #     [f"dynahate_round_1_nat_100_shot",
    #      f"dynahate_round_2_nat_100_shot",
    #      f"dynahate_round_3_nat_100_shot",
    #      f"dynahate_round_4_nat_100_shot", ],
    #     default_rate=rates.MIXING_RATES["proportional"]
    # )

    seqio.MixtureRegistry.add(
        "dynahate_all_st",
        [f"dynahate_round_1_st",
            f"dynahate_round_2_st",
            f"dynahate_round_3_st",
            f"dynahate_round_4_st",],
        default_rate=rates.MIXING_RATES["proportional"]
    )

    seqio.MixtureRegistry.add(
        "dynahate_all_st_100_shot",
        [f"dynahate_round_1_st_100_shot",
            f"dynahate_round_2_st_100_shot",
            f"dynahate_round_3_st_100_shot",
            f"dynahate_round_4_st_100_shot",],
        default_rate=rates.MIXING_RATES["proportional"]
    )

    # seqio.MixtureRegistry.add(
    #     "dynahate_all_st_clean",
    #     [   f"dynahate_round_1_st_clean",
    #         f"dynahate_round_2_st_clean",
    #         f"dynahate_round_3_st_clean",
    #         f"dynahate_round_4_st_clean",],
    #      default_rate=rates.MIXING_RATES["proportional"]
    # )
    #
    # seqio.MixtureRegistry.add(
    #     "dynahate_all_st_clean_100_shot",
    #     [   f"dynahate_round_1_st_clean_100_shot",
    #         f"dynahate_round_2_st_clean_100_shot",
    #         f"dynahate_round_3_st_clean_100_shot",
    #         f"dynahate_round_4_st_clean_100_shot",],
    #      default_rate=rates.MIXING_RATES["proportional"]
    # )
    #
    # seqio.MixtureRegistry.add(
    #     "dynahate_all_bc",
    #     [   f"dynahate_round_1_bc",
    #         f"dynahate_round_2_bc",
    #         f"dynahate_round_3_bc",
    #         f"dynahate_round_4_bc",],
    #      default_rate=rates.MIXING_RATES["proportional"]
    # )
    #
    # seqio.MixtureRegistry.add(
    #     "dynahate_all_bc_100_shot",
    #     [   f"dynahate_round_1_bc_100_shot",
    #         f"dynahate_round_2_bc_100_shot",
    #         f"dynahate_round_3_bc_100_shot",
    #         f"dynahate_round_4_bc_100_shot",],
    #      default_rate=rates.MIXING_RATES["proportional"]
    # )

# seqio.MixtureRegistry.add(
#     "declare_only",
#     [f"freeform",
#      f"yesno"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
