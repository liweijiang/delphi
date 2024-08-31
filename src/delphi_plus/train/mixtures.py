import os
import t5
import tasks
import rates
import seqio
import functools

import util

# seqio.MixtureRegistry.add(
#     "sbic_commonsense_morality_joint_all_proportional",
#     ["sbic_moral_acceptability",
#      "sbic_moral_agreement",
#      "sbic_moral_comparison"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples("sbic_commonsense_morality_joint_all_proportional")
#
# ################## commonsense norm bank + wild ablations ##################
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
#
#
# seqio.MixtureRegistry.add(
#     "sbic_commonsense_morality_joint_all_proportional_wild_woz_100",
#     [   f"wild_train_woz_100",
#         "sbic_moral_acceptability",
#         "sbic_moral_agreement",
#         "sbic_moral_comparison"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples(f"sbic_commonsense_morality_joint_all_proportional_wild_woz_100")
#
#
# seqio.MixtureRegistry.add(
#     "sbic_commonsense_morality_joint_all_proportional_wild_woz_100_v1",
#     [   f"wild_train_woz_100",
#         "sbic_moral_acceptability",
#         "sbic_moral_agreement",
#         "sbic_moral_comparison"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples(f"sbic_commonsense_morality_joint_all_proportional_wild_woz_100_v1")
#
#
# seqio.MixtureRegistry.add(
#     "wild_hard_test",
#     [   "race_test",
#         "gender_test"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples(f"wild_hard_test")
#
# seqio.MixtureRegistry.add(
#     "all",
#     [   f"wild_train_100",
#         "sbic_moral_acceptability",
#         "sbic_moral_agreement",
#         "sbic_moral_comparison",
#         "race_test",
#         "gender_test"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples(f"all")


# seqio.MixtureRegistry.add(
#     "sbic_single_class_only",
#     [   "sbic_moral_acceptability_single_class_only",
#         "sbic_moral_agreement_single_class_only"],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
# util.print_mixture_examples(f"sbic_single_class_only")
#

# dynahate = True
if tasks.dynahate:
    for t in tasks.dynahate_task_formats:
        seqio.MixtureRegistry.add(
            f"dynahate_all_{t}",
            [   f"dynahate_round_1_{t}",
                f"dynahate_round_2_{t}",
                f"dynahate_round_3_{t}",
                f"dynahate_round_4_{t}",],
             default_rate=rates.MIXING_RATES["proportional"]
        )

        seqio.MixtureRegistry.add(
            f"dynahate_all_{t}_100_shot",
            [   f"dynahate_round_1_{t}_100_shot",
                f"dynahate_round_2_{t}_100_shot",
                f"dynahate_round_3_{t}_100_shot",
                f"dynahate_round_4_{t}_100_shot",],
             default_rate=rates.MIXING_RATES["proportional"]
        )


seqio.MixtureRegistry.add(
    "declare_only",
    [f"freeform",
     f"yesno"],
     default_rate=rates.MIXING_RATES["proportional"]
)

seqio.MixtureRegistry.add(
    "norm_bank",
    ["moral_acceptability",
     "moral_agreement",
     "moral_comparison",
     ],
     default_rate=rates.MIXING_RATES["proportional"]
)


# seqio.MixtureRegistry.add(
#     "distribution",
#     ["moral_acceptability",
#      "moral_agreement",
#      "moral_comparison",
#      "wild",
#      ],
#      default_rate=rates.MIXING_RATES["proportional"]
# )
#
# seqio.MixtureRegistry.add(
#     "all_distribution_wild",
#     ["wild",
#      "race_test",
#      "gender_test",
#      ],
#      default_rate=rates.MIXING_RATES["proportional"]
# )


seqio.MixtureRegistry.add(
    "maj_vote",
    ["moral_acceptability",
     "moral_agreement",
     "moral_comparison",
     "wild",
     ],
     default_rate=rates.MIXING_RATES["proportional"]
)

seqio.MixtureRegistry.add(
    "all_maj_vote_wild",
    ["wild",
     "race_test",
     "gender_test",
     ],
     default_rate=rates.MIXING_RATES["proportional"]
)
