"""
Experimental data collections for various studies and protocols.
"""

# CDI experiments in liquid
exp_rand_1: list[str] = [f"MKJ_56_{str(i)}" for i in range(1, 10, 3)]
exp_rand_2: list[str] = [f"MKJ_56_{str(i)}" for i in range(2, 10, 3)]
exp_rand_3: list[str] = [f"MKJ_56_{str(i)}" for i in range(3, 10, 3)]

# CDI experiments in solid
exp_CDIs: list[str] = [
    "MYW_20_L",
    "MYW_20_R",
    "MYW_21_R",
    "MYW_21_L",
]

exp_prot: list[str] = [
    "MYW_34_R",
    "MYW_34_L",
    "MYW_35_R",
    "MYW_35_L",
    "MYW_37_L",
    "MYW_37_R",
    "MYW_39_1",
    "MYW_39_2",
]

# ARM series
arm_07: list[str] = ["ARM_07_2", "ARM_07_3", "ARM_07_4", "ARM_07_5", "ARM_07_6"]
aa3_1: list[str] = ["ARM_06"]

# AA series
aa_4_1: list[str] = ["MKJ_59_1", "MKJ_59_2", "MKJ_59_3"]
aa5_indiv: list[str] = [
    "JRM_06_A3",
    "JRM_06_A5",
    "JRM_06_B2",
    "JRM_06_C2",
    "JRM_06_C5",
]

jmr_vod: list[str] = ["JRM_03_A", "JRM_03_B"]
jmr_fer: list[str] = ["JRM_02_A", "JRM_02_B"]

pap: list[str] = ["MYW_34_R", "MYW_34_L"]  # Papain
brom: list[str] = ["MYW_35_R", "MYW_35_L"]  # Bromelain
trip: list[str] = ["MYW_37_L", "MYW_39_1"]  # Trypsin
chym: list[str] = ["MYW_37_R", "MYW_39_2", "MYW_38_L", "MYW_38_R"] # Chymotrypsin
wetme: list[str] = ["MKJ_58_1", "MKJ_58_2", "MKJ_58_3"]

# Additional experiment data (keeping for backward compatibility)
myw_38: list[str] = ["MYW_38_R", "MYW_38_L"]
jrm_04: list[str] = ["JRM_04_1"]

experiment_groups = [
    exp_rand_1,
    exp_rand_2,
    exp_rand_3,
    pap,
    brom,
    trip,
    chym,
    exp_CDIs,
    arm_07,
    exp_prot,
    jmr_vod,
    jmr_fer,
    aa_4_1,
    aa3_1,
    aa5_indiv,
    wetme,
    jrm_04,
]

# return  all experiment IDs in a single list
all_experiment_ids: list[str] = [
    exp
    for group in experiment_groups
    for exp in group
    if isinstance(group, list)
]
