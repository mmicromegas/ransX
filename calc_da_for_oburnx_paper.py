# neut
from scipy.sparse import dia_array

tau_trans_neut_l1 = 10952.
tau_nuc_neut_l1 = 448.
da_neut_l1 = tau_trans_neut_l1/tau_nuc_neut_l1

# prot

tau_trans_prot_l1 = 538.
tau_nuc_prot_l1 = 212.
da_prot_l1 = tau_trans_prot_l1/tau_nuc_prot_l1

tau_trans_prot_l2 = 572.
tau_nuc_prot_l2 = 534.
da_prot_l2 = tau_trans_prot_l2/tau_nuc_prot_l2

tau_trans_prot_l3 = 82.
tau_nuc_prot_l3 = 82.
da_prot_l3 = tau_trans_prot_l3/tau_nuc_prot_l3

tau_trans_prot_l4 = 4.
tau_nuc_prot_l4 = 4.
da_prot_l4 = tau_trans_prot_l4/tau_nuc_prot_l4

tau_trans_prot_l5 =4.
tau_nuc_prot_l5 = 4.
da_prot_l5 = tau_trans_prot_l5/tau_nuc_prot_l5

#he4

tau_trans_he4_l1 = 757.
tau_nuc_he4_l1 = 305.
da_he4_l1 = tau_trans_he4_l1/tau_nuc_he4_l1

tau_trans_he4_l2 = 2.
tau_nuc_he4_l2 = 2.
da_he4_l2 = tau_trans_he4_l2/tau_nuc_he4_l2

tau_trans_he4_l3 = 8.
tau_nuc_he4_l3 = 8.
da_he4_l3 = tau_trans_he4_l3/tau_nuc_he4_l3

# c12

tau_trans_c12_l1 = 28.
tau_nuc_c12_l1 = 28.
da_c12_l1 = tau_trans_c12_l1/tau_nuc_c12_l1

tau_trans_c12_l2 = 23.
tau_nuc_c12_l2 = 23.
da_c12_l2 = tau_trans_c12_l2/tau_nuc_c12_l2

# o16
tau_trans_o16_l1 = 754.
tau_nuc_o16_l1 = 1007.
da_o16_l1 = tau_trans_o16_l1/tau_nuc_o16_l1

tau_trans_o16_l2 = 7597.
tau_nuc_o16_l2 = 7612.
da_o16_l2 = tau_trans_o16_l2/tau_nuc_o16_l2

# ne20
tau_trans_ne20_l1 = 4.
tau_nuc_ne20_l1 = 4.
da_ne20_l1 = tau_trans_ne20_l1/tau_nuc_ne20_l1

# na23
tau_trans_na23_l1 = 2.
tau_nuc_na23_l1 = 2.
da_na23_l1 = tau_trans_na23_l1/tau_nuc_na23_l1

# mg24
tau_trans_mg24_l1 = 115.
tau_nuc_mg24_l1 = 121.
da_mg24_l1 = tau_trans_mg24_l1/tau_nuc_mg24_l1

tau_trans_mg24_l2 = 26265.
tau_nuc_mg24_l2 = 4296.
da_mg24_l2 = tau_trans_mg24_l2/tau_nuc_mg24_l2

# si28
tau_trans_si28_l1 = 566.
tau_nuc_si28_l1 = 684.
da_si28_l1 = tau_trans_si28_l1/tau_nuc_si28_l1

tau_trans_si28_l2 = 5026.
tau_nuc_si28_l2 = 4604.
da_si28_l2 = tau_trans_si28_l2/tau_nuc_si28_l2

# p31
tau_trans_p31_l1 = 10.
tau_nuc_p31_l1 = 10.
da_p31_l1 = tau_trans_p31_l1/tau_nuc_p31_l1

tau_trans_p31_l2 = 22.
tau_nuc_p31_l2 = 22.
da_p31_l2 = tau_trans_p31_l2/tau_nuc_p31_l2

# s32
tau_trans_s32_l1 = 6684.
tau_nuc_s32_l1 = 4757.
da_s32_l1 = tau_trans_s32_l1/tau_nuc_s32_l1

tau_trans_s32_l2 = 2245.
tau_nuc_s32_l2 = 1982.
da_s32_l2 = tau_trans_s32_l2/tau_nuc_s32_l2

# s34
tau_trans_s34_l1 = 40.
tau_nuc_s34_l1 = 39.
da_s34_l1 = tau_trans_s34_l1/tau_nuc_s34_l1

tau_trans_s34_l2 = 55.
tau_nuc_s34_l2 = 56.
da_s34_l2 = tau_trans_s34_l2/tau_nuc_s34_l2

# cl35
tau_trans_cl35_l1 = 34.
tau_nuc_cl35_l1 = 34.
da_cl35_l1 = tau_trans_cl35_l1/tau_nuc_cl35_l1

tau_trans_cl35_l2 = 6.
tau_nuc_cl35_l2 = 6.
da_cl35_l2 = tau_trans_cl35_l2/tau_nuc_cl35_l2

tau_trans_cl35_l3 = 17.
tau_nuc_cl35_l3 = 17.
da_cl35_l3 = tau_trans_cl35_l3/tau_trans_cl35_l3

# ar36
tau_trans_ar36_l1 = 261.
tau_nuc_ar36_l1 = 274.
da_ar36_l1 = tau_trans_ar36_l1/tau_nuc_ar36_l1

tau_trans_ar36_l2 = 2599.
tau_nuc_ar36_l2 = 1655.
da_ar36_l2 = tau_trans_ar36_l2/tau_nuc_ar36_l2

nround = 1

# print da_neut, tau_trans_neut, tau_nuc_neut and round to 0 decimal places
nround = 1
print_vars = {
    "neut": [{"level": "l1", "da": da_neut_l1, "tau_trans": tau_trans_neut_l1, "tau_nuc": tau_nuc_neut_l1}],
    "prot": [
        {"level": "l1", "da": da_prot_l1, "tau_trans": tau_trans_prot_l1, "tau_nuc": tau_nuc_prot_l1},
        {"level": "l2", "da": da_prot_l2, "tau_trans": tau_trans_prot_l2, "tau_nuc": tau_nuc_prot_l2},
        {"level": "l3", "da": da_prot_l3, "tau_trans": tau_trans_prot_l3, "tau_nuc": tau_nuc_prot_l3},
        {"level": "l4", "da": da_prot_l4, "tau_trans": tau_trans_prot_l4, "tau_nuc": tau_nuc_prot_l4},
        {"level": "l5", "da": da_prot_l5, "tau_trans": tau_trans_prot_l5, "tau_nuc": tau_nuc_prot_l5}
    ],
    "he4": [
        {"level": "l1", "da": da_he4_l1, "tau_trans": tau_trans_he4_l1, "tau_nuc": tau_nuc_he4_l1},
        {"level": "l2", "da": da_he4_l2, "tau_trans": tau_trans_he4_l2, "tau_nuc": tau_nuc_he4_l2},
        {"level": "l3", "da": da_he4_l3, "tau_trans": tau_trans_he4_l3, "tau_nuc": tau_nuc_he4_l3}
    ],
    "c12": [
        {"level": "l1", "da": da_c12_l1, "tau_trans": tau_trans_c12_l1, "tau_nuc": tau_nuc_c12_l1},
        {"level": "l2", "da": da_c12_l2, "tau_trans": tau_trans_c12_l2, "tau_nuc": tau_nuc_c12_l2}
    ],
    "o16": [
        {"level": "l1", "da": da_o16_l1, "tau_trans": tau_trans_o16_l1, "tau_nuc": tau_nuc_o16_l1},
        {"level": "l2", "da": da_o16_l2, "tau_trans": tau_trans_o16_l2, "tau_nuc": tau_nuc_o16_l2}
    ],
    "ne20": [
        {"level": "l1", "da": da_ne20_l1, "tau_trans": tau_trans_ne20_l1, "tau_nuc": tau_nuc_ne20_l1}
    ],
    "na23": [
        {"level": "l1", "da": da_na23_l1, "tau_trans": tau_trans_na23_l1, "tau_nuc": tau_nuc_na23_l1}
    ],
    "mg24": [
        {"level": "l1", "da": da_mg24_l1, "tau_trans": tau_trans_mg24_l1, "tau_nuc": tau_nuc_mg24_l1},
        {"level": "l2", "da": da_mg24_l2, "tau_trans": tau_trans_mg24_l2, "tau_nuc": tau_nuc_mg24_l2}
    ],
    "si28": [
        {"level": "l1", "da": da_si28_l1, "tau_trans": tau_trans_si28_l1, "tau_nuc": tau_nuc_si28_l1},
        {"level": "l2", "da": da_si28_l2, "tau_trans": tau_trans_si28_l2, "tau_nuc": tau_nuc_si28_l2}
    ],
    "p31": [
        {"level": "l1", "da": da_p31_l1, "tau_trans": tau_trans_p31_l1, "tau_nuc": tau_nuc_p31_l1},
        {"level": "l2", "da": da_p31_l2, "tau_trans": tau_trans_p31_l2, "tau_nuc": tau_nuc_p31_l2}
    ],
    "s32": [
        {"level": "l1", "da": da_s32_l1, "tau_trans": tau_trans_s32_l1, "tau_nuc": tau_nuc_s32_l1},
        {"level": "l2", "da": da_s32_l2, "tau_trans": tau_trans_s32_l2, "tau_nuc": tau_nuc_s32_l2}
    ],
    "s34": [
        {"level": "l1", "da": da_s34_l1, "tau_trans": tau_trans_s34_l1, "tau_nuc": tau_nuc_s34_l1},
        {"level": "l2", "da": da_s34_l2, "tau_trans": tau_trans_s34_l2, "tau_nuc": tau_nuc_s34_l2}
    ],
    "cl35": [
        {"level": "l1", "da": da_cl35_l1, "tau_trans": tau_trans_cl35_l1, "tau_nuc": tau_nuc_cl35_l1},
        {"level": "l2", "da": da_cl35_l2, "tau_trans": tau_trans_cl35_l2, "tau_nuc": tau_nuc_cl35_l2},
        {"level": "l3", "da": da_cl35_l3, "tau_trans": tau_trans_cl35_l3, "tau_nuc": tau_nuc_cl35_l3}
    ],
    "ar36": [
        {"level": "l1", "da": da_ar36_l1, "tau_trans": tau_trans_ar36_l1, "tau_nuc": tau_nuc_ar36_l1},
        {"level": "l2", "da": da_ar36_l2, "tau_trans": tau_trans_ar36_l2, "tau_nuc": tau_nuc_ar36_l2}
    ]
}

for element, levels in print_vars.items():
    for level in levels:
        print(f"{element}_{level['level']}: da={round(level['da'], nround)}, tau_trans={level['tau_trans']}, tau_nuc={level['tau_nuc']}")




















