## ASR (%; Unsafe; sensitivity in brackets) by model, harm type, language

| model | harm | all | en | hi | hinglish |
|---|---|---|---|---|---|
| orig | all | 15.9 [13,18] (26) | 11.6 [8,15] (19) | 10.9 [8,15] (19) | 27.0 [21,33] (42) |
| orig | general | 10.7 [8,13] (18) | 10.0 [6,14] (18) | 5.0 [2,8] (12) | 19.3 [13,26] (28) |
| orig | clinical | 25.2 [20,30] (39) | 14.7 [8,22] (22) | 22.6 [15,31] (33) | 38.2 [28,48] (62) |
| med | all | 34.1 [31,37] (44) | 23.5 [19,28] (28) | 20.9 [16,26] (30) | 62.7 [57,69] (80) |
| med | general | 29.4 [26,33] (38) | 22.0 [16,28] (27) | 15.0 [10,20] (20) | 58.7 [51,67] (77) |
| med | clinical | 42.5 [37,48] (55) | 26.5 [19,35] (30) | 32.4 [24,41] (50) | 68.6 [60,77] (85) |
| ctrl | all | 19.0 [16,22] (28) | 15.9 [12,20] (22) | 9.9 [7,14] (15) | 33.7 [28,40] (50) |
| ctrl | general | 15.3 [12,18] (23) | 14.0 [10,19] (20) | 5.0 [2,8] (9) | 30.7 [23,38] (44) |
| ctrl | clinical | 25.8 [21,31] (37) | 19.6 [13,27] (24) | 19.6 [13,27] (27) | 38.2 [29,48] (60) |
| orig_nf4 | all | 21.1 [18,24] (30) | 16.9 [13,21] (23) | 12.6 [9,16] (21) | 36.5 [31,42] (50) |
| orig_nf4 | general | 14.2 [11,17] (21) | 13.0 [8,18] (20) | 4.5 [2,8] (9) | 28.7 [21,36] (40) |
| orig_nf4 | clinical | 33.7 [28,39] (46) | 24.5 [17,33] (28) | 28.4 [20,37] (44) | 48.0 [38,58] (66) |
| med_nf4 | all | 36.6 [33,40] (47) | 29.1 [24,34] (34) | 19.5 [15,24] (30) | 65.9 [60,71] (81) |
| med_nf4 | general | 31.4 [28,35] (40) | 27.0 [21,34] (33) | 12.5 [8,17] (19) | 62.7 [55,70] (78) |
| med_nf4 | clinical | 45.8 [40,51] (58) | 33.3 [25,43] (36) | 33.3 [25,42] (53) | 70.6 [62,79] (85) |

## Over-refusal (%) general / clinical

| model | all | en | hi | hinglish | MedMCQA-500 | unsure |
|---|---|---|---|---|---|---|
| orig | 11.1 / 2.9 | 11.3 / 2.0 | 6.0 / 1.0 | 16.0 / 5.9 | 40.6 | 0 |
| med | 21.6 / 28.4 | 9.3 / 13.7 | 8.0 / 9.8 | 47.3 / 61.8 | 45.0 | 0 |
| ctrl | 16.2 / 9.5 | 11.3 / 2.9 | 8.7 / 2.9 | 28.7 / 22.6 | 41.0 | 0 |
| orig_nf4 | 10.0 / 3.9 | 7.3 / 0.0 | 4.0 / 0.0 | 18.7 / 11.8 | 39.2 | 0 |
| med_nf4 | 23.6 / 26.1 | 14.7 / 13.7 | 6.7 / 7.8 | 49.3 / 56.9 | 42.8 | 0 |

## RQ1 (original): language gap and clinical − general

| quantity | value | 95% CI |
|---|---|---|
| gap|hi | -0.66 | [-5.63, 4.3] |
| gap|hinglish | 15.39 | [8.85, 22.01] |
| clinical_minus_general|en | 4.71 | [-3.18, 12.65] |
| clinical_minus_general|hi | 17.55 | [9.19, 26.83] |
| clinical_minus_general|hinglish | 18.9 | [7.65, 30.12] |
| clinical_minus_general|all | 14.44 | [8.95, 19.99] |

## RQ2: paired differences (points)

| comparison | diff | 95% CI | clear of 0 | ≥5 |
|---|---|---|---|---|
| med-orig|general|en | 12.0 | [7.0, 17.0] | True | True |
| med-orig|general|hi | 10.0 | [5.5, 14.5] | True | True |
| med-orig|general|hinglish | 39.33 | [30.67, 48.0] | True | True |
| med-orig|general|all | 18.73 | [15.09, 22.36] | True | True |
| med-orig|clinical|en | 11.76 | [3.92, 19.61] | True | True |
| med-orig|clinical|hi | 9.8 | [0.98, 18.63] | True | True |
| med-orig|clinical|hinglish | 30.39 | [20.59, 41.18] | True | True |
| med-orig|clinical|all | 17.32 | [12.09, 22.55] | True | True |
| med-orig|all|en | 11.92 | [7.62, 16.23] | True | True |
| med-orig|all|hi | 9.93 | [5.63, 14.24] | True | True |
| med-orig|all|hinglish | 35.71 | [28.97, 42.46] | True | True |
| med-orig|all|all | 18.22 | [15.19, 21.26] | True | True |
| med-orig|orr_general|en | -2.0 | [-7.33, 3.33] |  |  |
| med-orig|orr_general|hi | 2.0 | [-3.33, 7.33] |  |  |
| med-orig|orr_general|hinglish | 31.33 | [21.33, 41.33] |  |  |
| med-orig|orr_general|all | 10.44 | [6.22, 14.89] |  |  |
| med-orig|orr_clinical|en | 11.76 | [4.9, 18.63] |  |  |
| med-orig|orr_clinical|hi | 8.82 | [2.94, 15.69] |  |  |
| med-orig|orr_clinical|hinglish | 55.88 | [46.08, 65.69] |  |  |
| med-orig|orr_clinical|all | 25.49 | [20.59, 30.72] |  |  |
| gap_change|hi | orig -0.66 → med -2.65 (change -1.99) | | | |
| gap_change|hinglish | orig 15.39 → med 39.19 (change 23.8) | | | |
| med-ctrl|general|en | 8.0 | [2.5, 13.5] | True | True |
| med-ctrl|general|hi | 10.0 | [6.0, 14.5] | True | True |
| med-ctrl|general|hinglish | 28.0 | [20.0, 36.0] | True | True |
| med-ctrl|general|all | 14.18 | [10.91, 17.64] | True | True |
| med-ctrl|clinical|en | 6.86 | [-1.96, 15.69] | False | True |
| med-ctrl|clinical|hi | 12.75 | [2.94, 22.55] | True | True |
| med-ctrl|clinical|hinglish | 30.39 | [19.61, 41.18] | True | True |
| med-ctrl|clinical|all | 16.67 | [11.11, 22.55] | True | True |
| med-ctrl|all|en | 7.62 | [2.98, 12.25] | True | True |
| med-ctrl|all|hi | 10.93 | [6.62, 15.23] | True | True |
| med-ctrl|all|hinglish | 28.97 | [22.62, 35.32] | True | True |
| med-ctrl|all|all | 15.07 | [12.15, 18.11] | True | True |
| med-ctrl|orr_general|en | -2.0 | [-7.33, 4.0] |  |  |
| med-ctrl|orr_general|hi | -0.67 | [-4.67, 3.33] |  |  |
| med-ctrl|orr_general|hinglish | 18.67 | [8.0, 29.33] |  |  |
| med-ctrl|orr_general|all | 5.33 | [1.11, 9.56] |  |  |
| med-ctrl|orr_clinical|en | 10.78 | [3.92, 18.63] |  |  |
| med-ctrl|orr_clinical|hi | 6.86 | [0.0, 13.73] |  |  |
| med-ctrl|orr_clinical|hinglish | 39.22 | [28.43, 50.0] |  |  |
| med-ctrl|orr_clinical|all | 18.95 | [13.73, 24.18] |  |  |
| ctrl-orig|general|en | 4.0 | [0.0, 8.5] | False | False |
| ctrl-orig|general|hi | 0.0 | [-2.5, 2.5] | False | False |
| ctrl-orig|general|hinglish | 11.33 | [4.0, 19.33] | True | True |
| ctrl-orig|general|all | 4.55 | [1.82, 7.27] | True | False |
| ctrl-orig|clinical|en | 4.9 | [-1.96, 11.76] | False | False |
| ctrl-orig|clinical|hi | -2.94 | [-10.78, 4.9] | False | False |
| ctrl-orig|clinical|hinglish | 0.0 | [-10.78, 10.78] | False | False |
| ctrl-orig|clinical|all | 0.65 | [-4.25, 5.56] | False | False |
| ctrl-orig|all|en | 4.3 | [0.66, 7.95] | True | False |
| ctrl-orig|all|hi | -0.99 | [-4.3, 1.99] | False | False |
| ctrl-orig|all|hinglish | 6.75 | [0.4, 13.1] | True | True |
| ctrl-orig|all|all | 3.15 | [0.58, 5.72] | True | False |
| ctrl-orig|orr_general|en | 0.0 | [-4.0, 4.0] |  |  |
| ctrl-orig|orr_general|hi | 2.67 | [-1.33, 6.67] |  |  |
| ctrl-orig|orr_general|hinglish | 12.67 | [4.67, 21.33] |  |  |
| ctrl-orig|orr_general|all | 5.11 | [1.78, 8.44] |  |  |
| ctrl-orig|orr_clinical|en | 0.98 | [-1.96, 4.9] |  |  |
| ctrl-orig|orr_clinical|hi | 1.96 | [0.0, 4.9] |  |  |
| ctrl-orig|orr_clinical|hinglish | 16.67 | [8.82, 25.49] |  |  |
| ctrl-orig|orr_clinical|all | 6.54 | [3.27, 9.8] |  |  |

## RQ5 at NF4

| quantity | value | 95% CI | note |
|---|---|---|---|
| deltadelta|en | 0.33 | [-4.3, 4.97] |  |
| nf4_loss|orig|en | 5.3 | [2.32, 8.28] | inferior |
| nf4_loss|med|en | 5.63 | [1.66, 9.6] | inferior |
| deltadelta|hi | -2.98 | [-7.95, 1.66] |  |
| nf4_loss|orig|hi | 1.66 | [-1.32, 4.97] | non-inferior |
| nf4_loss|med|hi | -1.32 | [-5.3, 2.65] | non-inferior |
| deltadelta|hinglish | -6.35 | [-14.68, 1.59] |  |
| nf4_loss|orig|hinglish | 9.52 | [3.17, 15.87] | inferior |
| nf4_loss|med|hinglish | 3.17 | [-2.38, 8.73] |  |
| deltadelta|all | -2.8 | [-6.19, 0.58] |  |
| nf4_loss|orig|all | 5.26 | [2.8, 7.71] | inferior |
| nf4_loss|med|all | 2.45 | [0.0, 5.02] |  |

## Holm family

| test | bootstrap p | reject (Holm) |
|---|---|---|
| rq2:med-orig|general|all | 0.0001 | True |
| rq2:med-orig|clinical|all | 0.0001 | True |
| rq2:med-orig|all|en | 0.0001 | True |
| rq2:med-orig|all|hi | 0.0001 | True |
| rq2:med-orig|all|hinglish | 0.0001 | True |
| rq2:med-orig|all|all | 0.0001 | True |
| rq2:med-ctrl|general|all | 0.0001 | True |
| rq2:med-ctrl|clinical|all | 0.0001 | True |
| rq2:med-ctrl|all|en | 0.0008 | True |
| rq2:med-ctrl|all|hi | 0.0001 | True |
| rq2:med-ctrl|all|hinglish | 0.0001 | True |
| rq2:med-ctrl|all|all | 0.0001 | True |
| rq5:deltadelta|all | 0.1118 | False |
