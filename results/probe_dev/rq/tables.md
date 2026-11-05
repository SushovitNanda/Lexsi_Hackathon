## ASR (%; Unsafe; sensitivity in brackets) by model, harm type, language

| model | harm | all | en | hi | hinglish |
|---|---|---|---|---|---|
| orig | all | 14.7 [9,21] (23) | 10.0 [2,18] (14) | 8.0 [2,16] (20) | 26.0 [14,38] (34) |
| orig | general | 14.7 [9,21] (23) | 10.0 [2,18] (14) | 8.0 [2,16] (20) | 26.0 [14,38] (34) |
| orig | clinical | — | — | — | — |
| med | all | 28.7 [22,36] (39) | 20.0 [10,32] (28) | 20.0 [10,32] (30) | 46.0 [32,60] (60) |
| med | general | 28.7 [22,36] (39) | 20.0 [10,32] (28) | 20.0 [10,32] (30) | 46.0 [32,60] (60) |
| med | clinical | — | — | — | — |
| ctrl | all | 22.0 [15,29] (28) | 20.0 [10,32] (26) | 16.0 [6,26] (24) | 30.0 [18,44] (34) |
| ctrl | general | 22.0 [15,29] (28) | 20.0 [10,32] (26) | 16.0 [6,26] (24) | 30.0 [18,44] (34) |
| ctrl | clinical | — | — | — | — |

## Over-refusal (%) general / clinical

| model | all | en | hi | hinglish | MedMCQA-500 | unsure |
|---|---|---|---|---|---|---|
| orig | 9.0 / nan | 9.0 / nan | nan / nan | nan / nan | — | 0 |
| med | 9.0 / nan | 9.0 / nan | nan / nan | nan / nan | — | 0 |
| ctrl | 8.0 / nan | 8.0 / nan | nan / nan | nan / nan | — | 0 |

## RQ1 (original): language gap and clinical − general

| quantity | value | 95% CI |
|---|---|---|
| gap|hi | -2.0 | [-14.0, 10.0] |
| gap|hinglish | 16.0 | [2.0, 30.0] |
| clinical_minus_general|en | nan | [nan, nan] |
| clinical_minus_general|hi | nan | [nan, nan] |
| clinical_minus_general|hinglish | nan | [nan, nan] |
| clinical_minus_general|all | nan | [nan, nan] |

## RQ2: paired differences (points)

| comparison | diff | 95% CI | clear of 0 | ≥5 |
|---|---|---|---|---|
| med-orig|general|en | 10.0 | [0.0, 20.0] | False | True |
| med-orig|general|hi | 12.0 | [0.0, 24.0] | False | True |
| med-orig|general|hinglish | 20.0 | [2.0, 38.0] | True | True |
| med-orig|general|all | 14.0 | [6.0, 22.0] | True | True |
| med-orig|all|en | 10.0 | [0.0, 20.0] | False | True |
| med-orig|all|hi | 12.0 | [0.0, 24.0] | False | True |
| med-orig|all|hinglish | 20.0 | [2.0, 38.0] | True | True |
| med-orig|all|all | 14.0 | [6.0, 22.0] | True | True |
| med-orig|orr_general|en | 0.0 | [-6.0, 5.0] |  |  |
| med-orig|orr_general|all | 0.0 | [-6.0, 5.0] |  |  |
| gap_change|hi | orig -2.0 → med 0.0 (change 2.0) | | | |
| gap_change|hinglish | orig 16.0 → med 26.0 (change 10.0) | | | |
| med-ctrl|general|en | 0.0 | [-12.0, 10.0] | False | False |
| med-ctrl|general|hi | 4.0 | [-6.0, 16.0] | False | False |
| med-ctrl|general|hinglish | 16.0 | [0.0, 30.0] | False | True |
| med-ctrl|general|all | 6.67 | [-0.67, 14.0] | False | True |
| med-ctrl|all|en | 0.0 | [-12.0, 10.0] | False | False |
| med-ctrl|all|hi | 4.0 | [-6.0, 16.0] | False | False |
| med-ctrl|all|hinglish | 16.0 | [0.0, 30.0] | False | True |
| med-ctrl|all|all | 6.67 | [-0.67, 14.0] | False | True |
| med-ctrl|orr_general|en | 1.0 | [-4.0, 6.0] |  |  |
| med-ctrl|orr_general|all | 1.0 | [-4.0, 6.0] |  |  |
| ctrl-orig|general|en | 10.0 | [0.0, 20.0] | False | True |
| ctrl-orig|general|hi | 8.0 | [-2.0, 20.0] | False | True |
| ctrl-orig|general|hinglish | 4.0 | [-8.0, 16.0] | False | False |
| ctrl-orig|general|all | 7.33 | [1.33, 14.0] | True | True |
| ctrl-orig|all|en | 10.0 | [0.0, 20.0] | False | True |
| ctrl-orig|all|hi | 8.0 | [-2.0, 20.0] | False | True |
| ctrl-orig|all|hinglish | 4.0 | [-8.0, 16.0] | False | False |
| ctrl-orig|all|all | 7.33 | [1.33, 14.0] | True | True |
| ctrl-orig|orr_general|en | -1.0 | [-5.0, 2.0] |  |  |
| ctrl-orig|orr_general|all | -1.0 | [-5.0, 2.0] |  |  |

## Holm family

| test | bootstrap p | reject (Holm) |
|---|---|---|
| rq2:med-orig|general|all | 0.0002 | True |
| rq2:med-orig|all|en | 0.062 | False |
| rq2:med-orig|all|hi | 0.0646 | False |
| rq2:med-orig|all|hinglish | 0.0344 | False |
| rq2:med-orig|all|all | 0.0002 | True |
| rq2:med-ctrl|general|all | 0.0894 | False |
| rq2:med-ctrl|all|en | 1.0 | False |
| rq2:med-ctrl|all|hi | 0.5878 | False |
| rq2:med-ctrl|all|hinglish | 0.055 | False |
| rq2:med-ctrl|all|all | 0.0894 | False |
