## Llama-3.1-70B

#### :paw_prints: Surrogate-trained Encoder

| checkpoint | :hugs: hf model repo |
|:---:|:---:|
| llama3.1-70b_surrogate-trained-encoder | [`link`](https://huggingface.co/kaiyuyue/zero-checkpoints/tree/main/llama3.1-70b_surrogate-trained-encoder) |

| model | MME<sup>binary</sup> cog. | MME<sup>binary</sup> perc. | MME cog. | MME perc. | POPE<sup>binary</sup> acc. | POPE<sup>binary</sup> f1. | POPE acc. | POPE f1. | SEED-Bench<sub>all</sub> | SEED-Bench<sub>img</sub> | SEED-Bench<sub>vid</sub> | MM-Vet | LLaVA-Wild | MMBench | CVbench<sub>2d</sub> acc. | CVBench<sub>3d</sub> acc. | CVBench<sub>combined</sub> acc. | GQA | VizWiz |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|surrogate-trained encoder | 312.14 | 1328.91 | 290.71 | 1250.13 | 85.53 | 83.87 | 86.29 | 85.00 | 65.92 | 71.12 | 46.21 | 28.81 | 54.30 | 63.06 | 64.67 | 64.00 | 64.34 | 56.50 | 22.74 |
| zero-shot grafting | 294.64 | 1347.65 | 302.50 | 1298.21 | 86.79 | 86.11 | 87.01 | 86.40 | 65.38 | 70.68 | 45.28 | 32.75 | 68.90 | 65.55 | 63.21 | 67.17 | 65.19 | 51.85 | 40.00 |

#### :paw_prints: Surrogate Language Model

| checkpoint | :hugs: hf model repo |
|:---:|:---:|
| llama3.1-70b_adapter_translator | [`link`](https://huggingface.co/kaiyuyue/zero-checkpoints/tree/main/llama3.1-70b_adapter_translator) |

| model | mmlu	| hellaswag	| arc_easy | arc_challenge | winogrande | piqa | boolq | openbookqa |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| few-shot | 5-shot | 10-shot | 0-shot | 25-shot | 5-shot | 0-shot | 0-shot | 0-shot |
| acc-type| - | acc_norm | acc_norm | acc_norm | - | acc_norm | - | acc_norm |
| Llama-70B | 82.55 | 86.86 | 83.42 | 71.16 | 85.40 | 83.73 | 89.05 | 47.60 |
| surrogate-5B | 80.76 | 70.38 | 67.30 | 56.57 | 77.98 | 73.99 | 86.88 | 37.80 |


## Llama-3.1-8B

#### :paw_prints: Surrogate-trained Encoder

| checkpoint | :hugs: hf model repo |
|:---:|:---:|
| llama3.1-8b_surrogate-trained-encoder | [`link`](https://huggingface.co/kaiyuyue/zero-checkpoints/tree/main/llama3.1-8b_surrogate-trained-encoder) |

| model | MME<sup>binary</sup> cog. | MME<sup>binary</sup> perc. | MME cog. | MME perc. | POPE<sup>binary</sup> acc. | POPE<sup>binary</sup> f1. | POPE acc. | POPE f1. | SEED-Bench<sub>all</sub> | SEED-Bench<sub>img</sub> | SEED-Bench<sub>vid</sub> | MM-Vet | LLaVA-Wild | MMBench | CVbench<sub>2d</sub> acc. | CVBench<sub>3d</sub> acc. | CVBench<sub>combined</sub> acc. | GQA | VizWiz |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|surrogate-trained encoder | 239.64 | 1117.48 | 281.43 | 1063.29 | 81.16 | 78.81 | 83.89 | 83.54 | 53.33 | 57.28 | 38.38 | 17.61 | 36.20 | 47.34 | 49.58 | 56.00 | 52.79 | 45.56 | 34.85 |
| zero-shot grafting | 254.64 | 954.59 | 228.57 | 982.64 | 80.09 | 82.12 | 80.34 | 82.08 | 54.29 | 58.40 | 38.70 | 24.68 | 54.30 | 53.52 | 50.83 | 60.08 | 55.46 | 40.08 | 51.24 |

#### :paw_prints: Surrogate Language Model

| checkpoint | :hugs: hf model repo |
|:---:|:---:|
| llama3.1-8b_adapter_translator | [`link`](https://huggingface.co/kaiyuyue/zero-checkpoints/tree/main/llama3.1-8b_adapter_translator) |

| model | mmlu	| hellaswag	| arc_easy | arc_challenge | winogrande | piqa | boolq | openbookqa |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| few-shot | 5-shot | 10-shot | 0-shot | 25-shot | 5-shot | 0-shot | 0-shot | 0-shot |
| acc-type| - | acc_norm | acc_norm | acc_norm | - | acc_norm | - | acc_norm |
| Llama-8B | 68.36 | 80.45 | 79.80 | 61.77 | 77.27 | 81.50 | 85.41 | 44.80 |
| surrogate-5B | 66.80 | 61.24 | 59.34 | 44.80 | 70.88 | 71.00 | 69.30 | 34.20 |


## Llama-3.2-3B

#### :paw_prints: Surrogate-trained Encoder

| checkpoint | :hugs: hf model repo |
|:---:|:---:|
| llama3.2-3b_surrogate-trained-encoder | [`link`](https://huggingface.co/kaiyuyue/zero-checkpoints/tree/main/llama3.2-3b_surrogate-trained-encoder) |

| model | MME<sup>binary</sup> cog. | MME<sup>binary</sup> perc. | MME cog. | MME perc. | POPE<sup>binary</sup> acc. | POPE<sup>binary</sup> f1. | POPE acc. | POPE f1. | SEED-Bench<sub>all</sub> | SEED-Bench<sub>img</sub> | SEED-Bench<sub>vid</sub> | MM-Vet | LLaVA-Wild | MMBench | CVbench<sub>2d</sub> acc. | CVBench<sub>3d</sub> acc. | CVBench<sub>combined</sub> acc. | GQA | VizWiz |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|surrogate-trained encoder | 241.07	| 902.51 | 266.79 | 971.79 | 77.32 | 73.32 | 78.10 | 75.17 | 50.06 | 53.44 | 37.26 | 18.94 | 36.10 | 44.85 | 41.40 | 48.25 | 44.82 | 43.51 | 16.69 | 
| zero-shot grafting | 263.57 | 983.59 | 294.64 | 916.38 | 78.83 | 79.27 | 79.33 | 78.79 | 48.99 | 52.29 | 36.49 | 22.61 | 44.00 | 41.75 | 42.47 | 52.45 | 47.46 | 39.37 | 28.36 |

#### :paw_prints: Surrogate Language Model

| checkpoint | :hugs: hf model repo |
|:---:|:---:|
| llama3.2-3b_adapter_translator | [`link`](https://huggingface.co/kaiyuyue/zero-checkpoints/tree/main/llama3.2-3b_adapter_translator) |

| model | mmlu	| hellaswag	| arc_easy | arc_challenge | winogrande | piqa | boolq | openbookqa |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| few-shot | 5-shot | 10-shot | 0-shot | 25-shot | 5-shot | 0-shot | 0-shot | 0-shot |
| acc-type| - | acc_norm | acc_norm | acc_norm | - | acc_norm | - | acc_norm |
| Llama-3B | 60.74 | 73.04 | 71.04 | 52.65 | 70.64 | 77.09 | 78.93 | 39.20 |
| surrogate-2B | 58.90 |57.24 |54.76 | 38.48 | 64.33 | 67.57 | 78.20 | 32.60 |
