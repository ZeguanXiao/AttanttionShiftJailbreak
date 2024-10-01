import os
import pandas as pd
from typing import Dict, List


def load_data(dataset, dataset_size) -> List[Dict]:
    if dataset == "advbench_behaviors_custom":
        df = pd.read_csv('data/advbench/harmful_behaviors_custom.csv')
        dataset = list(df["goal"][:dataset_size])
    elif dataset == "advbench_behaviors_custom_complementary":
        df = pd.read_csv('data/advbench/harmful_behaviors_custom.csv')
        filter_index = list(df["Original index"])
        df = pd.read_csv('data/advbench/harmful_behaviors.csv')
        harmful_behaviors = list(df["goal"][:dataset_size])
        dataset = [harmful_behaviors[i] for i in range(len(harmful_behaviors)) if i not in filter_index]
    elif dataset == "advbench_behaviors":
        df = pd.read_csv('data/advbench/harmful_behaviors.csv')
        dataset = list(df["goal"][:dataset_size])
    elif dataset == "advbench_behaviors_custom_addChar":
        df = pd.read_csv('data/query_level_augmented_avdbench_custom/advbench_behaviors_custom_addChar.csv')
        dataset = list(df["goal"][:dataset_size])
        dataset = [d.strip('"') for d in dataset]
    elif dataset == "advbench_behaviors_custom_changeOrder":
        df = pd.read_csv('data/query_level_augmented_avdbench_custom/advbench_behaviors_custom_changeOrder.csv')
        dataset = list(df["goal"][:dataset_size])
        dataset = [d.strip('"') for d in dataset]
    elif dataset == "advbench_behaviors_custom_languageMix":
        df = pd.read_csv('data/query_level_augmented_avdbench_custom/advbench_behaviors_custom_languageMix.csv')
        dataset = list(df["goal"][:dataset_size])
        dataset = [d.strip('"') for d in dataset]
    elif dataset == "advbench_behaviors_custom_misrewriteSentence":
        df = pd.read_csv('data/query_level_augmented_avdbench_custom/advbench_behaviors_custom_misrewriteSentence.csv')
        dataset = list(df["goal"][:dataset_size])
        dataset = [d.strip('"') for d in dataset]
    elif dataset == "advbench_behaviors_custom_translateToBengali":
        df = pd.read_csv('data/query_level_augmented_avdbench_custom/advbench_behaviors_custom_translateToBengali.csv')
        dataset = list(df["goal"][:dataset_size])
        dataset = [d.strip('"') for d in dataset]
    elif dataset == "advbench_behaviors_custom_translateToZulu":
        df = pd.read_csv('data/query_level_augmented_avdbench_custom/advbench_behaviors_custom_translateToZulu.csv')
        dataset = list(df["goal"][:dataset_size])
        dataset = [d.strip('"') for d in dataset]
    elif dataset == "advbench_behaviors_custom_translateToMorse":
        df = pd.read_csv('data/query_level_augmented_avdbench_custom/advbench_behaviors_custom_translateToMorse.csv')
        dataset = list(df["goal"][:dataset_size])
        dataset = [d.strip('"') for d in dataset]
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return dataset
