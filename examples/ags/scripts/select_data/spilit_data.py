import numpy as np
import json
from datasets import load_dataset
from collections import Counter


def set_seed(seed=42):
    """设置随机种子以确保结果可重复。"""
    np.random.seed(seed)


def split_jsonl(file_path, n_samples=264, seed=42, is_math_dataset=False):
    """
    将JSONL文件或数据集分割为训练集和测试集。

    参数：
    - file_path: JSONL文件的路径或数据集名称。
    - n_samples: 训练集的样本数量（对于非Math数据集）。
    - seed: 随机种子。
    - is_math_dataset: 是否为Math数据集。

    返回：
    - train_data: 训练集数据列表。
    - test_data: 测试集数据列表。
    """
    set_seed(seed)

    if is_math_dataset:
        return split_math_dataset(file_path)
    else:
        return split_general_dataset(file_path, n_samples)


def split_general_dataset(file_path, n_samples):
    """处理普通JSONL数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    n = len(data)
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_indices = indices[:n_samples]
    test_indices = indices[n_samples:]

    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, test_data


def split_math_dataset(dataset_name):
    """处理Math数据集"""
    dataset = load_dataset(dataset_name, 'all', split='test')
    selected_types = ['Prealgebra', 'Number Theory', 'Precalculus', 'Counting & Probability']
    filtered_dataset = dataset.filter(lambda x: x['level'] == 'Level 5' and x['type'] in selected_types)

    train_data = []
    test_data = []

    for data_type in selected_types:
        type_data = filtered_dataset.filter(lambda x: x['type'] == data_type)
        total_samples = len(type_data)
        train_samples = total_samples // 5  # 1/5 for training

        train_indices = np.random.choice(total_samples, train_samples, replace=False)
        test_indices = np.setdiff1d(np.arange(total_samples), train_indices)

        train_data.extend([type_data[int(i)] for i in train_indices])
        test_data.extend([type_data[int(i)] for i in test_indices])

    return train_data, test_data


def save_jsonl(data, output_path):
    """将数据保存为JSONL文件。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# 使用示例
def process_dataset(file_path, n_samples, is_math_dataset, output_prefix):
    train_data, test_data = split_jsonl(file_path, n_samples=n_samples, is_math_dataset=is_math_dataset)

    save_jsonl(train_data, f'{output_prefix}_validation.jsonl')
    save_jsonl(test_data, f'{output_prefix}_test.jsonl')

    print(f"{output_prefix} 训练集样本数: {len(train_data)}")
    print(f"{output_prefix} 测试集样本数: {len(test_data)}")

    if is_math_dataset:
        type_counts = Counter(item['type'] for item in train_data + test_data)
        print(f"{output_prefix} 类型统计:", type_counts)
        print(f"{output_prefix} 总计:", sum(type_counts.values()))


# 处理普通数据集
process_dataset("gsm8k.jsonl", 264, False, "gsm8k")

# 处理Math数据集
process_dataset("lighteval/MATH", 0, True, "math")  # n_samples对Math数据集不适用，所以设为0