import os
import json
from datasets import load_dataset as load_huggingface_dataset


def load_toxicity_data(data_dir, subset):
    with open(os.path.join(data_dir, "train_0.jsonl"), "r") as f:
        neg_data = list(map(json.loads, f.readlines()))
        for _datum in neg_data:
            _datum["label"] = 1
    with open(os.path.join(data_dir, "train_1.jsonl"), "r") as f:
        pos_data = list(map(json.loads, f.readlines()))
        for _datum in pos_data:
            _datum["label"] = -1

    if subset is not None:
        pos_data = pos_data[:subset]
        neg_data = neg_data[:subset]

    dataset = neg_data + pos_data
    return dataset


def load_sentiment_data(dataset_name):
    '''
    1. sentiment-sst2，原始标签0，1，映射成 -1 和 1
    2. sentiment-yelp，原始标签1-5,映射关系有点混乱
    3. sentiment-sst5，原始标签0-4，映射成-1,-0.5,0,0.5,1，与论文中的不太一样
    4. sentiment-sst5-positive，原始标签1
    5. sentiment-sst5-negative，原始标签0
    '''
    dataset = []
    # 代码处理 SST-2（Stanford Sentiment Treebank）情感分析数据集
    if dataset_name in ["sentiment-sst2", "sentiment-all"]:
        sst2 = list(load_huggingface_dataset("sst2")["train"])
        for _datum in sst2:
            # 转换标签为 -1 和 1，与其他数据集保持一致
            _datum["label"] = _datum["label"] * 2 - 1
            _datum["text"] = _datum["sentence"]
            _datum.pop("idx")
            _datum.pop("sentence")
        dataset.extend(sst2)
    if dataset_name in ["sentiment-yelp", "sentiment-all"]:
        yelp = list(load_huggingface_dataset("yelp_review_full")["train"])
        for _datum in yelp:
            _datum["label"] = _datum["label"] / 2.5 - 1
        dataset.extend(yelp)
    if dataset_name == "sentiment-sst5":
        # 映射后的label好像不太对，映射成了-1，-0.5，0，0.5，1，与论文中的不太一样
        sst5 = list(load_huggingface_dataset("SetFit/sst5")["train"])
        for _datum in sst5:
            _datum["label"] = _datum["label"] / 2 - 1
            _datum.pop("label_text")
        dataset.extend(sst5)
    elif dataset_name == "sentiment-sst5-positive":
        sst5 = list(load_huggingface_dataset("SetFit/sst5")["train"])
        # 原始代码有点问题，这里应该是 d["label"] == 4
        # sst5 = [d for d in sst5 if d["label"] == 1]
        sst5 = [d for d in sst5 if d["label"] == 4]
        for _datum in sst5:
            _datum["label"] = _datum["label"] / 2 - 1
            _datum.pop("label_text")
        dataset.extend(sst5)
    elif dataset_name == "sentiment-sst5-negative":
        sst5 = list(load_huggingface_dataset("SetFit/sst5")["train"])
        sst5 = [d for d in sst5 if d["label"] == 0]
        for _datum in sst5:
            _datum["label"] = _datum["label"] / 2 - 1
            _datum.pop("label_text")
        dataset.extend(sst5)

    return dataset


def load_toy_sentiment_data(dataset_name):
    dataset = {
        "toy-sentiment-1":
        [
            {"text": "This is excellent!", "label": 1},
            {"text": "This is terrible.", "label": -1},
        ],
        "toy-sentiment-2":
        [
            {"text": "This is excellent, perfect and fabulous.", "label": 1},
            {"text": "This is terrible, bad, intolerable.", "label": -1},
        ],
    }[dataset_name]
    return dataset

# load dataset from data_dir or huggingface hub
def load_dataset(dataset_name, data_dir, subset):
    if dataset_name == "toxicity":
        dataset = load_toxicity_data(data_dir, subset)
    elif dataset_name.startswith("sentiment"):
        dataset = load_sentiment_data(dataset_name)
    elif dataset_name.startswith("toy-sentiment"):
        dataset = load_toy_sentiment_data(dataset_name)
    else:
        raise NotImplementedError()

    return dataset
