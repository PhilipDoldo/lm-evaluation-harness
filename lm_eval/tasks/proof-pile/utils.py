import datasets
import sys

def process_docs(dataset: datasets.Dataset):
    def tokenize(example):
            tokenized = tokenizer( # how do we access the tokenizer in the preprocessing step?
                example["text"],
                add_special_tokens=False,
                padding=True,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
            )
            example["input_ids"] = tokenized["input_ids"]
            example["attention_mask"] = tokenized["attention_mask"]
            example["tokenized_len"] = len(tokenized["input_ids"])
            return example

    dataset = dataset.map(tokenize)
    dataset = dataset.filter(
            lambda x: x["tokenized_len"] >= 131072)
    dataset = dataset[10]
    return dataset
    