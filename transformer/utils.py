from torch import nn

act2fn = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}

import torch
import json
import logging

from torch.utils.data import Dataset

# {"tokens": ["[CLS]", "...", "[SEP]", "...", "[SEP]"], masked_positions: [...], masked_tokens: ["..."], "next_sentence_label": ...}
def load_dataset(data_file, tokenizer, is_debug=False, max_length=512):
    training_data = {
        # "tokens": [],
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "labels": [],
        "next_sentence_label": [],
        # "sentence_length": [],
    }
    with open(data_file) as f:
        if is_debug:
            limit = 10

        sent_cnt = 0
        for line in f:
            data = json.loads(line)

            tokens = data["tokens"]
            try:
                sep = tokens.index("[SEP]") + 1
            except:
                raise f"data: {data}"
            tokens_a = tokens[:sep]
            tokens_b = tokens[sep:]

            # padding
            tokens_b.extend(["[PAD]"] * (max_length - len(tokens)))
            
            labels = [-100] * max_length
            for (idx, pos) in enumerate(data["masked_positions"]):
                labels[pos] = tokenizer.convert_tokens_to_ids(
                    data["masked_tokens"][idx]
                )

            encodings = tokenizer.encode_plus(
                tokens_a,
                tokens_b,
                is_pretokenized=True,
                add_special_tokens=False,
            )
            next_sentence_label = data["next_sentence_label"]

            # if set add_special_tokens=False, all token_type_ids will 0
            # so we need make a correct token_type_ids manually
            # token_type_ids = [0] * sep + [1] * (len(tokens) - sep)
            token_type_ids = [0] * sep + [1] * (max_length - sep)

            # encode_plus will be disable to preduce a correct attention_mask when padding is set on
            # we need to create attention_mask manually
            attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))

            # training_data["tokens"].append(tokens)
            training_data["input_ids"].append(encodings["input_ids"])
            training_data["token_type_ids"].append(token_type_ids)
            training_data["attention_mask"].append(attention_mask)
            training_data["labels"].append(labels)
            training_data["next_sentence_label"].append(next_sentence_label)
            # training_data["sentence_length"].append(len(tokens))

            sent_cnt += 1
            if sent_cnt % 500 == 0:
                logging.info(f"{sent_cnt} processed.")
            if is_debug:
                limit -= 1
                if limit <= 0:
                    break

    return training_data


class PreTrainingDataset(Dataset):
    def __init__(
        self, input_ids, token_type_ids, attention_mask, labels, next_sentence_label
    ):
        self.input_ids = torch.LongTensor(input_ids)
        self.token_type_ids = torch.LongTensor(token_type_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.labels = torch.LongTensor(labels)
        self.next_sentence_label = torch.LongTensor(next_sentence_label)

    def __len__(self):
        return len(self.next_sentence_label)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        token_type_ids = self.token_type_ids[idx]
        attention_mask = self.attention_mask[idx]
        mlm_labels = self.labels[idx]
        next_sentence_label = self.next_sentence_label[idx]
        return {
            "input_ids": input_ids, 
            "token_type_ids": token_type_ids, 
            "attention_mask": attention_mask, 
            "labels": mlm_labels, 
            "next_sentence_label": next_sentence_label,
        }
