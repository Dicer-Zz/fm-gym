import os
import torch
import logging

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForPreTraining, BertTokenizer

from utils import load_dataset, PreTrainingDataset
from bert_model import BertModel

def train(model, train_dataset, eval_dataset, optimizer, device, epochs=3):
    # TODO: better display
    model.cuda()
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16)
    for epoch in range(epochs):
        # training
        train_loss_history = []
        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            next_sentence_label = batch["next_sentence_label"].to(device)

            outputs = model(
                input_ids,
                attention_mask,
                token_type_ids,
                labels=labels,
                next_sentence_label=next_sentence_label,
                # mlm_labels=labels,
                # nsp_labels=next_sentence_label,
            )
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                logging.info(f"training loss: {loss}")
                train_loss_history.append(loss)

        # evaluation
        eval_loss_history = []
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(eval_dataloader)):
                input_ids = batch["input_ids"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                next_sentence_label = batch["next_sentence_label"].to(device)

                outputs = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    labels=labels,
                    next_sentence_label=next_sentence_label,
                    # mlm_labels=labels,
                    # nsp_labels=next_sentence_label,
                )
                loss = outputs["loss"]
                if step % 50 == 0:
                    logging.info(f"eval loss: {loss}")
                    eval_loss_history.append(loss)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(filename)s %(levelname)s %(message)s"
    )

    model_name = "bert-base-uncased"
    config = BertConfig.from_pretrained(model_name)
    data_file = "/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/training_data_wikipedia_en_1k.jsonl"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # there will cost much time for preprocessing data
    training_data = load_dataset(data_file, tokenizer, is_debug=False)

    dataset = PreTrainingDataset(**training_data)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    model = BertForPreTraining(config)  # load a random-initialized model
    # model = BertModel()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8
    )
    # TODO: add a scheduler with warmup
    train(model, train_dataset, eval_dataset, optimizer, device=torch.device("cuda"))


if __name__ == "__main__":
    main()
