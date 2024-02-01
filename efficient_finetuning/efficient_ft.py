import transformers
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import torch
import time
import os
import json

# from dataset_factory.make_dataset import LazySupervisedDataset
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import random
import torch

# from modeling.dynamic_batching_trainer import split_then_stack
IGNORE_TOKEN_ID = -100
random.seed(42)
from loguru import logger


class DynamicBatchingTrainer(transformers.Trainer):
    def __init__(self, avg_num_train_tokens, *args, **kwargs):
        kwargs["train_dataset"] = DynamicbatchingDataset(
            kwargs["train_dataset"], data_max_length=kwargs["args"].data_max_length,
        )

        super().__init__(*args, **kwargs)
        self.avg_num_train_tokens = avg_num_train_tokens

    def get_train_dataloader(self) -> DataLoader:
        return self.accelerator.prepare(
            DataLoader(
                self.train_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                collate_fn=lambda x: self.collate_fn(
                    x, self.tokenizer.pad_token_id, IGNORE_TOKEN_ID, False
                ),
            )
        )

    def collate_fn(
        self,
        batch,
        input_ids_pad_token_id,
        labels_ignore_token_id=-100,
        mask_pad_token_id=False,
    ):
        inputs = batch[0]
        split_ids = inputs.pop("split_ids")
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                if k == "input_ids":
                    pad_token_id = input_ids_pad_token_id
                elif k == "labels":
                    pad_token_id = labels_ignore_token_id
                elif k == "attention_mask":
                    pad_token_id = mask_pad_token_id
                else:
                    raise NotImplementedError
                out = split_then_stack(inputs[k], split_ids, pad_token_id)
                inputs[k] = out
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        # t = time.time()
        loss = super().compute_loss(model, inputs, return_outputs)
        # batch_avg_num_train_tokens = (
        #     inputs["labels"].ne(IGNORE_TOKEN_ID).sum(dim=1).float().mean()
        # )
        # scale_factor = batch_avg_num_train_tokens / self.avg_num_train_tokens
        loss = loss  # * scale_factor
        # def logging_outputs(self, *args, **kwargs):
        # if self.state.global_step % 10 == 0:
        #     RANK = int(os.environ.get("LOCAL_RANK") or 0)
        #     padding_percentage = (
        #         inputs["input_ids"].eq(self.tokenizer.pad_token_id).sum().item()
        #         / inputs["input_ids"].numel()
        #         * 100
        #     )
        #     mean_loss_per_train_token = loss / batch_avg_num_train_tokens
        #     run_time = time.time() - t

        #     length_by_sample = (
        #         inputs["input_ids"].ne(self.tokenizer.pad_token_id).sum(dim=1)
        #     )

        # print(
        #     f"[{RANK=}] [length_by_sample={length_by_sample}, {padding_percentage=:.2f}%] {run_time=:0.2f}, {scale_factor=:2f}, {loss=:.4f}, {mean_loss_per_train_token=:.4f}"
        # )

        return loss


def split_then_stack(merged_ids, split_idxs, pad_value):
    # List to hold the split tensors
    batch_tensors = []

    # Previous split index, starting from 0
    prev_idx = 0
    for idx in split_idxs:
        # Split the tensor according to the split index
        batch = merged_ids[prev_idx:idx]
        batch_tensors.append(batch)
        prev_idx = idx

    # Find the longest length in the split tensors
    max_length = max(batch.size(0) for batch in batch_tensors)
    dtype = batch_tensors[0].dtype
    # Pad each tensor to have the same length as the longest tensor
    padded_batches = [
        torch.nn.functional.pad(
            batch, (0, max_length - batch.size(0)), "constant", pad_value
        )
        for batch in batch_tensors
    ]

    # Stack the padded tensors along a new dimension
    stacked_tensor = torch.stack(padded_batches).to(dtype=dtype)
    return stacked_tensor


class DynamicbatchingDataset(Dataset):
    def __init__(self, dataset, data_max_length, pad_val=-1):
        dataset.padding = False
        self.dataset = dataset
        self.pad_val = pad_val
        self.batches_with_splits = self.__get_random_batch_ids_with_splits(
            data_max_length
        )

    def get_lengths(self, dataset):
        dataset.padding = False
        assert dataset.padding == False
        lens = []
        for item in dataset:
            lens.append(item["input_ids"].shape[0])
        return lens

    def __validity_ratio(self, l, first_item_len):
        if first_item_len is None:
            return True
        return abs(l - first_item_len) < 256

    def __get_random_batch_ids_with_splits(self, max_length):
        lens = self.get_lengths(self.dataset)
        len_to_indexes = {}

        # Populate len_to_indexes with indices for elements longer than max_length
        for index, length in enumerate(lens):
            if (
                length > max_length
            ):  # Fix applied here: We want to include lengths up to max_length
                continue
            if length not in len_to_indexes:
                len_to_indexes[length] = []
            len_to_indexes[length].append(index)

        batches_with_splits = []
        from tqdm import tqdm

        pbar = tqdm(total=len(lens))
        while len(len_to_indexes) > 0:
            pbar.update(1)
            current_batch_indexes = []
            current_split_points = []
            len_left = max_length
            accumulated_length = 0

            first_item_length = None
            count = 0  # 'round' is a built-in function. Consider renaming this variable to avoid shadowing it.
            while True:
                available_lengths = [
                    l
                    for l in len_to_indexes.keys()
                    if l <= len_left and self.__validity_ratio(l, first_item_length)
                ]
                if not available_lengths:
                    break

                chosen_length = random.choice(available_lengths)
                if first_item_length is None:
                    first_item_length = chosen_length
                chosen_index = random.choice(len_to_indexes[chosen_length])
                current_batch_indexes.append(chosen_index)
                accumulated_length += chosen_length
                current_split_points.append(accumulated_length)
                len_left -= chosen_length

                len_to_indexes[chosen_length].remove(chosen_index)
                if len(len_to_indexes[chosen_length]) == 0:
                    del len_to_indexes[chosen_length]

                count += 1
                # Avoid infinite loops; careful with the use of 'assert' in production code
                if count >= 50:
                    logger.warning(
                        "Warning: Maximum rounds reached, breaking out to avoid infinite loop."
                    )
                    break

            if current_batch_indexes:
                batches_with_splits.append(
                    (current_batch_indexes, current_split_points)
                )
                this_batch_persample_length = [lens[i] for i in current_batch_indexes]
                if pbar.n % 1000 == 0:
                    logger.success(
                        f"Finished creating batch with {this_batch_persample_length} items."
                    )

        return batches_with_splits

    def __len__(self):
        return len(self.batches_with_splits)

    def __merge_dict(self, list_d):
        d = list_d[0]
        todo_keys = [k for k in d if isinstance(d[k], torch.Tensor)]
        ret_d = {}
        for k in todo_keys:
            ret_d[k] = []

        for d in list_d:
            for k in todo_keys:
                ret_d[k].append(d[k])
        for k in todo_keys:
            ret_d[k] = torch.cat(ret_d[k])
        return ret_d

    def __getitem__(self, idx):
        ids, split_ids = self.batches_with_splits[idx]
        items = [self.dataset[idx] for idx in ids]
        item = self.__merge_dict(items)
        item["split_ids"] = split_ids
        return item
