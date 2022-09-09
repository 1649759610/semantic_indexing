# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import random
import argparse
import numpy as np
from scipy import stats
from functools import partial

import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from visualdl import LogWriter

from model import SimCSE
from utils import set_seed, eval_metric
from data import read_text_single, read_text_pair, convert_example, create_dataloader, word_repetition


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "eval", "infer"], default="infer", help="Select which mode to run model, defaults to infer.")
parser.add_argument("--model_name_or_path", type=str, help="The full path of eval_set_file.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--output_emb_size", default=0, type=int, help="Output_embedding_size, 0 means use hidden_size as output embedding size.")
parser.add_argument("--infer_with_fc_pooler", action="store_true", help="Whether use fc layer after cls embedding or not for when infer.")
parser.add_argument("--train_set_file", type=str, help="The full path of train_set_file.")
parser.add_argument("--eval_set_file", type=str, help="The full path of eval_set_file.")
parser.add_argument("--infer_set_file", type=str, help="The full path of infer_set_file.")
parser.add_argument("--ckpt_dir", default="", type=str, help="The ckpt directory where the model checkpoints will be loaded.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, help="The save directory where the model checkpoints will be written.")
parser.add_argument("--log_dir", default=None, type=str, help="The save directory where the training log will be written.")
parser.add_argument("--save_infer_result_path", default="./infer_res.txt", type=str, help="The save directory where the infer result will be written.")
parser.add_argument("--save_steps", type=int, default=10000, help="Step interval for saving checkpoint.")
parser.add_argument("--eval_steps", type=int, default=10000, help="Step interval for evaluation.")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override ecpochs.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--epochs", default=1, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout for pretrained model encoder.")
parser.add_argument("--margin", default=0.0, type=float, help="Margin beteween pos_sample and neg_samples.")
parser.add_argument("--scale", default=20, type=int, help="Scale for pair-wise margin_rank_loss.")
parser.add_argument("--dup_rate", default=0.32, type=float, help="duplicate rate for word reptition.")
parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()


def do_infer(model, tokenizer, data_loader):
    sims = []
    model.eval()
    with paddle.no_grad():
        for batch_data in data_loader:
            query_input_ids, query_token_type_ids, key_input_ids, key_token_type_ids = batch_data
            cosine_sim = model.cosine_sim(
                query_input_ids=query_input_ids,
                key_input_ids=key_input_ids,
                query_token_type_ids=query_token_type_ids,
                key_token_type_ids=key_token_type_ids,
                with_pooler=args.infer_with_fc_pooler
            )
            sims.append(cosine_sim.numpy())
        sims = np.concatenate(sims, axis=0)
        return sims


def do_eval(model, tokenizer, data_loader):
    spearman_corr = 0.0
    sims = []
    labels = []
    model.eval()
    with paddle.no_grad():
        for batch in data_loader:
            query_input_ids, query_token_type_ids, key_input_ids, key_token_type_ids, label = batch
            cosine_sim = model.cosine_sim(
                query_input_ids=query_input_ids,
                key_input_ids=key_input_ids,
                query_token_type_ids=query_token_type_ids,
                key_token_type_ids=key_token_type_ids,
                with_pooler=args.infer_with_fc_pooler
            )
            sims.append(cosine_sim.numpy())
            labels.append(label.numpy())

    sims = np.concatenate(sims, axis=0)
    labels = np.concatenate(labels, axis=0)

    score = eval_metric(labels, sims)
    return score


def do_train(model, tokenizer, train_data_loader, dev_data_loader):
    num_training_steps = args.max_steps if args.max_steps > 0 else len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(
        args.learning_rate,
        num_training_steps,
        args.warmup_proportion
    )

    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params
    )

    global_step = 0
    best_spearman = 0.0
    tic_train = time.time()
    model = paddle.DataParallel(model)
    model.train()
    with LogWriter(logdir=os.path.join(args.log_dir, "scalar")) as writer:
        for epoch in range(1, args.epochs + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                query_input_ids, query_token_type_ids, key_input_ids, key_token_type_ids = batch
                if(args.dup_rate > 0):
                    query_input_ids, query_token_type_ids = word_repetition(query_input_ids, query_token_type_ids, args.dup_rate)
                    key_input_ids, key_token_type_ids = word_repetition(key_input_ids, key_token_type_ids, args.dup_rate)

                loss = model(
                    query_input_ids=query_input_ids,
                    key_input_ids=key_input_ids,
                    query_token_type_ids=query_token_type_ids,
                    key_token_type_ids=key_token_type_ids,
                    with_pooler=args.infer_with_fc_pooler
                )

                global_step += 1
                if global_step % (args.eval_steps // 10) == 0 and rank == 0:
                    print("global step {}, epoch: {}, batch: {}, loss: {:.5f}, speed: {:.2f} step/s".format(global_step, epoch, step, loss.item(), (args.eval_steps // 10) / (time.time() - tic_train)))
                    writer.add_scalar(tag="train/loss", step=global_step, value=loss.item())
                    tic_train = time.time()

                if global_step % args.eval_steps == 0 and rank == 0:
                    spearman = do_eval(model._layers, tokenizer, dev_data_loader)
                    if best_spearman < spearman:
                        print("best checkpoint has been updated: from last best_spearman {} --> new spearman {}.".format(best_spearman, spearman))
                        best_spearman = spearman
                        # save best model
                        save_dir = os.path.join(args.save_dir, "best")
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        save_param_path = os.path.join(save_dir, "model_state.pdparams")
                        paddle.save(model.state_dict(), save_param_path)
                        tokenizer.save_pretrained(save_dir)
                    
                    writer.add_scalar(tag="eval/spearman", step=global_step, value=spearman)
                    model.train()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % args.save_steps == 0 and rank == 0:
                    save_dir = os.path.join(args.save_dir, "checkpoint_{}".format(global_step))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, "model_state.pdparams")
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    return model


if __name__ == "__main__":
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    tokenizer = ppnlp.transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    pretrained_model = ppnlp.transformers.AutoModel.from_pretrained(
       args.model_name_or_path,
       hidden_dropout_prob=args.dropout,
       attention_probs_dropout_prob=args.dropout
    )

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length
    )

    model = SimCSE(
        pretrained_model,
        margin=args.margin,
        scale=args.scale,
        output_emb_size=args.output_emb_size
    )


    if args.mode == "train":
        start_time = time.time()

        train_ds = load_dataset(read_text_single, data_path=args.train_set_file, lazy=False)
        dev_ds = load_dataset(read_text_pair, data_path=args.eval_set_file, lazy=False)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # key_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
        ): [data for data in fn(samples)]
        dev_batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # key_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
            Stack(dtype="int64"),  # labels
        ): [data for data in fn(samples)]

        train_data_loader = create_dataloader(
            train_ds,
            mode="train",
            batch_size=args.batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func
        )
        dev_data_loader = create_dataloader(
            dev_ds,
            mode="eval",
            batch_size=args.batch_size,
            batchify_fn=dev_batchify_fn,
            trans_fn=trans_func
        )

        do_train(model, tokenizer, train_data_loader, dev_data_loader)

        end_time = time.time()
        print("running time {} s".format(end_time - start_time))

    if args.mode == "eval":
        start_time = time.time()
        if args.ckpt_dir:
            init_from_ckpt = os.path.join(args.ckpt_dir, "model_state.pdparams")
            if os.path.isfile(init_from_ckpt):
                print("*************************initializing model from {}*****************************".format(init_from_ckpt))
                state_dict = paddle.load(init_from_ckpt)
                model.set_dict(state_dict)

        dev_ds = load_dataset(read_text_pair, data_path=args.eval_set_file, lazy=False)

        dev_batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # key_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
            Stack(dtype="int64"),  # labels
        ): [data for data in fn(samples)]

        dev_data_loader = create_dataloader(
            dev_ds,
            mode="eval",
            batch_size=args.batch_size,
            batchify_fn=dev_batchify_fn,
            trans_fn=trans_func
        )

        score = do_eval(model, tokenizer, dev_data_loader)
        print("[Evaluation] score: {}".format(score))

        end_time = time.time()
        print("running time {} s".format(end_time - start_time))

    if args.mode == "infer":
        start_time = time.time()
        if args.ckpt_dir:
            init_from_ckpt = os.path.join(args.ckpt_dir, "model_state.pdparams")
            if os.path.isfile(init_from_ckpt):
                print("*************************initializing model from {}*****************************".format(init_from_ckpt))
                state_dict = paddle.load(init_from_ckpt)
                model.set_dict(state_dict)

        infer_ds = load_dataset(read_text_pair, data_path=args.infer_set_file, lazy=False, is_infer=True)
        
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # key_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
        ): [data for data in fn(samples)]

        infer_data_loader = create_dataloader(
            infer_ds,
            mode="predict",
            batch_size=args.batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func
        )

        cosin_sim = do_infer(model, tokenizer, infer_data_loader)

        with open(args.save_infer_result_path, "w", encoding="utf-8") as f:
            for idx, cos in enumerate(cosin_sim):
                msg = "{}\n".format(cos)
                f.write(msg)
        print("[Infer] infer result has been saved to {}.".format(args.save_infer_result_path))
        end_time = time.time()
        print("running time {} s".format(end_time - start_time))
        
        

