import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import lightning as L
import numpy as np
from lightning.fabric.strategies import FSDPStrategy
from functools import partial
torch.set_float32_matmul_precision('high')
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from model import Transformer, ModelArgs, TransformerBlock  # Assuming model.py is in the same directory
import os
import time
from tqdm.auto import tqdm
import math
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from preprocess_data import load_data
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from datetime import datetime
import argparse

os.environ['OMP_NUM_THREADS'] = '4'
out_dir = "true_out/training"
dim = 2048
n_heads = 4
n_layers = 4
vocab_size = 32000
log_interval = 25
training_sample = 30000

# Hyperparameters
learning_rate = 1e-2
batch_size = 64
max_iters = 100
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Load the dataset
train_data, valid_data, test_data, train_len, validation_len = load_data(batch_size, training_sample)

def main():
    m = Model()
    local_rank, world_size = m.setup_model_parallel()

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=TransformerBlock)

    fabric = L.Fabric(accelerator="cuda", devices=4, precision="bf16-mixed", strategy=strategy)
    #fabric = L.Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    # Initialize the tokenizer
    #tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", fast=False)
    #tokenizer.pad_token = "<unk>"

    #with fabric.device:
    model_args = fabric.to_device(ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads, vocab_size=vocab_size, max_batch_size=batch_size))  # Update these parameters as needed
    model = fabric.to_device(Transformer(model_args))
    model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    optimizer = fabric.setup_optimizers(optimizer)

    # initialize a model/ start from checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--load_iter_path', type=str, default="")
    args = parser.parse_args()
    load_file_path = args.load_iter_path
    if load_file_path == "" or load_file_path=="pretrain":
        print("initial weight!!!")
        model.apply(m.initialize_weights)
    elif not os.path.isfile(load_file_path):
        raise Exception(f'{load_file_path} is not a valid file path')
    else:
        m.load_model_checkpoint(fabric, model, load_file_path)
        m.file_cnt += 1
    for param in model.parameters():
        param.requires_grad = True

    # save initial weights
    m.save_model_checkpoint(fabric, model, optimizer)

    filename = "true_out/loss/"+str(m.file_cnt)+".txt"
    stat_f = open(filename, "w", buffering=1)

    # save parameters to file
    stat_f.write('Parameters:\n')
    stat_f.write(f'learning_rate = {learning_rate}\n')
    stat_f.write(f'dim = {dim}\n')
    stat_f.write(f'n_heads = {n_heads}\n')
    stat_f.write(f'n_layers = {n_layers}\n')
    stat_f.write(f'vocab_size = {vocab_size}\n')
    #stat_f.write(f'epochs = {epochs}\n')
    stat_f.write(f'log_interval = {log_interval}\n')
    stat_f.write(f'max_iters = {max_iters}\n')
    stat_f.write(f'batch_size = {batch_size}\n')
    stat_f.write(f'weight_decay = {weight_decay}\n')
    stat_f.write(f'beta1 = {beta1}\n')
    stat_f.write(f'beta2 = {beta2}\n')
    stat_f.write(f'grad_clip = {grad_clip}\n\n')

    print("Start training!")
    m.train(fabric, model, optimizer, train_data, valid_data, stat_f)

    stat_f.close()
    m.file_cnt += 1
    print("Training Complete")


class Model():
    def __init__(self):
        self.file_cnt = 1
        self.iter_num = 0

    def setup_model_parallel(self):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))

        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)
        return local_rank, world_size

    def save_model_checkpoint(self, fabric, model, optimizer):
        # if isinstance(fabric.strategy, FSDPStrategy):
        #     save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        #     with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        #         state_dict = model._forward_module.state_dict()
        # else:
        #     state_dict = model.state_dict()

        # if fabric.global_rank == 0:
        #     torch.save(state_dict, file_path)
        # fabric.barrier()
        file_path = os.path.join(out_dir, f"iter-{self.iter_num:06d}-ckpt.pth")
        save_state = {"model": model, "optimizer": optimizer, "iter_num": self.iter_num}
        fabric.save(file_path, save_state)

    def load_model_checkpoint(self, fabric, model, file_path):
        # if not file_path.is_file():
        #     print(f'model checkpoint {file_path} is not present. Returning...')
        #     return
        # checkpoint = torch.load(file_path)
        # #integrate into loaded model
        # model.load
        # example: "out/training/iter-000004-ckpt.pth"
        index = file_path.find('iter-')
        self.iter_num = int(file_path[index+5:index+11])
        fabric.load(file_path)
        

    # Initialize weights
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
            m.weight.requires_grad = True
            m.bias.requires_grad = True
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    @torch.no_grad()
    def validate(self, fabric, model, validation_dataset):
        model.eval()
        losses = 0
        with torch.no_grad():
            for i, batch in enumerate(validation_dataset):
                input_ids, labels = fabric.to_device(batch)   

                #labels = fabric.to_device(batch["labels"])
                logits = model(input_ids, 0)
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                masked_label = labels[labels.nonzero().view(-1)].type(torch.cuda.LongTensor)
                masked_logits = logits[labels.nonzero().view(-1), :]
                loss = torch.nn.functional.cross_entropy(masked_logits, masked_label)

                losses += loss.item()
        out = losses / validation_len
        model.train()
        return out

    def train(self,
        fabric,
        model,
        optimizer: torch.optim.Optimizer,
        train_data,
        val_data,
        stat_f):

        # run epoches
        while True:
            
            # TODO: add learning rate scheduling
            model.train()
            t0 = time.time()
            total_iter_loss = 0
            # run batches
            #for i, batch in tqdm(enumerate(train_data), desc ="Epoch " + str(self.iter_num) + ": ", miniters = 100):
            for i, batch in enumerate(train_data):

                input_ids, labels = fabric.to_device(batch)   

                #labels = fabric.to_device(batch["labels"])
                logits = model(input_ids, 0)
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                masked_label = labels[labels.nonzero().view(-1)].type(torch.cuda.LongTensor)
                masked_logits = logits[labels.nonzero().view(-1), :]
                loss = torch.nn.functional.cross_entropy(masked_logits, masked_label)
                loss = Variable(loss, requires_grad = True)

                total_iter_loss += loss.item()

                # print and save log every log_interval iter
                if i % log_interval == 0:
                    elapsed = time.time() - t0
                    log = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | {:5.4f} s/batch  | loss {:5.4f} | ppl {:8.2f}\n'.format(\
                    self.iter_num, i+1, train_len , learning_rate, elapsed / log_interval, total_iter_loss/(i+1), math.exp(total_iter_loss/(i+1)))
                    print(log)
                    print(log, file=stat_f)

                fabric.backward(loss)

                # Gradient clipping
                if grad_clip != 0.0:
                    fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

                optimizer.step()
                optimizer.zero_grad()
            epoch_loss = total_iter_loss / train_len

            dt = time.time() - t0
            
            # evaluate the loss on train/val sets at end of epoch and write checkpoints
            if self.iter_num > 0:
                val_loss = self.validate(fabric, model, val_data)
                log = '| end of epoch {:3d} | time elapsed {:3f}s | \
                valid loss {:5.2f} | valid ppl {:8.2f}\n | epoch loss {:5.4f} | epoch loss ppl {:5.2f}'.format(self.iter_num, dt, val_loss, \
                np.exp(val_loss), epoch_loss, math.exp(epoch_loss))
                print(log, file=stat_f)
                fabric.print(log)
                fabric.print(f"Saving checkpoint to {out_dir}")
                self.save_model_checkpoint(fabric, model, optimizer)
            
            
            self.iter_num += 1
            t0 = time.time()
            if self.iter_num > max_iters:
                break

main()

