import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import lightning as L
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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
eval_iters = 200

def setup_model_parallel():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

local_rank, world_size = setup_model_parallel()

auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=TransformerBlock)

# fabric = L.Fabric(accelerator="cuda", devices=2, precision="bf16-mixed", strategy=strategy)
fabric = L.Fabric(accelerator="cuda", devices=2, strategy=strategy)
fabric.launch()
fabric.seed_everything(1337 + fabric.global_rank)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", fast=False)
tokenizer.pad_token = "<unk>"


dim = 4096
n_heads = 2
n_layers = 2
vocab_size = 32000
epochs = 1
out_dir = "out/training"
eval_interval = 500
eval_iters = 100
log_interval = 500


# Hyperparameters
learning_rate = 3e-4
batch_size = 2
max_iters = 60
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

#with fabric.device:
model_args = fabric.to_device(ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads, vocab_size=vocab_size))  # Update these parameters as needed
model = fabric.to_device(Transformer(model_args))

model = fabric.setup_module(model)

# Load the dataset
train_data, valid_data, test_data = load_data()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
optimizer = fabric.setup_optimizers(optimizer)

def save_model_checkpoint(fabric, model, file_path):
    """Handles boilerplate logic for retrieving and saving the state_dict.
    
    This will be upstreamed to Fabric soon.
    """
    if isinstance(fabric.strategy, FSDPStrategy):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model._forward_module.state_dict()
    else:
        state_dict = model.state_dict()

    if fabric.global_rank == 0:
        torch.save(state_dict, file_path)
    fabric.barrier()

# Initialize weights
def initialize_weights(m):

    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        m.weight.requires_grad = True
        m.bias.requires_grad = True
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))


@torch.no_grad()
def validate(fabric, model, validation_dataset):
    model.eval()
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for k in range(eval_iters):
            batch = validation_dataset[k]
            logits = model(fabric.to_device(batch["input_ids"]), 0)
            labels = fabric.to_device(batch["labels"])
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            #loss = Variable(loss, requires_grad = True)
            losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

# @torch.no_grad()
# def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
#     fabric.print("Validating ...")
#     model.eval()
#     losses = torch.zeros(eval_iters)
#     for batch in validation_dataset:
#     for k in range(eval_iters):
#         input_ids, targets = get_batch(
#             fabric,
#             val_data,
#             block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
#         )
#         logits = model(input_ids)
#         loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
#         losses[k] = loss.item()
#     out = losses.mean()
#     model.train()
#     return out

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def validate(model, val_loader):
#     model.eval()
#     with torch.no_grad():
#         all_preds = []
#         all_labels = []
#         for inputs, labels in val_loader:
#             outputs = model(inputs)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#         accuracy = accuracy_score(all_labels, all_preds)
#         precision = precision_score(all_labels, all_preds, average='macro')
#         recall = recall_score(all_labels, all_preds, average='macro')
#         f1 = f1_score(all_labels, all_preds, average='macro')
#     return accuracy, precision, recall, f1

#print("start_init")
model.apply(initialize_weights)
# TODO: save initial weight
for param in model.parameters():
    param.requires_grad = True
#print("end_init")

f_count = 1
filename = "out/loss/"+str(f_count)+".txt"
stat_f = open(filename, "w")

def train(
    fabric,
    model,
    optimizer: torch.optim.Optimizer,
    train_data,
    val_data):

    iter_num = 0
    
    # save parameters to file
    stat_f.write('Parameters:\n')
    stat_f.write(f'learning_rate = {learning_rate}\n')
    stat_f.write(f'eval_iters = {eval_iters}\n')
    stat_f.write(f'dim = {dim}\n')
    stat_f.write(f'n_heads = {n_heads}\n')
    stat_f.write(f'n_layers = {n_layers}\n')
    stat_f.write(f'vocab_size = {vocab_size}\n')
    stat_f.write(f'epochs = {epochs}\n')
    stat_f.write(f'log_interval = {log_interval}\n')
    stat_f.write(f'max_iters = {max_iters}\n')
    stat_f.write(f'eval_interval = {eval_interval}\n\n')

    while True:
        
        # TODO: add learning rate scheduling
        model.train()
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss = validate(fabric, model, val_data)
            stat_f.write(f'Iteration {iter_num}: val loss = {val_loss:.4f}\n')
            print("test")
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            fabric.print(f"Saving checkpoint to {out_dir}")
            # TODO: save checkpoint
            save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"))
            # state = {"model": model, "optimizer": optimizer, "iteration": iter_num, "hparams": ...}

        t0 = time.time()
        total_iter_loss = 0
        for i, batch in tqdm(enumerate(train_data), desc ="Iteration " + str(iter_num) + ": "):
            logits = model(fabric.to_device(batch["input_ids"]), 0)
            labels = fabric.to_device(batch["labels"])

            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = Variable(loss, requires_grad = True)
            total_iter_loss += loss.item()

            fabric.backward(loss)

            # Gradient clipping
            if grad_clip != 0.0:
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

            optimizer.step()
            optimizer.zero_grad()
        batch_loss = total_iter_loss / len(train_data)


        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"loss: {batch_loss:.4f}, time: {dt:.2f}s")
            stat_f.write(f'Iteration {iter_num}: training loss = {batch_loss:.4f}, time: {dt:.2f}s\n')
            
        iter_num += 1
        if iter_num > max_iters:
            break

train(fabric, model, optimizer, train_data, valid_data)
stat_f.close()
f_count+=1


