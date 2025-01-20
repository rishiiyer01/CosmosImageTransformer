import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math
import sys
import torch.optim as optim
sys.path.append('cosmos')
from cosmos_tokenizer.image_lib import ImageTokenizer
from model import LatentModel, Block  # Assuming you still want Block imported for potential future use

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import os

from functools import reduce
import operator
import numpy as np

from torch.distributions import Categorical
from torchvision import transforms

import wandb
from PIL import Image
from datasets import load_dataset
import sys


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import torch.distributed as dist
import functools


dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)


if rank == 0:
    wandb.init(project="latent-transformer-imagenet", name="training-run-fsdp-simple")


model_name = "Cosmos-Tokenizer-DI16x16"
encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit').to(local_rank)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        ds = load_dataset("evanarlian/imagenet_1k_resized_256")
        self.dataset = ds[split]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
        ])

    def __getitem__(self, index):
        image_data = self.dataset[index]['image']
        condition = self.dataset[index]['label']
        image = self.transform(image_data)
        return image, condition

    def __len__(self):
        return len(self.dataset)

dataset = ImageDataset()


sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=512 // world_size,  # Adjust batch size per GPU
    shuffle=(sampler is None),
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    sampler=sampler
)

# --- Model and FSDP Wrapping ---
model = LatentModel(6, 768, 12).to(local_rank)  # Move to local_rank


model = FSDP(model,
             mixed_precision=torch.distributed.fsdp.MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                cast_forward_inputs=True
             ),
             device_id=torch.device(local_rank))


optimizer = optim.Adam(model.parameters(), lr=0.0004)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
max_grad_norm = 5

def loss_fn(logits, target):
    B, S, V = logits.shape  # batch, sequence length, vocab size
    logits = logits.view(-1, V)  # reshape to (batch*sequence, vocab)
    target = target.view(-1).long()  # reshape to (batch*sequence)
    ce_loss = F.cross_entropy(logits, target)
    return ce_loss


num_epochs = 50

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

if rank == 0:
    print('param count', count_params(model))

from tqdm import tqdm

for epoch in range(num_epochs):
    model.train()
    total = 0

    
    dataloader.sampler.set_epoch(epoch)

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0):
        images, condition = batch
        images = images.to(local_rank, non_blocking=True)
        images = images.to(torch.bfloat16)
        condition = condition.to(local_rank, non_blocking=True)
        condition = condition.to(torch.bfloat16)

        with torch.no_grad():
            indices, embeddings = encoder.encode(images)
            b, h, w = indices.shape
            indices = indices.reshape(b, h * w)  # tokens target
            target = indices

        optimizer.zero_grad()
        output = model(embeddings, condition)

        total_loss = loss_fn(output, target)

        if not torch.isnan(total_loss):
            total_loss.backward()
            optimizer.step()
            total += total_loss.item()

        if rank == 0:
            wandb.log({
                'total_loss': total_loss.item(),
            })

    avg_loss = total / len(dataloader)

   
    reduced_loss = torch.tensor(avg_loss).to(local_rank)
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    reduced_loss = reduced_loss / world_size

    scheduler.step(reduced_loss)

    if rank == 0:
        wandb.log({'avg_loss': reduced_loss.item()})
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {reduced_loss.item():.4f}")


if rank == 0:
    torch.save(model.module.state_dict(), 'latenttransformer_fsdp.pth')  # Save the original module's state_dict
    print("Training completed and model saved.")


dist.destroy_process_group()