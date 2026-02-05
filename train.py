# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using Hugging Face Accelerate.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import json

# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    # Accelerate handles process filtering, but we explicitly check main process for file logging
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args, loaded_state_dict=None):
    """
    Trains a new SiT model using Accelerate.
    """
    # Setup Accelerate:
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.wandb else None
    )
    device = accelerator.device

    # Setup seeds
    seed = args.global_seed + accelerator.process_index
    set_seed(seed)

    # Calculate local batch size based on number of processes
    if args.global_batch_size % accelerator.num_processes != 0:
        raise ValueError(f"Batch size {args.global_batch_size} must be divisible by number of processes {accelerator.num_processes}.")
    local_batch_size = int(args.global_batch_size // accelerator.num_processes)
    
    print(f"Starting process: {accelerator.process_index}, seed={seed}, world_size={accelerator.num_processes}. Local BS={local_batch_size}")

    # Setup an experiment folder:
    if accelerator.is_main_process:
        if args.ckpt is not None:            
            ckpt_path_abs = os.path.abspath(args.ckpt)
            checkpoint_dir = os.path.dirname(ckpt_path_abs) # .../checkpoints
            experiment_dir = os.path.dirname(checkpoint_dir) # .../00X-Experiment-Name
            
            experiment_name = os.path.basename(experiment_dir)
            logger = create_logger(experiment_dir)
            logger.info(f"Resuming experiment in existing directory: {experiment_dir}")
            wandb_init_kwargs = {
                "wandb": {
                    "entity": os.environ.get("ENTITY", ""),
                    "name": experiment_name,
                    "id": experiment_name,     # Force the run ID to match the folder name
                    "resume": "allow"          # Append to run if exists, else create new
                }
            }

        else:
            os.makedirs(args.results_dir, exist_ok=True)  
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-") 
            experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                            f"{args.path_type}-{args.prediction}-{args.loss_weight}"
            experiment_dir = f"{args.results_dir}/{experiment_name}" 
            checkpoint_dir = f"{experiment_dir}/checkpoints" 
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            logger = create_logger(experiment_dir)
            logger.info(f"Created new experiment directory: {experiment_dir}")
            
            wandb_init_kwargs = {
                "wandb": {
                    "entity": os.environ.get("ENTITY", ""),
                    "name": experiment_name
                }
            }

        if args.wandb:
            project = os.environ.get("PROJECT", "SiT")
            accelerator.init_trackers(
                project_name=project, 
                config=vars(args),
                init_kwargs=wandb_init_kwargs
            )
            
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        if args.ckpt is not None:
             ckpt_path_abs = os.path.abspath(args.ckpt)
             checkpoint_dir = os.path.dirname(ckpt_path_abs)
             experiment_dir = os.path.dirname(checkpoint_dir)
        else:
             checkpoint_dir = None
             experiment_dir = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    # EMA Model
    ema = deepcopy(model).to(device)  
    
    if args.ckpt is not None:
        if loaded_state_dict is not None:
            state_dict = loaded_state_dict
        else:
            state_dict = find_model(args.ckpt)
        ckpt_path = args.ckpt
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"]) 
        if accelerator.is_main_process:
            logger.info(f"Loaded {args.ckpt}")
    requires_grad(ema, False)
    
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    ) 
    transport_sampler = Sampler(transport)
    
    # Load VAE and freeze
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    requires_grad(vae, False)
        
    if args.mixed_precision == "bf16":
        vae.to(torch.bfloat16)
    elif args.mixed_precision == "fp16":
        vae.to(torch.float16)    

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Load optimizer state if resuming
    if args.ckpt is not None and "opt" in state_dict:
        opt.load_state_dict(state_dict["opt"])
    
    if args.latent_data:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])
        full_dataset = DatasetFolder(root=args.data_path, 
            loader=lambda x: torch.from_numpy(np.load(x)), 
            extensions=('.npy',),
            transform=transform)
    else:
        # Setup real image data:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        full_dataset = ImageFolder(args.data_path, transform=transform)
    
    if experiment_dir is not None:
        mapping_path = os.path.join(experiment_dir, "class_labels.json")
    else:
        mapping_path = None
    if args.ckpt and os.path.exists(mapping_path):
        if accelerator.is_main_process:
            logger.info(f"Loading class labels from {mapping_path}")
        with open(mapping_path, "r") as f:
            class_to_new_idx = json.load(f)
        target_classes = sorted(list(class_to_new_idx.keys()))
    else:
        target_classes = sorted(full_dataset.classes)[:args.num_classes]
        class_to_new_idx = {cls_name: i for i, cls_name in enumerate(target_classes)}
        
        if accelerator.is_main_process:
            logger.info(f"Filtering dataset: Keeping {len(target_classes)}/{len(full_dataset.classes)} classes.")
            logger.info(f"Class 0: {target_classes[0]}")
            logger.info(f"Class {len(target_classes)-1}: {target_classes[-1]}")
            
            with open(mapping_path, "w") as f:
                json.dump(class_to_new_idx, f, indent=4)
            logger.info(f"Saved class mapping to {mapping_path}")
            
    # Apply filtering to dataset
    new_samples = []
    # ImageFolder.samples is list of (path, class_index)
    for path, old_idx in full_dataset.samples:
        class_name = full_dataset.classes[old_idx]
        if class_name in target_classes:
            new_idx = class_to_new_idx[class_name]
            new_samples.append((path, new_idx))

    # Overwrite the dataset object internals
    full_dataset.samples = new_samples
    full_dataset.classes = target_classes
    full_dataset.class_to_idx = class_to_new_idx
    
    # Overwrite targets if they exist
    if hasattr(full_dataset, 'targets'):
        full_dataset.targets = [s[1] for s in new_samples]

    # DataLoader - Accelerate handles the sampler automatically
    loader = DataLoader(
        full_dataset,
        batch_size=local_batch_size, # Use calculated local batch size
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Prepare models, optimizer, and dataloader with Accelerate
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Initialize EMA weights with synced model weights
    update_ema(ema, accelerator.unwrap_model(model), decay=0) 
    model.train() 
    ema.eval()

    # Variables for monitoring/logging:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_epochs=0

    if args.ckpt is not None:
        try:
            ckpt_name = os.path.basename(args.ckpt)
            stats = ckpt_name.split('.')[0].split('_')
            train_steps = int(stats[1])
            start_epochs = int(stats[0])
            print(f"Resuming training from step: {train_steps}")
        except ValueError:
            print("Warning: Could not parse step count from checkpoint filename. Starting at step 0.")

    start_time = time()

    all_ys = torch.arange(0, args.num_classes, device=device)
    ys = torch.tensor_split(all_ys, accelerator.num_processes)[accelerator.process_index]
    use_cfg = args.cfg_scale > 1.0
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance for sampling:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([args.num_classes] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
        
    for epoch in range(start_epochs, args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
            
        for x, y in loader:
            # If using latent data, x is already latents.
            if not args.latent_data:
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            # If latent_data is True, x is passed directly to loss
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            
            # Optimization step
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
            
            update_ema(ema, accelerator.unwrap_model(model))

            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # Gather loss across processes for logging
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.reduce(avg_loss, reduction="mean") # Average across GPUs
                
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss.item():.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    if args.wandb:
                        accelerator.log(
                            {"train loss": avg_loss.item(), "train steps/sec": steps_per_sec},
                            step=train_steps
                        )
                
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save Checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{epoch}_{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                accelerator.wait_for_everyone()
            
            # Sampling
            if train_steps % args.sample_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    logger.info("Generating EMA samples...")
                
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode()
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    
                    if use_cfg: # remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    
                    decoded_chunks = []
                    decode_bs = 32
                    
                    for i in range(0, samples.shape[0], decode_bs):
                        batch_chunk = samples[i : i + decode_bs] / 0.18215
                        chunk_out = vae.decode(batch_chunk).sample
                        decoded_chunks.append(chunk_out)
                        
                    samples = torch.cat(decoded_chunks, dim=0)
                    
                    # Gather samples from all GPUs
                    # (B, 3, H, W) -> (World_Size * B, 3, H, W)
                    all_samples = accelerator.gather(samples)

                if accelerator.is_main_process:
                     if args.wandb:
                        wandb_utils.log_image(all_samples, train_steps)
                     logger.info("Generating EMA samples done.")
                
                accelerator.wait_for_everyone()

    model.eval()
    if accelerator.is_main_process:
        logger.info("Done!")
    
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--latent-data", action="store_true", default=False)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256) 
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose 'no', 'fp16' or 'bf16'")

    parse_transport_args(parser)
    args = parser.parse_args()
    loaded_state_dict = None

    if args.ckpt:
        print(f"Loading config from checkpoint: {args.ckpt}")
        loaded_state_dict = find_model(args.ckpt)
        ckpt_args = loaded_state_dict["args"]     
        fixed_args = ["wandb", "model", 
        "image_size", "num_classes", "prediction", "vae"]
        for k, v in vars(ckpt_args).items():
            if not hasattr(args, k):
                continue
            if k in fixed_args:
                print(f"  Using fixed arg {k} = {v}")
                setattr(args, k, v)
                continue
            if getattr(args, k) == getattr(ckpt_args, k):                 
                print(f"  Resuming arg: {k} = {v}")
            else:
                print(f"  Overriding arg: {k} = {getattr(args, k)} (Command Line)")

    main(args, loaded_state_dict)
