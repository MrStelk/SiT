# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using Accelerate.
Subsequently saves a .npz file that can be used to compute FID.
"""
import torch
import torch.distributed as dist 
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import sys
import time  # <--- Added import
from accelerate import Accelerator
from accelerate.utils import set_seed

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    found_files = 0
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        file_path = f"{sample_dir}/{i:06d}.png"
        if os.path.exists(file_path):
            sample_pil = Image.open(file_path)
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
            found_files += 1
    
    if found_files < num:
        print(f"Warning: Expected {num} samples but found {found_files}.")

    samples = np.stack(samples)
    assert samples.shape == (found_files, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(mode, args):
    """
    Run sampling.
    """
    # Setup Accelerate
    accelerator = Accelerator()
    device = accelerator.device
    
    # Enable TF32 if requested
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup seeds
    seed = args.global_seed + accelerator.process_index
    set_seed(seed)
    
    if accelerator.is_main_process:
        print(f"Starting rank={accelerator.process_index}, seed={seed}, world_size={accelerator.num_processes}.")

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.image_size == 256, "512x512 models are not yet available for auto-download." 
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = True 

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    )
    model = model.to(device) 

    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    
    if isinstance(state_dict, dict):
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
            
    model.load_state_dict(state_dict)
    model.eval() 
    
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
        
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    
    if mode == "ODE":
        folder_name = f"{model_string_name}-{ckpt_string_name}-" \
                      f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"\
                      f"{mode}-{args.num_sampling_steps}-{args.sampling_method}"
    elif mode == "SDE":
        folder_name = f"{model_string_name}-{ckpt_string_name}-" \
                      f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"\
                      f"{mode}-{args.num_sampling_steps}-{args.sampling_method}-"\
                      f"{args.diffusion_form}-{args.last_step}-{args.last_step_size}"
                      
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    
    if accelerator.is_main_process:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    
    accelerator.wait_for_everyone()

    n = args.per_proc_batch_size
    global_batch_size = n * accelerator.num_processes
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    
    if accelerator.is_main_process:
        print(f"Total number of images that will be sampled: {total_samples}")

    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    iterations = int(math.ceil(samples_needed_this_gpu / n))
    
    pbar = range(iterations)
    pbar = tqdm(pbar) if accelerator.is_main_process else pbar
    
    # --- TIME MEASUREMENT START ---
    start_time = time.time()
    
    for i in pbar:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            model_fn = model.forward

        with torch.no_grad():
            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)

            samples = vae.decode(samples / 0.18215).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for j, sample in enumerate(samples):
            index = (i * global_batch_size) + accelerator.process_index + (j * accelerator.num_processes)
            if index < args.num_fid_samples:
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

    # --- TIME MEASUREMENT END ---
    accelerator.wait_for_everyone()
    end_time = time.time()
    elapsed_time = end_time - start_time

    if accelerator.is_main_process:
        print(f"Sampling finished in {elapsed_time:.2f} seconds.")
        print(f"Throughput: {args.num_fid_samples / elapsed_time:.2f} images/sec") # Uses requested samples for calc
        
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]
    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)

    args = parser.parse_known_args()[0]
    
    if args.ckpt:
        print(f"Loading config from: {args.ckpt}")
        loaded_state_dict = find_model(args.ckpt)
        if isinstance(loaded_state_dict, dict) and "args" in loaded_state_dict:
            ckpt_args = loaded_state_dict["args"]
            fixed = ["model", "image_size", "num_classes", "vae", "path_type", "prediction", "loss_weight"]
            for k, v in vars(ckpt_args).items():
                if k in fixed and hasattr(args, k):
                    print(f"  Forcing structural arg: {k} = {v}")
                    setattr(args, k, v)

    main(mode, args)
