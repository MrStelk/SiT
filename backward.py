# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
import os
from PIL import Image
import numpy as np
from time import time
from torchvision import transforms

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

def main(mode, args):
	if args.intermediates is not None and args.intermediates > args.num_sampling_steps:
        print(f"cannot have intermediates{args.intermediates} more than function evaluations:{args.num_sampling_steps}")
    assert os.path.exists(args.input_dir), f"Input directory {args.input_dir} not found"
    # Setup PyTorch:
    print("Setting up...")
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = True

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    print(args.model)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path,sample=True)
    model.load_state_dict(state_dict)
    print("Loaded Model")
    model.eval()  # important!

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=args.sampling_method,
        num_steps=args.num_sampling_steps,
        atol=args.atol,
        rtol=args.rtol,
        reverse=True,
    )

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    

    image_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_paths) == 0:
        print("No images found in input directory.")
        return

    print(f"Found {len(image_paths)} images. Encoding...")

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    latents_list = []
    
    # Process images (simple batching could be added here, currently loop)
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dim
        
        with torch.no_grad():
            # For inversion, we usually use the deterministic mode (mean)
            # Scaling is crucial: x * 0.18215
            latent = vae.encode(img_tensor).latent_dist.mode().mul_(0.18215)
            latents_list.append(latent)

    # Concat all latents: Shape (N, 4, H, W)
    z_0 = torch.cat(latents_list, dim=0)

    # Labels to condition the model with (feel free to change):
    # sample.py
    class_labels = torch.randint(args.num_classes, (len(images),))
    # class_labels = [0, 10, 20, 30, 40, 50, 60, 70]
    
    # Create sampling noise:
    n = len(class_labels)
    z=[]
    for x in images:
		with torch.no_grad():
		    z.append(vae.encode(x).latent_dist.sample().mul_(0.18215))
    z = torch.tensor(z, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    start_time = time()
    all_samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)
    samples = all_samples[-1]
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    with torch.no_grad():
        samples = vae.decode(samples / 0.18215).sample
    
    if args.intermediates:
        stride = args.num_sampling_steps//args.intermediates
        intermediate_samples = all_samples[::stride]
        intermediates, _ = intermediate_samples.chunk(2,dim=1)
        intermediates = intermediates.reshape(-1, 4, latent_size, latent_size)
        with torch.no_grad():
            intermediates = vae.decode(intermediates/0.18215).sample
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    if args.intermediates:
        process_name = args.name.replace(".png", "_intermediates.png")
        save_image(intermediates, process_name, nrow=args.num_samples, normalize=True, value_range=(-1, 1))
        print(f"Saved intermediates to {process_name}")
    save_image(samples, args.name, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved final image to {args.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--intermediates", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    parse_ode_args(parser)
    args = parser.parse_known_args()[0]
    print("args parsed")
    main(mode, args)
