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
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
from time import time
from PIL import Image
import numpy as np
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

def split_grid(pil_image, image_size):
    w, h = pil_image.size
    
    # Case 1: Perfect Grid (No padding)
    if w % image_size == 0 and h % image_size == 0:
        padding = 0
        cols = w // image_size
        rows = h // image_size
        
    # Case 2: Saved Grid (Has padding, usually 2px)
    # Formula: W = cols * size + (cols + 1) * padding
    # Solving for cols: cols = (W - padding) / (size + padding)
    else:
        padding = 2 # Standard save_image padding
        # Check if this padding theory fits width
        if (w - padding) % (image_size + padding) == 0:
            cols = (w - padding) // (image_size + padding)
            rows = (h - padding) // (image_size + padding)
        else:
            # Case 3: Just a random image -> Treat as single crop
            print(f"Warning: Input image size ({w}x{h}) does not match grid logic. Treating as single image.")
            return [center_crop_arr(pil_image, image_size)], 1, 1

    print(f"Detected Grid: {rows}x{cols} with padding={padding}")
    
    images = []
    for r in range(rows):
        for c in range(cols):
            # Calculate coordinates with padding
            x_start = padding + c * (image_size + padding)
            y_start = padding + r * (image_size + padding)
            
            crop = pil_image.crop((
                x_start,
                y_start,
                x_start + image_size,
                y_start + image_size
            ))
            images.append(crop)
            
    return images, rows, cols

def main(mode, args):
    # Setup PyTorch:
    if args.intermediates is not None and args.intermediates > args.num_sampling_steps:
        print(f"cannot have intermediates{args.intermediates} more than function evaluations:{args.num_sampling_steps}")
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
                reverse=args.reverse,
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

    if args.input_img:
        transform = transforms.Compose([            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True),
        ])

        grid = Image.open(args.input_img).convert("RGB")
        tiles, rows, cols = split_grid(grid, args.image_size)

        imgs = torch.stack(
            [transform(im) for im in tiles],
            dim=0
        ).to(device)

        with torch.no_grad():
            z = vae.encode(imgs).latent_dist.mode().mul_(0.18215)
        
        class_labels = torch.tensor([args.num_classes] * z.shape[0])
    elif args.input_latent:
        z = torch.load(args.input_latent, map_location=device)
        class_labels = torch.tensor([args.num_classes] * z.shape[0])        
    else:
        class_labels = torch.randint(args.num_classes, (args.num_samples,))
        # class_labels = torch.randint(1, (args.num_samples,))
        z = torch.randn(len(class_labels), 4, latent_size, latent_size, device=device)

    n = len(class_labels)    
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
    if args.input_img:
        latent_name = args.input_img.replace(".png", "_inverted.pt")
        torch.save(samples, latent_name)
        print(f"Saved latent to {latent_name}")
    with torch.no_grad():
        #print(f"generated image: {samples}")
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
        if args.input_img:
            process_name = args.input_img.replace(".png", "_inverted_intermediates.png")
        else:
            process_name = args.name.replace(".png", "_intermediates.png")
        save_image(intermediates, process_name, nrow=n, normalize=True, value_range=(-1, 1))
        print(f"Saved intermediates to {process_name}")
    
    if args.input_img:
        name = args.input_img.replace(".png", "_inverted.png")
        save_image(samples, name, nrow=cols, normalize=True, value_range=(-1, 1))
        print(f"Saved latent image to {name}")
    elif args.input_latent:
        name = args.input_latent.replace(".pt", "_reconstruction.png")
        save_image(samples, name, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"Saved final image reconstruction to {name}")
    else:
        save_image(samples, args.name, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"Saved final image to {args.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"
    
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="sample.png")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    parser.add_argument("--intermediates", type=int, default=None)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to generate (generation mode)"
    )

    group.add_argument(
        "--input-img",
        type=str,
        default=None,
        help="Path to a grid image (inversion mode)"
    )
    group.add_argument(
        "--input-latent",
        type=str,
        default=None,
        help="Path to a VAE latent to start flow"
    )

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    args = parser.parse_known_args()[0]
    if args.input_img:
        args.reverse = True
    print("args parsed")
    main(mode, args)
