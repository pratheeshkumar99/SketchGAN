import torch
import argparse
import matplotlib.pyplot as plt
from models.unet_generator import UNetGenerator
from dataset_loader import create_dataloaders
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from tqdm import tqdm

# Unnormalize the tensor
def unnormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # Scale from [-1, 1] to [0, 1]
    return tensor


def calculate_metrics(generated_image, original_image):
    # Convert tensors to numpy arrays
    gen_img_np = generated_image.permute(1, 2, 0).cpu().numpy()
    org_img_np = original_image.permute(1, 2, 0).cpu().numpy()

    # Determine the smallest image dimension
    min_dim = min(gen_img_np.shape[0], gen_img_np.shape[1])

    # Set win_size based on the smaller dimension
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)

    # Ensure the minimum size is 3x3 to avoid errors
    if min_dim < 3:
        raise ValueError("Image size is too small for SSIM computation")

    # Calculate SSIM and PSNR with dynamic win_size and channel_axis for multichannel images
    ssim_value = ssim(gen_img_np, org_img_np, win_size=win_size, channel_axis=-1, data_range=gen_img_np.max() - gen_img_np.min())
    psnr_value = psnr(org_img_np, gen_img_np, data_range=gen_img_np.max() - gen_img_np.min())

    return ssim_value, psnr_value

# Evaluate the model on the test dataset
def evaluate_model(generator, test_loader, device):
    generator.eval()  # Set the generator to evaluation mode

    total_ssim = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for sketches, real_images in tqdm(test_loader, desc="Evaluating"):
            sketches = sketches.to(device)
            real_images = real_images.to(device)

            # Generate images from sketches
            generated_images = generator(sketches)

            # Unnormalize the images
            generated_images_unnorm = unnormalize(generated_images)
            real_images_unnorm = unnormalize(real_images)

            # Calculate metrics for each image in the batch
            for gen_img, org_img in zip(generated_images_unnorm, real_images_unnorm):
                ssim_value, psnr_value = calculate_metrics(gen_img, org_img)
                total_ssim += ssim_value
                total_psnr += psnr_value
                count += 1

    # Calculate average SSIM and PSNR over the dataset
    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count

    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f}")

    return avg_ssim, avg_psnr

if __name__ == "__main__":
    # Argument parser for passing arguments to the script
    parser = argparse.ArgumentParser()
    parser.add_argument('--photo_dir', type=str, required=True, help='Directory for real photos')
    parser.add_argument('--sketch_dir', type=str, required=True, help='Directory for sketch images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for loading the data')
    parser.add_argument('--generator_path', type=str,required=True, help='Path to the trained generator model')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} (GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # M1/M2 Macs with Metal support
        print(f"Using device: {device} (MPS)")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CPU)")

    # Load the data
    _, _, test_loader = create_dataloaders(args.sketch_dir, args.photo_dir, batch_size=args.batch_size)

    # Load the trained generator model
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(args.generator_path, map_location=device))
    generator.eval()

    # Evaluate the model on the test set
    evaluate_model(generator, test_loader, device)