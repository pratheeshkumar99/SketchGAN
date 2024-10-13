import torch
import random
import argparse
import matplotlib.pyplot as plt
from models.unet_generator import UNetGenerator  # Ensure proper imports from your structure
from dataset_loader import create_dataloaders  # Your dataloader creation method

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Random image selection from test data for visualization')
    parser.add_argument('--photo_dir', type=str, default="Data/raw/portraits",required=False, help='Path to photo directory')
    parser.add_argument('--sketch_dir', type=str,default="Data/raw/sketches", required=False, help='Path to sketch directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the dataloader')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test data split ratio')
    parser.add_argument('--val_split', type=float, default=0.5, help='Validation split from test set')
    parser.add_argument('--checkpoint_path', type=str, default="savemodels/best_generator.pth", help='path of the trained model')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the test data')
    parser.add_argument('--idx', type=int, default=0, help='Index of the batch')
    parser.add_argument('--example_idx', type=int, default=0, help='Index of the example')
    
    args = parser.parse_args()
    return args

def load_generator(checkpoint_path, device):
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def visualize_random_image(test_loader, generator, device, shuffle=True, idx=0, example_idx=0):
    if not shuffle:
        batch = test_loader[idx]  # Get the specified batch directly
    else:
        test_loader = list(test_loader)
        random.shuffle(test_loader)
        batch = test_loader[idx]

    sketches, real_images = batch

    # Use a fixed index if shuffle is False
    random_idx = example_idx if not shuffle else random.randint(0, len(sketches) - 1)

    random_sketch = sketches[random_idx].unsqueeze(0).to(device)
    random_real_image = real_images[random_idx]

    with torch.no_grad():
        generated_image = generator(random_sketch)

    def unnormalize(tensor):
        tensor = tensor * 0.5 + 0.5
        return tensor

    generated_image = unnormalize(generated_image.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    random_real_image = unnormalize(random_real_image).permute(1, 2, 0).cpu().numpy()
    random_sketch_image = unnormalize(random_sketch.squeeze(0)).permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(random_sketch_image); axs[0].axis('off'); axs[0].set_title("Original Sketch")
    axs[1].imshow(generated_image); axs[1].axis('off'); axs[1].set_title("Generated Image")
    axs[2].imshow(random_real_image); axs[2].axis('off'); axs[2].set_title("Original Real Image")

    plt.show()

def main():
    # Parse arguments
    args = parse_args()

    # Set up device: use GPU if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} (GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # M1/M2 Macs with Metal support
        print(f"Using device: {device} (MPS)")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CPU)")



    # Create test dataloader
    _, _, test_loader = create_dataloaders(args.sketch_dir, args.photo_dir, batch_size=args.batch_size)

    # Load the trained generator
    generator = load_generator(args.checkpoint_path, device)


    # Visualize a random image from the test set
    visualize_random_image(test_loader, generator, device,shuffle=args.shuffle,idx=args.idx,example_idx=args.example_idx)


if __name__ == "__main__":
    main()