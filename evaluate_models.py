import os
import torch
import json
import argparse
from models.unet_generator import UNetGenerator
from dataset_loader import create_dataloaders
from tqdm import tqdm
from evaluate import evaluate_model  # Correct import statement

def parse_params_from_name(folder_name):
    parts = folder_name.split('_')
    params = {
        'lr': float(parts[1]),
        'batch_size': int(parts[3]),
        'l1_weight': int(parts[5]),
        'dropout_rate': float(parts[7])
    }
    return params

def load_model(model_path, device):
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_all_models(root_dir, sketch_dir, photo_dir):
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    for subdir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, subdir, 'best_generator.pth')
        if os.path.isfile(model_path):
            params = parse_params_from_name(subdir)
            print(f"Evaluating {subdir} with params: {params}")
            _, _, test_loader = create_dataloaders(sketch_dir, photo_dir, batch_size=params['batch_size'])
            generator = load_model(model_path, device)
            ssim, psnr = evaluate_model(generator, test_loader, device)  # Directly use the existing function
            results[subdir] = {
                'SSIM': ssim,
                'PSNR': psnr,
                'params': params
            }

    with open('model_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GAN models on a test dataset")
    parser.add_argument('--root_dir', type=str, default="savemodels", help='Directory containing model subdirectories')
    parser.add_argument('--sketch_dir', type=str, default="Data/raw/sketches", help='Path to the sketches directory.')
    parser.add_argument('--photo_dir', type=str, default="Data/raw/portraits", help='Path to the photos directory.')
    args = parser.parse_args()

    evaluate_all_models(args.root_dir, args.sketch_dir, args.photo_dir)


"""
Grid Search Hyperparameter Tuning Results:
Summary of Top Performing Models Based on Both SSIM and PSNR:
| Model ID | Learning Rate | Batch Size | L1 Weight | Dropout Rate | SSIM  | PSNR  | Overview                                      |
|----------|---------------|------------|-----------|--------------|-------|-------|----------------------------------------------|
| 1        | 0.005         | 8          | 100       | 0.1          | 0.6858| 18.06 | Best overall, excellent balance and reduction|
| 2        | 0.001         | 8          | 10        | 0.5          | 0.6259| 18.40 | Strong in both metrics, high image quality   |
| 3        | 0.005         | 8          | 10        | 0.5          | 0.6549| 17.11 | Good image similarity with reasonable noise  |
| 4        | 0.001         | 8          | 100       | 0.1          | 0.6055| 18.00 | Well-balanced, effective in fidelity & noise |
| 5        | 0.005         | 8          | 50        | 0.1          | 0.6569| 17.25 | Strong performance, good overall quality     |

Top Five Based on SSIM (Structural Similarity Index):

| Rank | Learning Rate | Batch Size | L1 Weight | Dropout Rate | SSIM  | PSNR  |
|------|---------------|------------|-----------|--------------|-------|-------|
| 1    | 0.005         | 8          | 200       | 0.3          | 0.7025| 17.85 |
| 2    | 0.005         | 8          | 100       | 0.1          | 0.6858| 18.06 |
| 3    | 0.005         | 8          | 10        | 0.5          | 0.6549| 17.11 |
| 4    | 0.005         | 8          | 50        | 0.3          | 0.6642| 17.56 |
| 5    | 0.005         | 8          | 10        | 0.1          | 0.6569| 17.25 |
Top Five Based on PSNR (Peak Signal-to-Noise Ratio):
| Rank | Learning Rate | Batch Size | L1 Weight | Dropout Rate | SSIM  | PSNR  |
|------|---------------|------------|-----------|--------------|-------|-------|
| 1    | 0.001         | 8          | 10        | 0.5          | 0.6259| 18.40 |
| 2    | 0.005         | 8          | 100       | 0.1          | 0.6858| 18.06 |
| 3    | 0.001         | 8          | 100       | 0.1          | 0.6055| 18.00 |
| 4    | 0.0002        | 16         | 10        | 0.5          | 0.6031| 18.01 |
| 5    | 0.0005        | 8          | 10        | 0.3          | 0.5591| 18.24 |
"""