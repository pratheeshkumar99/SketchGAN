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



#grid search for hyperparameter tuning Results:

# summary of the models that perform well in both SSIM and PSNR:

    # 	1.	lr_0.005_batch_8_l1_100_dropout_0.1
	# •	SSIM: 0.6858 (Top in SSIM)
	# •	PSNR: 18.06 (Top in PSNR)
	# •	Params: lr=0.005, batch_size=8, l1_weight=100, dropout_rate=0.1
	# •	Overview: Best overall in both metrics, showing excellent balance between similarity and noise reduction.
	# 2.	lr_0.001_batch_8_l1_10_dropout_0.5
	# •	SSIM: 0.6259
	# •	PSNR: 18.40 (Top in PSNR)
	# •	Params: lr=0.001, batch_size=8, l1_weight=10, dropout_rate=0.5
	# •	Overview: Strong in both metrics, especially in PSNR, making it excellent for maintaining high image quality.
	# 3.	lr_0.005_batch_8_l1_10_dropout_0.5
	# •	SSIM: 0.6549
	# •	PSNR: 17.11
	# •	Params: lr=0.005, batch_size=8, l1_weight=10, dropout_rate=0.5
	# •	Overview: High SSIM score and decent PSNR, indicating good image similarity with reasonable noise reduction.
	# 4.	lr_0.001_batch_8_l1_100_dropout_0.1
	# •	SSIM: 0.6055
	# •	PSNR: 18.00 (Top in PSNR)
	# •	Params: lr=0.001, batch_size=8, l1_weight=100, dropout_rate=0.1
	# •	Overview: Well-balanced with high scores in both metrics, demonstrating effectiveness in both image fidelity and noise handling.
	# 5.	lr_0.005_batch_8_l1_50_dropout_0.1
	# •	SSIM: 0.6569
	# •	PSNR: 17.25
	# •	Params: lr=0.005, batch_size=8, l1_weight=50, dropout_rate=0.1
	# •	Overview: Strong performance in both metrics, indicating good overall image quality and similarity.



    # Top Five Based on SSIM (Structural Similarity Index)

	# 1.	lr_0.005_batch_8_l1_200_dropout_0.3
	# •	SSIM: 0.7025
	# •	PSNR: 17.85
	# •	Params: lr=0.005, batch_size=8, l1_weight=200, dropout_rate=0.3
	# 2.	lr_0.005_batch_8_l1_100_dropout_0.1
	# •	SSIM: 0.6858
	# •	PSNR: 18.06
	# •	Params: lr=0.005, batch_size=8, l1_weight=100, dropout_rate=0.1
	# 3.	lr_0.005_batch_8_l1_10_dropout_0.5
	# •	SSIM: 0.6549
	# •	PSNR: 17.11
	# •	Params: lr=0.005, batch_size=8, l1_weight=10, dropout_rate=0.5
	# 4.	lr_0.005_batch_8_l1_50_dropout_0.3
	# •	SSIM: 0.6642
	# •	PSNR: 17.56
	# •	Params: lr=0.005, batch_size=8, l1_weight=50, dropout_rate=0.3
	# 5.	lr_0.005_batch_8_l1_10_dropout_0.1
	# •	SSIM: 0.6569
	# •	PSNR: 17.25
	# •	Params: lr=0.005, batch_size=8, l1_weight=10, dropout_rate=0.1





# Top Five Based on PSNR (Peak Signal-to-Noise Ratio)

# 	1.	lr_0.001_batch_8_l1_10_dropout_0.5
# 	•	SSIM: 0.6259
# 	•	PSNR: 18.40
# 	•	Params: lr=0.001, batch_size=8, l1_weight=10, dropout_rate=0.5
# 	2.	lr_0.005_batch_8_l1_100_dropout_0.1
# 	•	SSIM: 0.6858
# 	•	PSNR: 18.06
# 	•	Params: lr=0.005, batch_size=8, l1_weight=100, dropout_rate=0.1
# 	3.	lr_0.001_batch_8_l1_100_dropout_0.1
# 	•	SSIM: 0.6055
# 	•	PSNR: 18.00
# 	•	Params: lr=0.001, batch_size=8, l1_weight=100, dropout_rate=0.1
# 	4.	lr_0.0002_batch_16_l1_10_dropout_0.5
# 	•	SSIM: 0.6031
# 	•	PSNR: 18.01
# 	•	Params: lr=0.0002, batch_size=16, l1_weight=10, dropout_rate=0.5
# 	5.	lr_0.0005_batch_8_l1_10_dropout_0.3
# 	•	SSIM: 0.5591
# 	•	PSNR: 18.24
# 	•	Params: lr=0.0005, batch_size=8, l1_weight=10, dropout_rate=0.3
