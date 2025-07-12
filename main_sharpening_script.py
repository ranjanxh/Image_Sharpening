import os
import shutil
import torch
import gc 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image, ImageDraw, ImageFont
from natsort import natsorted
import glob
import numpy as np 
import math
import warnings
from tqdm import tqdm
import time
from runpy import run_path 
from copy import deepcopy 


warnings.filterwarnings("ignore", category=UserWarning, module="skimage.metrics")

# --- Component 1: Local PC Environment Setup & Prerequisites ---
print("--- Step 1: Local PC Environment Setup & Prerequisites ---")


local_project_root = r"C:\Users\ASUS\OneDrive\Documents\sharpen_kd"
print(f"Local project root set to: {local_project_root}")
os.chdir(local_project_root) # Ensure current working directory is the project root


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print("PYTORCH_CUDA_ALLOC_CONF set to 'expandable_segments:True'.")


if torch.cuda.is_available():
    print(f"GPU is available! Using: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True 
else:
    print("GPU not found. Using CPU. Training will be significantly slower.")
    device = torch.device("cpu")


restormer_repo_path_abs = os.path.join(local_project_root, 'Restormer')
if restormer_repo_path_abs not in sys.path:
    sys.path.insert(0, restormer_repo_path_abs)
    print(f"Added '{restormer_repo_path_abs}' to sys.path.")
basicsr_path_abs = os.path.join(restormer_repo_path_abs, 'basicsr')
if basicsr_path_abs not in sys.path:
    sys.path.insert(0, basicsr_path_abs)
    print(f"Added '{basicsr_path_abs}' to sys.path.")



if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


# Download pre-trained Restormer weights for Motion Deblurring
task = 'Motion_Deblurring'
pretrained_model_dir = os.path.join(local_project_root, 'Restormer', task, 'pretrained_models')
os.makedirs(pretrained_model_dir, exist_ok=True)

weights_filename = "motion_deblurring.pth"
weights_path = os.path.join(pretrained_model_dir, weights_filename)

if os.path.exists(weights_path):
    print(f"Weights already exist at {weights_path}.")
else:
    print(f"Weights not found at {weights_path}.")


# Prepare directories for dataset and model checkpoints
print("\nCreating project directories...")

data_root_dir = os.path.join(local_project_root, 'GoPro_dataset')
model_checkpoints_dir = os.path.join(local_project_root, 'model_checkpoints')

os.makedirs(model_checkpoints_dir, exist_ok=True)

os.makedirs(os.path.join(data_root_dir, 'train', 'blurred'), exist_ok=True)
os.makedirs(os.path.join(data_root_dir, 'train', 'sharp'), exist_ok=True)
os.makedirs(os.path.join(data_root_dir, 'test', 'blurred'), exist_ok=True)
os.makedirs(os.path.join(data_root_dir, 'test', 'sharp'), exist_ok=True)


print(f"Dataset root directory created/ensured at: {data_root_dir}")
print(f"Model checkpoints directory created/ensured at: {model_checkpoints_dir}")

print("\n--- Environment setup complete! ---")



print("\n" + "="*50)
print("--- Utility Functions: PSNR and SSIM Calculation ---")

def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_ssim_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def calculate_ssim_torch(img1_batch, img2_batch, data_range=1.0, window_size=11):
    """
    Calculates SSIM for a batch of images using pure PyTorch. Inputs are B, C, H, W tensors in [0, data_range] range.
    """
    
    img1_batch = img1_batch.float()
    img2_batch = img2_batch.float()

    
    if data_range != 1.0:
        img1_batch = img1_batch / data_range
        img2_batch = img2_batch / data_range

    channel = img1_batch.size(1)
    window = create_ssim_window(window_size, channel).to(img1_batch.device) # Move window to device

    mu1 = F.conv2d(img1_batch, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2_batch, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1_batch * img1_batch, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2_batch * img2_batch, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1_batch * img2_batch, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01 * 1.0)**2 
    C2 = (0.03 * 1.0)**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12) # Add epsilon for stability

    
    return ssim_map.mean()

# PSNR Calculation Function 
def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculates PSNR (Peak Signal-to-Noise Ratio) between two PyTorch tensors. Inputs are expected to be C, H, W tensors in [0, data_range] range.
    data_range: Maximum possible pixel value of the image (e.g., 1.0 for [0,1] images)
    """
    
    img1 = img1.detach().cpu().float()
    img2 = img2.detach().cpu().float()


    mse = torch.mean((img1 - img2) ** 2)

    
    if mse == 0:
        return float('inf')

    psnr_val = 10 * math.log10(data_range ** 2 / mse.item())
    return psnr_val

print("PSNR and SSIM calculation functions defined.")



def evaluate_model(model, data_loader, device, checkpoint_path=None, save_output_images=False, output_dir=None):
    if not isinstance(model, nn.Module):
        print(f"ERROR: Invalid model provided to evaluate_model. Expected nn.Module, got {type(model)}. Skipping evaluation.")
        return 0.0, 0.0

    model.eval() 
    total_ssim = 0
    total_psnr = 0
    num_samples = 0

    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Model loaded successfully from checkpoint.")
        except FileNotFoundError:
            print(f"ERROR: Checkpoint file not found at {checkpoint_path}. Skipping evaluation.")
            return 0.0, 0.0
        except RuntimeError as e:
            print(f"ERROR: Failed to load checkpoint due to model architecture mismatch or corrupted file: {e}. Skipping evaluation.")
            return 0.0, 0.0
        except Exception as e:
            print(f"ERROR loading checkpoint {checkpoint_path}: {e}. Skipping evaluation.")
            return 0.0, 0.0

    print("  Starting evaluation on test set...")
    with torch.no_grad():
        if not data_loader or len(data_loader) == 0:
            print("  Warning: Data loader is empty. Cannot perform evaluation.")
            return 0.0, 0.0

        for blurry_images, sharp_images, blurry_paths in tqdm(data_loader, desc="Evaluating"):
            blurry_images = blurry_images.to(device)
            sharp_images = sharp_images.to(device)

            
            model_output, *rest = model(blurry_images)
            model_output_clamped = model_output.clamp(0, 1)

            for i in range(model_output_clamped.shape[0]):
                current_ssim = calculate_ssim_torch(model_output_clamped[i].unsqueeze(0), sharp_images[i].unsqueeze(0))
                current_psnr = calculate_psnr(model_output_clamped[i], sharp_images[i], data_range=1.0)
                total_ssim += current_ssim.item()
                total_psnr += current_psnr
                num_samples += 1

                if save_output_images and output_dir:
                    original_blurry_filename = os.path.basename(blurry_paths[i])
                    output_filename = original_blurry_filename.replace('.', '_sharpened.')
                    output_path = os.path.join(output_dir, output_filename)
                    sharpened_pil_image = transforms.ToPILImage()(model_output_clamped[i].cpu())
                    sharpened_pil_image.save(output_path)

    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
    print(f"\n  Evaluation SSIM (Test Set): {avg_ssim:.4f}")
    print(f"  Evaluation PSNR (Test Set): {avg_psnr:.4f} dB")
    return avg_ssim, avg_psnr

print("Evaluation function defined.")
print("--- Utility functions setup complete! ---")


# Data Preparation & Loading
print("\n" + "="*50)
print("--- Step 2: Data Preparation & Loading ---")

class DataConfig:
    def __init__(self):
        
        self.dataset_root = os.path.join(local_project_root, 'GoPro_dataset')

        self.train_blurry_dir = os.path.join(self.dataset_root, 'train', 'blurred')
        self.train_sharp_dir = os.path.join(self.dataset_root, 'train', 'sharp')
        self.test_blurry_dir = os.path.join(self.dataset_root, 'test', 'blurred')
        self.test_sharp_dir = os.path.join(self.dataset_root, 'test', 'sharp')

        
        self.output_image_dir = os.path.join(local_project_root, 'inference_samples', 'sharpened_test_outputs')
        os.makedirs(self.output_image_dir, exist_ok=True)

        self.image_size = (512, 512)
        self.batch_size = 1
        self.num_channels = 3
        self.num_workers = 0


data_config = DataConfig()
print(f"Dataset root path: {data_config.dataset_root}")
print(f"Training image size (for random crops) set to: {data_config.image_size[0]}x{data_config.image_size[1]}")
print(f"Training batch size set to: {data_config.batch_size}")
print(f"Sharpened output images will be saved to: {data_config.output_image_dir}")
print("Training with high resolutions can be memory-intensive. Current target is "
      f"{data_config.image_size[0]}x{data_config.image_size[1]} for memory efficiency. "
      "If OOM occurs, consider further gradient accumulation.")


print("\n--- Dataset Requirements Reminder (Component 2) ---")
print(f"Ensure the directory '{data_config.test_blurry_dir}' contains over 100 images for testing.")
print("The dataset should include diverse categories: text, nature, people, animals, and games.")
print("Ensure your blurry images simulate video conferencing conditions (e.g., by downscaling and upscaling ground truth images using bicubic/bilinear methods to generate blurry/compressed inputs). This script reads existing blurry/sharp pairs; it does not generate these varied blurry images. Ensure your data preparation pipeline does this beforehand.")


class CustomToTensor(object):
    """Convert a PIL Image or numpy.ndarray to a tensor. Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)

        img_np = np.array(pic, dtype=np.uint8)
        if img_np.ndim < 3:
            img_np = np.expand_dims(img_np, axis=-1)
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1)))
        return img_tensor.float().div(255)


class GoProSharpeningDataset(Dataset):
    def __init__(self, blurry_dir, sharp_dir, train=True, target_size=(512, 512)):
        self.blurry_dir = blurry_dir
        self.sharp_dir = sharp_dir
        self.train = train
        self.target_size = target_size

        self.blurry_image_paths = natsorted(glob.glob(os.path.join(blurry_dir, '*.*')))
        self.sharp_image_paths = natsorted(glob.glob(os.path.join(sharp_dir, '*.*')))

        if not self.blurry_image_paths or not self.sharp_image_paths:
            raise RuntimeError(f"No images found in {blurry_dir} or {sharp_dir}. "
                               f"Please check your dataset paths and ensure images are present.")

        self.image_pairs = []
        for blurry_path in self.blurry_image_paths:
            base_name_blurry = os.path.basename(blurry_path).split('.')[0]

            matched_sharp_path = None
            for sharp_path in self.sharp_image_paths:
                base_name_sharp = os.path.basename(sharp_path).split('.')[0]
                if base_name_blurry.replace('_blurred', '').replace('_blur', '') == base_name_sharp.replace('_sharp', ''):
                    matched_sharp_path = sharp_path
                    break

            if matched_sharp_path:
                self.image_pairs.append((blurry_path, matched_sharp_path))
            else:
                print(f"Could not find matching sharp image for blurry: {blurry_path}. Skipping.")

        if not self.image_pairs:
            raise RuntimeError("No matching blurry-sharp image pairs could be formed. "
                               "Check your file naming conventions and data structure.")

        if len(self.blurry_image_paths) != len(self.sharp_image_paths):
            print(f"Number of blurry images ({len(self.blurry_image_paths)}) "
                  f"does not match number of sharp images ({len(self.sharp_image_paths)}). "
                  f"Using {len(self.image_pairs)} matched pairs.")

        # Data Augmentation Transforms 
        
        self.base_transforms = transforms.Compose([
            transforms.Resize(self.target_size, transforms.InterpolationMode.BICUBIC),
            CustomToTensor()
        ])

        self.train_additional_transforms = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])


    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        blurry_path, sharp_path = self.image_pairs[idx]

        blurry_image = Image.open(blurry_path).convert('RGB')
        sharp_image = Image.open(sharp_path).convert('RGB')

        if self.train:
            
            if blurry_image.size[0] < self.target_size[1] or blurry_image.size[1] < self.target_size[0]:
                blurry_image = transforms.functional.resize(blurry_image, self.target_size, transforms.InterpolationMode.BICUBIC)
                sharp_image = transforms.functional.resize(sharp_image, self.target_size, transforms.InterpolationMode.BICUBIC)

            
            i, j, h, w = transforms.RandomCrop.get_params(blurry_image, output_size=self.target_size)
            blurry_image = transforms.functional.crop(blurry_image, i, j, h, w)
            sharp_image = transforms.functional.crop(sharp_image, i, j, h, w)

            
            if torch.rand(1) < 0.5:
                blurry_image = transforms.functional.hflip(blurry_image)
                sharp_image = transforms.functional.hflip(sharp_image)

    
            blurry_image_tensor = self.base_transforms(blurry_image)
            sharp_image_tensor = self.base_transforms(sharp_image)

            
            blurry_image_tensor = self.train_additional_transforms(blurry_image_tensor)


        else: 
            blurry_image_tensor = self.base_transforms(blurry_image)
            sharp_image_tensor = self.base_transforms(sharp_image)

        return blurry_image_tensor, sharp_image_tensor, blurry_path


train_loader = None
test_loader = None

try:
    print("\nInitializing training dataset and DataLoader...")
    train_dataset = GoProSharpeningDataset(
        blurry_dir=data_config.train_blurry_dir,
        sharp_dir=data_config.train_sharp_dir,
        train=True,
        target_size=data_config.image_size
    )
    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config.batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
            pin_memory=True
        )
        print(f"Training data loader created with {len(train_dataset)} samples.")
        print(f"Number of training batches: {len(train_loader)}")
    else:
        print(f"No training samples found in {data_config.train_blurry_dir} or {data_config.train_sharp_dir}.")

except RuntimeError as e:
    print(f"\nERROR: Could not initialize training dataset. {e}")
    print(f"Expected train blurry dir: {data_config.train_blurry_dir}")
    print(f"Expected train sharp dir: {data_config.train_sharp_dir}")
    train_loader = None

try:
    print("\nInitializing test dataset and DataLoader...")
    test_dataset = GoProSharpeningDataset(
        blurry_dir=data_config.test_blurry_dir,
        sharp_dir=data_config.test_sharp_dir,
        train=False,
        target_size=data_config.image_size
    )
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=True
        )
        print(f"Test data loader created with {len(test_dataset)} samples.")
        print(f"Number of test batches: {len(test_loader)}")
    else:
        print(f"No test samples found in {data_config.test_blurry_dir} or {data_config.test_sharp_dir}.")

except RuntimeError as e:
    print(f"\nERROR: Could not initialize test dataset. {e}")
    print(f"Expected test blurry dir: {data_config.test_blurry_dir}")
    print(f"Expected test sharp dir: {data_config.test_sharp_dir}")
    test_loader = None

print("\n--- Data Preparation & Loading complete! ---")
if train_loader and test_loader:
    try:
        sample_blurry_batch, sample_sharp_batch, _ = next(iter(train_loader))
        print(f"Example: Blurry batch shape -> {sample_blurry_batch.shape}")
        print(f"Example: Sharp batch shape -> {sample_sharp_batch.shape}")
    except StopIteration:
        print("Train loader is empty. Dummy data will be used for training if needed.")
    except Exception as e:
        print(f"Could not get sample batch after DataLoader initialization: {e}")
else:
    print("Data loading failed for one or both datasets. Please check error messages above.")


# Teacher Model: Restormer 
print("\n" + "="*50)
print("--- Step 3: Teacher Model: Restormer ---")

_task_name = 'Motion_Deblurring'

_pretrained_model_dir_relative_to_restormer = os.path.join('Restormer', _task_name, 'pretrained_models')
_weights_path_in_restormer_dir = os.path.join(_pretrained_model_dir_relative_to_restormer, 'motion_deblurring.pth')

_restormer_arch_path = os.path.join('Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py')



try:
    from runpy import run_path
    load_arch = run_path(_restormer_arch_path)
    TeacherModelRaw = load_arch['Restormer']
    print("Restormer (TeacherModelRaw) imported successfully via run_path.")
except Exception as e:
    print(f"ERROR: Could not import Restormer architecture using run_path. Error: {e}")
    class TeacherModelRaw(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, x): return x
    TeacherModelRaw = TeacherModelRaw


class TeacherModelConfig:
    def __init__(self):
        self.in_channels = 3
        self.out_channels = 3
        self.model_parameters = {
            'inp_channels': self.in_channels,
            'out_channels': self.out_channels,
            'dim':48, 
            'num_blocks':[4,6,6,8],
            'num_refinement_blocks':4,
            'heads':[1,2,4,8],
            'ffn_expansion_factor':2.66,
            'bias':False,
            'LayerNorm_type':'WithBias',
            'dual_pixel_task':False
        }
        self.img_multiple_of = 8

teacher_model_config = TeacherModelConfig()
print(f"Teacher model configuration: {teacher_model_config.__dict__}")

class RestormerTeacherWrapper(nn.Module):
    def __init__(self, config, TeacherModelRaw, weights_path, student_base_channels):
        super(RestormerTeacherWrapper, self).__init__()
        self.config = config
        self.weights_path = weights_path

        self.restormer = TeacherModelRaw(**config.model_parameters)

        print(f"Loading pre-trained Restormer weights from {self.weights_path}...")
        try:
            state_dict = torch.load(self.weights_path, map_location=device)
            if 'params' in state_dict:
                state_dict = state_dict['params']

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.restormer.load_state_dict(new_state_dict)
            print("Pre-trained Teacher Model weights loaded successfully.")
        except FileNotFoundError:
            print(f"Pre-trained Restormer weights not found at {self.weights_path}. Teacher model will use random initialization.")
        except Exception as e:
            print(f"ERROR loading Restormer weights from {self.weights_path}: {e}. Teacher model might be uninitialized or incorrectly loaded.")

        self.restormer.eval()
        self.restormer.to(device)
        print(f"Restormer Teacher model moved to {device} and set to evaluation mode.")
        print("\n--- DEBUG: Restormer Model Structure ---")
        print(self.restormer)
        print("---------------------------------------")


        
        self.proj_e2 = None
        self.proj_e4 = None
        self.proj_b = None

       
        self.teacher_features = {
            'e2': None,
            'e4': None,
            'b': None
        }

        def get_activation(name, store_key):
            def hook(model, input, output):
                self.teacher_features[store_key] = output
            return hook

        # Dynamically find and register hooks based on common Restormer module names
        if hasattr(self.restormer, 'encoder_level2'):
            self.restormer.encoder_level2.register_forward_hook(get_activation('encoder_level2', 'e2'))
            self.proj_e2 = nn.Conv2d(96, student_base_channels * 2, kernel_size=1).to(device)
            print("Hooked 'encoder_level2' for feature 'e2'.")
        else:
            print("Teacher model has no 'encoder_level2' attribute. Feature distillation for e2 will be skipped.")

        if hasattr(self.restormer, 'encoder_level4'):
            self.restormer.encoder_level4.register_forward_hook(get_activation('encoder_level4', 'e4'))
            self.proj_e4 = nn.Conv2d(384, student_base_channels * 8, kernel_size=1).to(device)
            print("Hooked 'encoder_level4' for feature 'e4'.")
        else:
            print("Teacher model has no 'encoder_level4' attribute. Feature distillation for e4 will be skipped.")

        if hasattr(self.restormer, 'latent'):
            self.restormer.latent.register_forward_hook(get_activation('latent', 'b'))
            self.proj_b = nn.Conv2d(384, student_base_channels * 16, kernel_size=1).to(device)
            print("Hooked 'latent' for feature 'b'.")
        else:
            print("Teacher model has no 'latent' (bottleneck) attribute. Feature distillation for bottleneck will be skipped.")
        print("Forward hooks registration attempt complete.")

    def forward(self, x):
        h_orig, w_orig = x.shape[2], x.shape[3]
        H, W = ((h_orig + self.config.img_multiple_of - 1) // self.config.img_multiple_of) * self.config.img_multiple_of, \
               ((w_orig + self.config.img_multiple_of - 1) // self.config.img_multiple_of) * self.config.img_multiple_of
        padh = H - h_orig
        padw = W - w_orig
        x_padded = F.pad(x, (0, padw, 0, padh), 'reflect')

        # Forward pass through Restormer. Hooks will populate self.teacher_features
        restored_output_padded = self.restormer(x_padded)
        restored_output = restored_output_padded[:, :, :h_orig, :w_orig]

        # Get captured features and project them, handling cases where a feature might be None
        teacher_features_e2_raw = self.teacher_features['e2']
        teacher_features_e4_raw = self.teacher_features['e4']
        teacher_features_b_raw = self.teacher_features['b']

        teacher_features_e2 = self.proj_e2(teacher_features_e2_raw) if teacher_e2_feat_raw is not None and self.proj_e2 is not None else None
        teacher_features_e4 = self.proj_e4(teacher_features_e4_raw) if teacher_e4_feat_raw is not None and self.proj_e4 is not None else None
        teacher_features_b = self.proj_b(teacher_features_b_raw) if teacher_b_feat_raw is not None and self.proj_b is not None else None

        self.teacher_features = {
            'e2': None,
            'e4': None,
            'b': None
        }


        return restored_output, teacher_features_e2, teacher_features_e4, teacher_features_b

teacher_model = None
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("Cleared CUDA cache and ran garbage collection before Teacher Model instantiation.")

    class StudentModelConfigTemp: 
        def __init__(self):
            self.base_channels = 256
    student_model_config_temp = StudentModelConfigTemp()

    
    teacher_model = RestormerTeacherWrapper(teacher_model_config, TeacherModelRaw, _weights_path_in_restormer_dir, student_base_channels=student_model_config_temp.base_channels)
    print("\nRestormer Teacher model successfully initialized and loaded (or with warnings).")

    print("\n--- Verifying Teacher Model Performance (Initial Check) ---")
    dummy_input_for_teacher = torch.randn(1, 3, data_config.image_size[0], data_config.image_size[1]).to(device)
    dummy_sharp_for_teacher = torch.randn(1, 3, data_config.image_size[0], data_config.image_size[1]).to(device)

    with torch.no_grad():

        dummy_teacher_output, dummy_teacher_e2_feat, dummy_teacher_e4_feat, dummy_teacher_b_feat = teacher_model(dummy_input_for_teacher)
        dummy_teacher_ssim = calculate_ssim_torch(dummy_teacher_output.clamp(0,1), dummy_sharp_for_teacher.clamp(0,1))
        print(f"Dummy teacher output shape: {dummy_teacher_output.shape}")
        print(f"Dummy teacher e2 features shape (projected): {dummy_teacher_e2_feat.shape if dummy_teacher_e2_feat is not None else 'None'}")
        print(f"Dummy teacher e4 features shape (projected): {dummy_teacher_e4_feat.shape if dummy_teacher_e4_feat is not None else 'None'}")
        print(f"Dummy teacher bottleneck features shape (projected): {dummy_teacher_b_feat.shape if dummy_teacher_b_feat is not None else 'None'}")
        print(f"Dummy Teacher SSIM (on random data, for functional check): {dummy_teacher_ssim:.4f}")

    print("\nEvaluating Teacher Model on actual test set...")
    if test_loader and len(test_loader) > 0:
        teacher_avg_ssim, teacher_avg_psnr = evaluate_model(teacher_model, test_loader, device)
        print(f"Teacher Model Average SSIM on Test Set: {teacher_avg_ssim:.4f}")
        print(f"Teacher Model Average PSNR on Test Set: {teacher_avg_psnr:.4f} dB")
    else:
        print("Test DataLoader is empty or not initialized. Skipping Teacher Model evaluation.")

except Exception as e:
    print(f"\nFATAL ERROR: Failed to initialize Teacher Model: {e}")
    teacher_model = None

print("\n--- Teacher Model setup complete! ---")
if teacher_model:
    print("The `teacher_model` object is now available for use in knowledge distillation.")
    print("Teacher features are now projected to match student feature dimensions.")
else:
    print("Teacher model setup failed. Please check the error messages above.")


# Student Model: Lightweight Network 
print("\n" + "="*50)
print("--- Step 4: Student Model: Lightweight Network ---")

class StudentModelConfig:
    def __init__(self):
        self.in_channels = 3
        self.out_channels = 3
        self.base_channels = 256
        self.image_size = (data_config.image_size[0], data_config.image_size[1])

student_model_config = StudentModelConfig()
print(f"Student model input channels: {student_model_config.in_channels}")
print(f"Student model output channels: {student_model_config.out_channels}")
print(f"Student model base channels: {student_model_config.base_channels}")
print(f"Student model expected input size: {student_model_config.image_size[0]}x{student_model_config.image_size[1]}")
print("Student model will be trained on this resolution. Ensure your GPU can handle it.")


#Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Student Model Class (U-Net variant with SE Blocks) 
class StudentModel(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels):
        super(StudentModel, self).__init__()

        
        def conv_block(in_c, out_c, add_se=True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ]
            if add_se:
                layers.append(SEBlock(out_c))
            return nn.Sequential(*layers)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Path (Downsampling)
        self.enc1 = conv_block(in_channels, base_channels)
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = conv_block(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = conv_block(base_channels * 8, base_channels * 16)

        # Decoder Path (Upsampling with Transposed Convolutions and Skip Connections)
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_channels * 16, base_channels * 8)

        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)


    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with Skip Connections (Handles potential size mismatch)
        d4 = self.upconv4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))

        output = self.out_conv(d1)

        # Return final output and intermediate features for KD
        return output, e2, e4, b


student_model = None
try:
    student_model = StudentModel(
        in_channels=student_model_config.in_channels,
        out_channels=student_model_config.out_channels,
        base_channels=student_model_config.base_channels
    ).to(device)

    print(f"\nStudent Model architecture:\n{student_model}")
    print(f"\nStudent model successfully initialized and moved to {device}.")

    num_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in Student Model: {num_params:,}")

    # Test with a dummy input to verify forward pass and output shapes
    dummy_input = torch.randn(1, student_model_config.in_channels,
                               student_model_config.image_size[0],
                               student_model_config.image_size[1]).to(device)

    with torch.no_grad():
        dummy_output, dummy_e2_feat, dummy_e4_feat, dummy_b_feat = student_model(dummy_input)
    print(f"Dummy student output shape: {dummy_output.shape}")
    print(f"Dummy student e2 features shape: {dummy_e2_feat.shape}")
    print(f"Dummy student e4 features shape: {dummy_e4_feat.shape}")
    print(f"Dummy student bottleneck features shape: {dummy_b_feat.shape}")

except Exception as e:
    print(f"\nFATAL ERROR: Failed to initialize Student Model: {e}")
    student_model = None

print("\n--- Student Model setup complete! ---")
if student_model:
    print("The `student_model` object is now available for training.")
else:
    print("Student model setup failed. Please check the error messages above.")


# --- Component 5: Loss Functions for Knowledge Distillation ---
print("\n" + "="*50)
print("--- Step 5: Loss Functions for Knowledge Distillation ---")

class LossConfig:
    def __init__(self):
        self.lambda_recon = 1.0
        self.lambda_perceptual = 0.01
        self.lambda_feature_distillation = 1.0
        self.lambda_kl_div = 0.01
        self.lambda_ssim = 10.0

loss_config = LossConfig()
print(f"Loss weights: {loss_config.__dict__}")

criterion_recon = nn.L1Loss().to(device)
print("Reconstruction Loss (L1) initialized.")

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        try:
            vgg19_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        except ImportError:
            print("Perceptual Loss will be a dummy.")
            self.feature_extractor = None
            self.mse_loss = None
            return
        except Exception as e:
            print(f"Failed to load VGG19 model for Perceptual Loss: {e}. Perceptual Loss will be a dummy.")
            self.feature_extractor = None
            self.mse_loss = None
            return


        self.feature_extractor = nn.Sequential()
        vgg_layers = {'2': 'relu1_1', '7': 'relu2_1', '12': 'relu3_1', '21': 'relu4_1', '30': 'relu5_1'}
        i = 0
        for name, layer in vgg19_model._modules.items():
            if isinstance(layer, nn.Conv2d):
                i += 1
                if str(i) in vgg_layers:
                    self.feature_extractor.add_module(vgg_layers[str(i)], layer)
            self.feature_extractor.add_module(name, layer)
            if str(i) == '30':
                break

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, generated_image, target_image):
        if self.feature_extractor is None:
            return torch.tensor(0.0).to(generated_image.device)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        features_gen = self.feature_extractor(normalize(generated_image))
        features_target = self.feature_extractor(normalize(target_image))

        loss = self.mse_loss(features_gen, features_target)
        return loss

criterion_perceptual = None
try:
    criterion_perceptual = PerceptualLoss(device).to(device)
    print("Perceptual Loss (VGG-based) initialized.")
except Exception as e:
    print(f"ERROR: Could not initialize Perceptual Loss (VGG-based). Perceptual loss will return 0. Error: {e}")
    class DummyPerceptualLoss(nn.Module):
        def forward(self, gen, target): return torch.tensor(0.0).to(gen.device)
    criterion_perceptual = DummyPerceptualLoss().to(device)


criterion_feature_distillation = nn.L1Loss().to(device)
print("Feature Distillation Loss (L1) initialized. This acts as the 'Imitation Loss' for intermediate features.")

print("\n--- Loss Functions setup complete! ---")


# --- Component 6: Training Loop ---
print("\n" + "="*50)
print("--- Step 6: Training Loop ---")

class TrainingConfig:
    def __init__(self):
        self.num_epochs = 50
        self.learning_rate = 2e-4
        self.log_interval = 10
        self.save_interval = self.num_epochs
        self.eval_interval = 5
        self.gradient_accumulation_steps = 4
        self.temperature = 4.0
        self.ema_decay = 0.999

training_config = TrainingConfig()
print(f"Training will run for {training_config.num_epochs} epochs.")
print(f"KL Divergence Temperature (T): {training_config.temperature}")
print(f"EMA Decay Rate: {training_config.ema_decay}")

_actual_train_loader = None
if 'train_loader' in locals() and train_loader is not None and len(train_loader) > 0:
    print(f"Gradient accumulation steps: {training_config.gradient_accumulation_steps} (Effective batch size: {data_config.batch_size * training_config.gradient_accumulation_steps})")
    _actual_train_loader = train_loader
else:
    print(f"Gradient accumulation steps: {training_config.gradient_accumulation_steps} (Effective batch size: Unknown or DataLoader empty)")
    print("Train DataLoader is not ready or empty. Creating a dummy DataLoader for training demonstration.")
    dummy_blurry_images = torch.randn(data_config.batch_size, data_config.num_channels, data_config.image_size[0], data_config.image_size[1]).to(device)
    dummy_sharp_images = torch.randn(data_config.batch_size, data_config.num_channels, data_config.image_size[0], data_config.image_size[1]).to(device)
    _actual_train_loader = [(dummy_blurry_images, dummy_sharp_images, 'dummy_path.png')] * 10


optimizer = None
scheduler = None
ema_model = None
if 'student_model' in locals() and student_model is not None:
    optimizer = optim.Adam(student_model.parameters(), lr=training_config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    ema_model = deepcopy(student_model).eval()
    print("Optimizer (Adam), Scheduler (CosineAnnealingWarmRestarts), and EMA model initialized.")
else:
    print("ERROR: Student model not initialized. Optimizer, Scheduler, and EMA skipped.")

# Manual EMA update function
def update_ema(ema_model, student_model, decay):
    with torch.no_grad():
        for ema_param, student_param in zip(ema_model.parameters(), student_model.parameters()):
            ema_param.copy_(decay * ema_param + (1. - decay) * student_param)

def train_model():
    global _actual_train_loader
    global optimizer, scheduler, ema_model

    if student_model is None or teacher_model is None:
        print("Pre-requisite models are not initialized. Skipping training.")
        return

    if not _actual_train_loader or len(_actual_train_loader) == 0:
        print("Train DataLoader is empty. Training cannot proceed without data.")
        print("Please ensure your dataset paths are correct and contain images.")
        return

    if optimizer is None or scheduler is None or ema_model is None:
        print("Optimizer, Scheduler, or EMA not initialized. Skipping training.")
        return


    student_model.train()
    teacher_model.eval()
    ema_model.eval()

    print("\nStarting training...")
    print(f"Training for {training_config.num_epochs} epochs.")
    for epoch in range(training_config.num_epochs):
        total_recon_loss = 0
        total_perceptual_loss = 0
        total_feature_dist_loss = 0
        total_kl_div_loss = 0
        total_ssim_loss = 0
        total_combined_loss = 0
        batch_ssims = []

        pbar = tqdm(enumerate(_actual_train_loader), total=len(_actual_train_loader), desc=f"Epoch {epoch+1}/{training_config.num_epochs}")

        optimizer.zero_grad()

        for batch_idx, (blurry_images, sharp_images, _) in pbar:
            blurry_images = blurry_images.to(device)
            sharp_images = sharp_images.to(device)

            with torch.no_grad():
                teacher_output, teacher_e2_feat_raw, teacher_e4_feat_raw, teacher_b_feat_raw = teacher_model(blurry_images)

            student_output, student_e2_feat, student_e4_feat, student_b_feat = student_model(blurry_images)

            teacher_e2_feat_downsampled = F.interpolate(teacher_e2_feat_raw.detach(), size=student_e2_feat.shape[2:], mode='bilinear', align_corners=False) if teacher_e2_feat_raw is not None else None
            teacher_e4_feat_downsampled = F.interpolate(teacher_e4_feat_raw.detach(), size=student_e4_feat.shape[2:], mode='bilinear', align_corners=False) if teacher_e4_feat_raw is not None else None
            teacher_b_feat_downsampled = F.interpolate(teacher_b_feat_raw.detach(), size=student_b_feat.shape[2:], mode='bilinear', align_corners=False) if teacher_b_feat_raw is not None else None


            # Calculate Loss Components
            student_output_clamped = student_output.clamp(0, 1)
            teacher_output_clamped = teacher_output.clamp(0, 1)

            recon_loss = criterion_recon(student_output, sharp_images)
            perceptual_loss = criterion_perceptual(student_output_clamped, teacher_output_clamped) if criterion_perceptual else torch.tensor(0.0).to(device)

            feature_losses = []
            if teacher_e2_feat_downsampled is not None and student_e2_feat is not None:
                feature_losses.append(criterion_feature_distillation(student_e2_feat, teacher_e2_feat_downsampled))
            if teacher_e4_feat_downsampled is not None and student_e4_feat is not None:
                feature_losses.append(criterion_feature_distillation(student_e4_feat, teacher_e4_feat_downsampled))
            if teacher_b_feat_downsampled is not None and student_b_feat is not None:
                feature_losses.append(criterion_feature_distillation(student_b_feat, teacher_b_feat_downsampled))

            total_feature_dist_loss_component = sum(feature_losses) / len(feature_losses) if feature_losses else torch.tensor(0.0).to(device)


            # KL Divergence Loss
            kl_div_loss = F.kl_div(
                F.log_softmax(student_output / training_config.temperature, dim=1),
                F.softmax(teacher_output / training_config.temperature, dim=1),
                reduction='batchmean'
            ) * (training_config.temperature ** 2)

            # SSIM Loss
            ssim_loss_component = 1 - calculate_ssim_torch(student_output_clamped, sharp_images)

            
            combined_loss = (loss_config.lambda_recon * recon_loss +
                             loss_config.lambda_perceptual * perceptual_loss +
                             loss_config.lambda_feature_distillation * total_feature_dist_loss_component +
                             loss_config.lambda_kl_div * kl_div_loss +
                             loss_config.lambda_ssim * ssim_loss_component)

            combined_loss = combined_loss / training_config.gradient_accumulation_steps
            combined_loss.backward()

            if (batch_idx + 1) % training_config.gradient_accumulation_steps == 0:
                optimizer.step()
                update_ema(ema_model, student_model, training_config.ema_decay) # Manual EMA update
                optimizer.zero_grad()

            total_recon_loss += recon_loss.item()
            total_perceptual_loss += perceptual_loss.item()
            total_feature_dist_loss += total_feature_dist_loss_component.item()
            total_kl_div_loss += kl_div_loss.item()
            total_ssim_loss += ssim_loss_component.item()
            total_combined_loss += combined_loss.item() * training_config.gradient_accumulation_steps

            if batch_idx % training_config.log_interval == 0:
                current_ssim = calculate_ssim_torch(student_output_clamped[0].unsqueeze(0), sharp_images[0].unsqueeze(0))
                batch_ssims.append(current_ssim.item())

                pbar.set_postfix({
                    'Loss': f'{combined_loss.item() * training_config.gradient_accumulation_steps:.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'Percept': f'{perceptual_loss.item():.4f}',
                    'FeatDist': f'{total_feature_dist_loss_component.item():.4f}',
                    'KLDiv': f'{kl_div_loss.item():.4f}',
                    'SSIM_L': f'{ssim_loss_component.item():.4f}',
                    'SSIM_Val': f'{current_ssim:.4f}'
                })

        if (batch_idx + 1) % training_config.gradient_accumulation_steps != 0:
            optimizer.step()
            update_ema(ema_model, student_model, training_config.ema_decay)
            optimizer.zero_grad()

        scheduler.step()

        avg_recon_loss = total_recon_loss / len(_actual_train_loader)
        avg_perceptual_loss = total_perceptual_loss / len(_actual_train_loader)
        avg_feature_dist_loss = total_feature_dist_loss / len(_actual_train_loader)
        avg_kl_div_loss = total_kl_div_loss / len(_actual_train_loader)
        avg_ssim_loss = total_ssim_loss / len(_actual_train_loader)
        avg_combined_loss = total_combined_loss / len(_actual_train_loader)
        avg_epoch_ssim = np.mean(batch_ssims) if batch_ssims else 0.0

        print(f"\n--- Epoch {epoch+1}/{training_config.num_epochs} Summary ---")
        print(f"  Avg Combined Loss: {avg_combined_loss:.4f}")
        print(f"  Avg Recon Loss: {avg_recon_loss:.4f}")
        print(f"  Avg Perceptual Loss: {avg_perceptual_loss:.4f}")
        print(f"  Avg Feature Distillation Loss: {avg_feature_dist_loss:.4f}")
        print(f"  Avg KL Divergence Loss: {avg_kl_div_loss:.4f}")
        print(f"  Avg SSIM Loss (1-SSIM): {avg_ssim_loss:.4f}")
        print(f"  Avg SSIM (sampled): {avg_epoch_ssim:.4f}")
        print(f"  Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        
        if (epoch + 1) == training_config.num_epochs:
            checkpoint_path = os.path.join(model_checkpoints_dir, f'student_model_ema_final_epoch_{epoch+1:03d}.pth')
            torch.save(ema_model.state_dict(), checkpoint_path)
            print(f"  Final EMA Student model checkpoint saved to {checkpoint_path}")
        else:
            print(f"  Skipping checkpoint save for epoch {epoch+1}.")


        if test_loader and len(test_loader) > 0 and (epoch + 1) % training_config.eval_interval == 0:
            print("\n  Running evaluation on test set using EMA model...")
            
            evaluate_model(ema_model, test_loader, device, save_output_images=True, output_dir=data_config.output_image_dir)
            student_model.train() 


# training
train_model()

print("\n--- Training Loop setup complete! ---")


# Performance Evaluation ---
print("\n" + "="*50)
print("Performance Evaluation ---")

print("Evaluation function defined in Utility Functions section.")

try:
    # Path to the last saved EMA checkpoint
    latest_checkpoint = os.path.join(model_checkpoints_dir, f'student_model_ema_final_epoch_{training_config.num_epochs:03d}.pth')

    if os.path.exists(latest_checkpoint):
        print(f"\nAttempting to evaluate model from {latest_checkpoint} (EMA model)...")
        if test_loader and len(test_loader) > 0:
            # Create a new StudentModel instance to load the EMA weights into
            eval_student_model = StudentModel(
                in_channels=student_model_config.in_channels,
                out_channels=student_model_config.out_channels,
                base_channels=student_model_config.base_channels
            ).to(device)
            evaluate_model(eval_student_model, test_loader, device, checkpoint_path=latest_checkpoint, save_output_images=True, output_dir=data_config.output_image_dir)
        else:
            print("Test DataLoader is empty or not initialized. Skipping final evaluation.")
    else:
        print(f"\nCheckpoint not found at {latest_checkpoint}. Skipping standalone evaluation.")

except Exception as e:
    print(f"An error occurred during standalone evaluation: {e}")


print("\n" + "="*50)
print("Inference & Deployment Considerations ---")

class InferenceConfig:
    def __init__(self):
        
        self.trained_model_path = os.path.join(model_checkpoints_dir, f'student_model_ema_final_epoch_{training_config.num_epochs:03d}.pth')
        self.model_input_size = (data_config.image_size[0], data_config.image_size[1])
        self.target_output_resolution = (1920, 1080)

        self.use_tiling = True
        self.tiling_patch_size = 512
        self.tiling_overlap = 64


inference_config = InferenceConfig()
print(f"Default model internal processing size: {inference_config.model_input_size[0]}x{inference_config.model_input_size[1]}")
print(f"Expected path for trained student model: {inference_config.trained_model_path}")
print(f"Target output resolution (for final image save): {inference_config.target_output_resolution[0]}x{inference_config.target_output_resolution[1]}")
if inference_config.use_tiling:
    print(f"Tiling enabled: {inference_config.use_tiling} with patch size: {inference_config.tiling_patch_size} and overlap: {inference_config.tiling_overlap}")
else:
    print("Tiling is currently disabled.")


def sharpen_image_inference(input_image_path, output_image_path, model, device,
                            model_checkpoint_path=None, target_resolution=None,
                            use_tiling=False, tiling_patch_size=256, tiling_overlap=64):
    """
    Loads a blurry image, sharpens it using the trained student model, and saves the output.
    Supports loading a model checkpoint, simple resizing, and a basic tiling strategy.
    Args:
        input_image_path (str): Path to the blurry input image.
        output_image_path (str): Path to save the sharpened output image.
        model (nn.Module): The student model instance.
        device (torch.device): The device (cuda or cpu) to run inference on.
        model_checkpoint_path (str, optional): Path to the trained model's state_dict.
                                                If None, uses the model as is (already loaded).
        target_resolution (tuple, optional): (width, height) to resize the final output to.
                                            If None, outputs at the same size as model's processed input.
        use_tiling (bool): Whether to use tiling for inference on large images.
        tiling_patch_size (int): Size of square patches for tiling (e.g., 512).
        tiling_overlap (int): Overlap between patches for smooth stitching.
    """
    if model_checkpoint_path:
        print(f"Loading student model weights from {model_checkpoint_path}...")
        try:
            model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
            print("Student model weights loaded successfully for inference.")
        except FileNotFoundError:
            print(f"ERROR: Model checkpoint not found at {model_checkpoint_path}. Cannot perform inference.")
            return
        except RuntimeError as e:
            print(f"ERROR: Failed to load checkpoint due to model architecture mismatch or corrupted file: {e}. Cannot perform inference.")
            return
        except Exception as e:
            print(f"ERROR loading model checkpoint: {e}. Cannot perform inference.")
            return

    model.eval()
    model.to(device)

    if input_image_path is None:
        input_image_pil = Image.new('RGB', (tiling_patch_size, tiling_patch_size), color = 'black')
        print(f"  Processing dummy image (in-memory) for warm-up/FPS measurement.")
    else:
        print(f"Processing image: {os.path.basename(input_image_path)}")
        try:
            input_image_pil = Image.open(input_image_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Input image not found at {input_image_path}.")
            return
        except Exception as e:
            print(f"ERROR loading input image {input_image_path}: {e}")
            return

    original_width, original_height = input_image_pil.size

    start_time = time.time()

    if use_tiling:
        print(f"  Performing tiled inference with patch size {tiling_patch_size} and overlap {tiling_overlap}")

        stride = tiling_patch_size - tiling_overlap
        num_patches_h = math.ceil((original_height - tiling_overlap) / stride)
        num_patches_w = math.ceil((original_width - tiling_overlap) / stride)

        output_accumulator = torch.zeros(3, original_height, original_width).to(device)
        output_weights = torch.zeros(3, original_height, original_width).to(device)
        patch_mask = torch.ones(tiling_patch_size, tiling_patch_size).to(device)

        preprocess_patch = transforms.Compose([
            transforms.Resize((tiling_patch_size, tiling_patch_size), transforms.InterpolationMode.BICUBIC),
            CustomToTensor()
        ])

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                top = i * stride
                left = j * stride
                bottom = min(top + tiling_patch_size, original_height)
                right = min(left + tiling_patch_size, original_width)

                if bottom == original_height and top + tiling_patch_size > original_height:
                    top = original_height - tiling_patch_size
                if right == original_width and left + tiling_patch_size > original_width:
                    left = original_width - tiling_patch_size

                patch_pil = input_image_pil.crop((left, top, right, bottom))
                patch_tensor = preprocess_patch(patch_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    
                    sharpened_patch_tensor, *rest = model(patch_tensor)

                sharpened_patch_tensor = sharpened_patch_tensor.squeeze(0).clamp(0, 1)

                output_accumulator[:, top:bottom, left:right] += sharpened_patch_tensor * patch_mask
                output_weights[:, top:bottom, left:right] += patch_mask

        output_weights[output_weights == 0] = 1e-12
        final_sharpened_tensor = output_accumulator / output_weights

        sharpened_image_pil = transforms.ToPILImage()(final_sharpened_tensor.cpu())

    else:
        preprocess_for_model = transforms.Compose([
            transforms.Resize(inference_config.model_input_size, transforms.InterpolationMode.BICUBIC),
            CustomToTensor()
        ])
        input_tensor = preprocess_for_model(input_image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            sharpened_tensor_model_output, *rest = model(input_tensor)

        sharpened_image_pil = transforms.ToPILImage()(sharpened_tensor_model_output.squeeze(0).cpu().clamp(0, 1))

    end_time = time.time()
    processing_time_ms = (end_time - start_time) * 1000
    print(f"  Total inference time (including potential tiling/resizing): {processing_time_ms:.2f} ms")


    if target_resolution and sharpened_image_pil.size != target_resolution:
        sharpened_image_pil = sharpened_image_pil.resize(target_resolution, Image.LANCZOS)
        print(f"  Final output resolution resized to target: {sharpened_image_pil.size[0]}x{sharpened_image_pil.size[1]}")
    elif (input_image_path is not None) and ((original_width, original_height) != sharpened_image_pil.size):
        sharpened_image_pil = sharpened_image_pil.resize(input_image_pil.size, Image.LANCZOS)
        print(f"  Final output resolution (resized to original input if not tiled): {sharpened_image_pil.size[0]}x{sharpened_image_pil.size[1]}")

    if output_image_path is not None:
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        sharpened_image_pil.save(output_image_path)
        print(f"Sharpened image saved to: {output_image_path}")

print("Inference function `sharpen_image_inference` defined.")

print("\n--- High-Resolution Inference Strategy ---")
print(f"Your model's internal processing size is: {inference_config.model_input_size[0]}x{inference_config.model_input_size[1]}.")
print(f"Your desired final output resolution is: {inference_config.target_output_resolution[0]}x{inference_config.target_output_resolution[1]}.")
print("1.  Simple Scaling: Input images are resized to the model's internal processing size, inference is run, then the output is resized to the target resolution.")
print("2.  Tiling/Patching: Splits the high-resolution input image into smaller, overlapping patches. Runs inference on each patch independently. Stitches the sharpened patches back together, handling overlaps using blending to avoid visible seams.")


def measure_fps(model, device, num_runs=10, input_size=(1080, 1920), use_tiling=False, tiling_patch_size=512, tiling_overlap=64):
    """
    Measures the approximate Frames Per Second (FPS) of the student model, optionally with tiling.
    """
    if not isinstance(model, nn.Module):
        print(f"ERROR: Invalid model provided to measure_fps. Expected nn.Module, got {type(model)}. Skipping FPS measurement.")
        return 0.0

    model.eval()
    model.to(device)

    dummy_image_width, dummy_image_height = input_size[1], input_size[0]
    dummy_large_image_pil = Image.new('RGB', (dummy_image_width, dummy_image_height), color = 'black')
    draw = ImageDraw.Draw(dummy_large_image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except IOError:
        font = ImageFont.load_default()
    draw.text((dummy_image_width // 4, dummy_image_height // 2 - 25), "Dummy FPS Test Image", fill=(255,255,255), font=font)

    
    temp_fps_files_dir = os.path.join(local_project_root, 'temp_fps_files')
    os.makedirs(temp_fps_files_dir, exist_ok=True)

    temp_blurry_path = os.path.join(temp_fps_files_dir, 'temp_dummy_blurry_for_fps.png')
    temp_output_path = os.path.join(temp_fps_files_dir, 'temp_dummy_sharpened_for_fps.png')


    print(f"\nWarm-up runs for FPS measurement ({input_size[0]}x{input_size[1]} input, tiling={use_tiling}) on {device}...")
    for _ in range(5):
        dummy_large_image_pil.save(temp_blurry_path)
        sharpen_image_inference(
            input_image_path=temp_blurry_path,
            output_image_path=None,
            model=model,
            device=device,
            use_tiling=use_tiling,
            tiling_patch_size=tiling_patch_size,
            tiling_overlap=tiling_overlap
        )
        if os.path.exists(temp_blurry_path): os.remove(temp_blurry_path)

    print(f"Measuring FPS over {num_runs} runs...")

    individual_times_ms = []

    for _ in tqdm(range(num_runs)):
        iter_start_time = time.time()

        dummy_large_image_pil.save(temp_blurry_path)

        sharpen_image_inference(
            input_image_path=temp_blurry_path,
            output_image_path=temp_output_path,
            model=model,
            device=device,
            model_checkpoint_path=None,
            target_resolution=input_size[::-1],
            use_tiling=use_tiling,
            tiling_patch_size=tiling_patch_size,
            tiling_overlap=tiling_overlap
        )

        if os.path.exists(temp_blurry_path): os.remove(temp_blurry_path)
        if os.path.exists(temp_output_path): os.remove(temp_output_path)

        iter_end_time = time.time()
        individual_times_ms.append((iter_end_time - iter_start_time) * 1000)


    avg_time_per_frame_ms = np.mean(individual_times_ms)
    fps = 1000 / avg_time_per_frame_ms if avg_time_per_frame_ms > 0 else float('inf')

    print(f"  Average inference time per {input_size[0]}x{input_size[1]} frame (tiling={use_tiling}): {avg_time_per_frame_ms:.2f} ms")
    print(f"  Estimated FPS for {input_size[0]}x{input_size[1]} input (tiling={use_tiling}): {fps:.2f} FPS")

    
    if os.path.exists(temp_fps_files_dir):
        shutil.rmtree(temp_fps_files_dir)
        print(f"Cleaned up temporary FPS files directory: {temp_fps_files_dir}")

    return fps

print("FPS measurement function defined.")


print("\n--- Running example inference on test set images ---")


sharpened_test_outputs_dir = data_config.output_image_dir
os.makedirs(sharpened_test_outputs_dir, exist_ok=True)


test_blurry_image_paths = natsorted(glob.glob(os.path.join(data_config.test_blurry_dir, '*.*')))

if student_model is None or not isinstance(student_model, nn.Module):
    print("\nFATAL ERROR: Student model is not properly initialized. Skipping inference and FPS measurement examples.")
else:
    if not test_blurry_image_paths:
        print(f"No test blurry images found in {data_config.test_blurry_dir}. Skipping test inference example.")
    else:
        num_test_images_to_process = min(5, len(test_blurry_image_paths))
        print(f"Processing {num_test_images_to_process} images from the test set...")

        for i in range(num_test_images_to_process):
            input_path = test_blurry_image_paths[i]
            output_filename = f"sharpened_test_{os.path.basename(input_path)}"
            output_path = os.path.join(sharpened_test_outputs_dir, output_filename)

        
            inference_model = StudentModel(
                in_channels=student_model_config.in_channels,
                out_channels=student_model_config.out_channels,
                base_channels=student_model_config.base_channels
            ).to(device)

            sharpen_image_inference(
                input_image_path=input_path,
                output_image_path=output_path,
                model=inference_model,
                device=device,
                model_checkpoint_path=inference_config.trained_model_path,
                target_resolution=inference_config.target_output_resolution,
                use_tiling=inference_config.use_tiling,
                tiling_patch_size=inference_config.tiling_patch_size,
                tiling_overlap=inference_config.tiling_overlap
            )
            print("-" * 30)

    
    print("\nMeasuring FPS for model's trained input size (512x512), no tiling:")
    if ema_model:
        measure_fps(ema_model, device, num_runs=10, input_size=inference_config.model_input_size, use_tiling=False)
    else:
        print("EMA model not initialized, skipping FPS measurement.")


    print("\nMeasuring FPS for target output resolution (1920x1080) WITH TILING:")
    if ema_model:
        measure_fps(ema_model, device, num_runs=5, input_size=(1080, 1920), use_tiling=True,
                    tiling_patch_size=inference_config.tiling_patch_size, tiling_overlap=inference_config.tiling_overlap)
    else:
        print("EMA model not initialized, skipping FPS measurement.")


print("\n--- Inference & Deployment component setup complete! ---")
print("You now have functions to perform inference and measure FPS.")
print("Remember: for high-resolution images, implement a tiling strategy for best results after training on smaller patches.")
