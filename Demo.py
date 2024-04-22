import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings("ignore")
from utils import seamLessClone, load_images, gaussian_mean, getFlowMaskGlobal, stabilize_GPU_optimized, computePseudoCn2V2, load_and_predict

# Configuration Settings
doStabilize = True                  # Enable or disable image stabilization (True to enable stabilization)
ProcessNumberOfFrames = 100         # Number of frames to process from the input images
resizeFactor = 1                    # Factor to resize images (2 implies downsizing by half)
MaxStb = 50                         # Maximum allowed pixel shift for image stabilization
path = 'Input/Single_Car/*.png'     # Path to input images (format to ensure compatibility with glob)
savePath = 'Output/Single_Car/'     # Path to save output images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
restormer_weight_path = 'PretrainedModel/restormer_ASUSim_trained.pth'

# Load images
imgTensor = load_images(path,  ResizeFactor=resizeFactor)
print(f'Loaded {len(imgTensor)} frames')
imgTensor = imgTensor[:ProcessNumberOfFrames]

# Compute pseudo Cn2 for image quality and stabilization
pseudoCn2 = computePseudoCn2V2(imgTensor)
if doStabilize:
    imgTensor, x_list, y_list = stabilize_GPU_optimized(imgTensor, MaxStb=50)
pseudoCn2 = computePseudoCn2V2(imgTensor)
pseudoCn2 = (pseudoCn2**0.8) * 1.3

# Generate global segmentation masks
GlobalSegMaskList = getFlowMaskGlobal(imgTensor, n=5, ThMagnify=1.5)

# Prepare mask tensor for 3D convolution
MaskTensor = torch.tensor(np.array(GlobalSegMaskList), dtype=torch.float).to(device)    # [N, H, W]
MaskTensor = rearrange(MaskTensor, 'n h w -> 1 1 n h w')                                # [1, 1, N, H, W]

# Define kernel
kernel_size = [int(pseudoCn2*2+1), 20, 20]      # Adaptive Temporal blur + spatial blur on 20x20 pixels
kernel_size = [k + 1 if k % 2 == 0 else k for k in kernel_size]
kernel = torch.ones(kernel_size, device=MaskTensor.device)
kernel = rearrange(kernel, 'd h w -> 1 1 d h w')                                        # [1, 1, D, H, W]

# Apply 3D convolution with padding
padding = [k // 2 for k in kernel_size]
dilated_mask = F.conv3d(MaskTensor, kernel, padding=padding)
Masks = rearrange((dilated_mask > 0).float(), '1 1 n h w -> n h w')
Masks = repeat(Masks, 'n h w -> n c h w', c=3)

# Apply adaptive Gaussian weighting for temporal blurring on the images
Gaussian_ImgCube = gaussian_mean(imgTensor.to(device), Masks, sigma=pseudoCn2)


# Prepare background and foreground image cubes
BG_NHWC = rearrange(Gaussian_ImgCube, 'n c h w -> n h w c')[:ProcessNumberOfFrames].cpu().numpy()
FG_NHWC = rearrange(imgTensor, 'n c h w -> n h w c')[:ProcessNumberOfFrames].cpu().numpy()
Masks = np.array(Masks.cpu().numpy())[:ProcessNumberOfFrames]

# Combine background and foreground images
N, H, W, C = len(BG_NHWC), BG_NHWC[0].shape[0], BG_NHWC[0].shape[1], BG_NHWC[0].shape[2]
CB_NHWC = torch.zeros((N, H, W, C))
Masks3Ch = rearrange(Masks, 'n c h w -> n h w c')
Masks3Ch = (Masks3Ch * 255).astype(np.uint8)

for i in tqdm(range(N)):
    BG = (BG_NHWC[i]*255).astype(np.uint8)
    FG = (FG_NHWC[i]*255).astype(np.uint8)
    CB = seamLessClone(BG, FG, Masks3Ch[i, :, :, 0])
    CB_NHWC[i] = torch.tensor(CB)

CB_NHWC = CB_NHWC/255

os.makedirs(savePath, exist_ok=True)

# Save processed images
for i in range(len(CB_NHWC)):
    image_np = CB_NHWC[i].cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    
    try:
        output = load_and_predict(image_np, restormer_weight_path)
        plt.imsave(f'{savePath}/{str(i).zfill(4)}_Restored_Enhanced.jpg', output)
    except:
        # Check if the pretrained model is available
        assert os.path.exists(restormer_weight_path), "Pretrained model not found."

        # Check available GPU VRAM is sufficient for Restormer
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'Available GPU VRAM: {gpu_memory:.2f} GB is not enough for Restormer. Please use a GPU with at least 24GB VRAM.')
        
        # Save the image without Restormer
        plt.imsave(f'{savePath}/{str(i).zfill(4)}_JustRestored_NotEnhanced.jpg', image_np)
