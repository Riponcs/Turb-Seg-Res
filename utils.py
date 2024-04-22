# Standard library imports
from PIL import Image
from scipy.signal import gaussian
from glob import glob
from tqdm import tqdm

# Third-party imports
import cv2
import imageio.v3 as io
import kornia.geometry.transform as kg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from restormer_model import Restormer
from torchvision import transforms as transforms
from PIL import Image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load Images
def load_images(path, ResizeFactor=2):
    imgList = sorted(glob(path))

    oh, ow = io.imread(imgList[0]).shape[:2]
    h, w = ((oh // (8*ResizeFactor)) * 8, (ow // (8*ResizeFactor)) * 8) if oh > 0 and ow > 0 else (0, 0)
    resizeRatio, resizeRatioY = (h/oh, w/ow) if oh > 0 and ow > 0 else (0, 0)

    assert h > 0 and w > 0, "Invalid original dimensions!"

    imgTensor = torch.stack([T.Compose([T.Resize((h, w)), T.ToTensor()])(Image.open(img)) for img in imgList]) #torch.Size([time, ch, h, w])
    return imgTensor


# Function to find the gradient of an image
def imgradient(img):
    if type(img) != np.ndarray:
        img = img.convert('L')
        img = np.asarray(img)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gx = cv2.filter2D(img, -1, sobel_x)
    Gy = cv2.filter2D(img, -1, sobel_y)
    grad_mag = np.sqrt(Gx**2 + Gy**2)
    return grad_mag

# Function to find the Windows with minimum movement
def find_min_window(arr, window_size=(5, 5)):
    min_value = float('inf')  # Initialize minimum value to positive infinity
    min_position = (0, 0)  # Initialize minimum position to (0, 0)
    
    rows, cols = arr.shape
    w, h = window_size
    
    for i in range(0, rows - h + 1):
        for j in range(0, cols - w + 1):
            window = arr[i:i+h, j:j+w]
            window_mean = np.mean(window)
            
            if window_mean < min_value:
                min_value = window_mean
                min_position = (i, j)
                
    return min_position, min_value


def seamLessClone(BG, FG, mask, kernel_size=30):
    """
    BG: Background Image of size (H, W, 3)
    FG: Foreground Image of size (H, W, 3)
    mask: Mask of size (H, W) with values 0 or 255
    kernel_size: Kernel size for the dilation operation
    """
    if mask.max() == 0:
        return BG

    # Make sure the BG, FG and Mask are of shame shape
    assert BG.shape == FG.shape, "BG and FG must be of same shape"
    assert BG.shape[:2] == mask.shape, "BG and Mask must be of same shape"

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernel_size+1, 2*kernel_size+1))
    mask = cv2.dilate(mask, kernel)

    br = cv2.boundingRect(mask) # bounding rect (x,y,width,height)
    center = (br[0] + br[2] // 2, br[1] + br[3] // 2)

    # Perform seamless cloning
    output = cv2.seamlessClone(FG, BG, mask, center, cv2.NORMAL_CLONE)

    return output



def gaussian_weights(n, current, sigma):
    x = np.arange(n)
    weights = np.exp(-(x - current)**2 / (2 * sigma**2))
    return torch.tensor(weights / weights.sum(), dtype=torch.float32, device=device)

def gaussian_mean(frames, masks, sigma):
    n, ch, h, w = frames.shape
    smoothed_frames = torch.zeros_like(frames)
    
    for i in range(n):
        weights = gaussian_weights(n, i, sigma)
        background_mask = torch.tensor(masks[i] == 0, dtype=torch.float32, device=device)
        foreground_mask = torch.tensor(masks[i] == 1, dtype=torch.float32, device=device)
        
        # Apply weights only to the background
        weighted_background = frames * weights[:, None, None, None] * background_mask
        smoothed_background = weighted_background.sum(dim=0)

        # Add the original foreground
        smoothed_frames[i] = smoothed_background + frames[i] * foreground_mask
    
    return smoothed_frames
    

def stabilize_GPU_optimized(video_cube, MaxStb=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_cube = video_cube.to(device)
    x_list, y_list = [], []

    video_cube_mean = video_cube.mean()
    video_cube = video_cube - video_cube_mean  # 0 mean
    stabilized_video = torch.empty_like(video_cube).to(device)
    ref_frame = video_cube[0]                                                         
    M = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]]).to(device)

    for i in tqdm(range(len(video_cube))):
        current_frame = video_cube[i]
        frame_Cropped = current_frame[:, MaxStb:-MaxStb, MaxStb:-MaxStb]

        # Perform template matching to get the shift
        HeatMat = torch.nn.functional.conv2d(ref_frame[None, :1, :, :], frame_Cropped[None, :1, :, :])

        # Find the indices of the maximum of the HeatMat
        N, C, H, W = HeatMat.shape
        max_index = torch.argmax(HeatMat)
        y, x = max_index // W, max_index % W
        y, x = y - MaxStb, x - MaxStb

        x_list.append(x.item())  # Convert tensor to Python scalar
        y_list.append(y.item())  # Convert tensor to Python scalar

        # Update the transformation matrix for the current shift
        M[0, 0, 2] = x.float()
        M[0, 1, 2] = y.float()

        # Warp the current frame using the updated transformation matrix
        stabilized_video[i] = kg.warp_affine(current_frame.unsqueeze(0).float(), M, dsize=(current_frame.shape[-2], current_frame.shape[-1]))[0]

    stabilized_video = stabilized_video + video_cube_mean

    return stabilized_video, x_list, y_list


# Function for Global Video Segmentation
def otsu_threshold_from_histogram(hist):
    total = np.sum(hist)
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = np.dot(np.arange(256), hist)
    for i in range(256):
        wB += hist[i]
        wF = total - wB
        if wB == 0 or wF == 0:
            continue
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between >= maximum:
            level = i
            maximum = between
    return level

def getFlowMaskGlobal(imgTensor, ResizeFactor=2, deviceId = "cuda", n = 5, ThMagnify=1.5):

    """
    imgList: List of image paths can be extracted from glob
    ResizeFactor: Resize factor for the image. Default is 2 times downsampled
    deviceId: Device ID for the GPU. Default is "cuda" so that it doesn't use the main GPU
    n = 5: Number of frames to consider for the optical flow. Default is 5
    """
    N, C, H, W = imgTensor.shape
    resized = False
    if max(imgTensor.shape) > 1000:
        resized = True
        ResizeFactor = 2
        imgTensor = F.interpolate(imgTensor, size=((H//ResizeFactor//8)*8, (W//ResizeFactor//8)*8), mode='bilinear', align_corners=False)

    # Resize the images by ResizeFactor to Downsample


    device = deviceId if torch.cuda.is_available() else "cpu"
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()

    batch = 1
    SegMaskList = []
    perfectMeanList = []
    imgTensor = imgTensor.to(device)

    combined_histogram = np.zeros(256)  # Step 1

    for idx in tqdm(range(0, len(imgTensor))):
        HSVStack = []

        # Loop through max(i-n, 0) to min(len(dataset), i+n)
        for diff in range(-n, n + 1):
            if diff == 0 or idx + diff < 0 or idx + diff >= len(imgTensor):
                continue
            
            imgTensor1 = imgTensor[idx: idx + batch]
            imgTensor2 = imgTensor[idx + diff: idx + batch + diff]

            with torch.no_grad():
                out = model(imgTensor1, imgTensor2)[-1][0]
                # RGB = flow_to_image(out)
                # RGB = einops.rearrange(RGB, 'c h w -> h w c').cpu().numpy()
                # HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)                  #(480, 992, 3)

                # Measure the magnitude of the flow
                HSV = torch.sqrt(torch.sum(out ** 2, dim=0, keepdim=True)).cpu().numpy()[0]
                HSV = np.stack([HSV, HSV, HSV], axis=2)

                HSVStack.append(HSV)
            
        HSVStack = np.array(HSVStack)
        HSVMeanList = [np.mean(HSVStack[max(0, i - n):min(len(HSVStack), i + n + 1)], axis=0)[:, :, 1].astype(np.uint8) for i in range(len(HSVStack))]
        
        # Calculate perfectness of the mask
        minDistance = np.max(HSVMeanList)                # Initial Assign, will be updated
        perfectMean = HSVMeanList[0].copy()
        HM_dis_list = []
        
        for HM in HSVMeanList:
            # Normalize the mask
            # HM = ((HM - np.min(HM)) / (np.max(HM) - np.min(HM)) * 255).astype(np.uint8)
            distance = np.max(HM)/2 - np.mean(abs(np.max(HM)/2 - HM))
            HM_dis_list.append(distance)
            if distance < minDistance:
                minDistance = distance
                perfectMean = HM
        
        hist, _ = np.histogram(perfectMean, bins=np.arange(256 + 1))
        combined_histogram += hist

        perfectMeanList.append(perfectMean)

        # Now Normalize it from 0 to 255
        # perfectMean = ((perfectMean - np.min(perfectMean)) / (np.max(perfectMean) - np.min(perfectMean)) * 255).astype(np.uint8)
        # Save the perfect mean
        _, thresh = cv2.threshold(np.uint8(perfectMean), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if resized:
            thresh = cv2.resize(thresh, (W, H))
        SegMaskList.append(thresh)


        
    # Calculate the global Otsu's threshold from the combined_histogram
    global_otsu_thresh = otsu_threshold_from_histogram(combined_histogram)

    # Step 4: Apply global Otsu's threshold to each frame
    GlobalSegMaskList = []
    for perfectMean in perfectMeanList:
        _, thresh = cv2.threshold(np.uint8(perfectMean), global_otsu_thresh/ThMagnify, 255, cv2.THRESH_BINARY)
        if resized:
            thresh = cv2.resize(thresh, (W, H))
        GlobalSegMaskList.append(thresh)

    return GlobalSegMaskList


# Compute the Pseudo Cn2
def imgradient_torch(img_tensor):
    # Ensure img_tensor is a torch.Tensor
    if not isinstance(img_tensor, torch.Tensor):
        raise ValueError("Input should be a torch.Tensor")

    # Define Sobel kernels
    sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    sobel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)

    # Check if the tensor is on a GPU and if so move the sobel kernels to the GPU
    if img_tensor.is_cuda:
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()

    # Get the number of channels (C) and create C copies of the sobel filters to be used with group convolution
    C = img_tensor.shape[1]
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)

    # Apply the filters using depthwise convolution
    Gx = F.conv2d(img_tensor, sobel_x, groups=C, padding=1)
    Gy = F.conv2d(img_tensor, sobel_y, groups=C, padding=1)
    
    # Compute gradient magnitude
    grad_mag = torch.sqrt(Gx**2 + Gy**2)
    return grad_mag


def computePseudoCn2(videoTensorStab):
    """
    videoTensorStab: Video Tensor of size (N, C, H, W)
    """
    assert len(videoTensorStab.shape) == 4, "videoTensorStab must be of shape (N, C, H, W)"
    assert videoTensorStab.shape[1] <=4, "videoTensorStab must be of shape (N, <=4, H, W)"
    
    videoTensorStabMean = videoTensorStab.mean(dim=1, keepdim=True)       # torch (N, 1, H, W)
    # Get First 10 Frames
    videoTensorStabMean = videoTensorStabMean[:10]

    # Compute the variance of GrayScale images across the time dimension
    var_I = videoTensorStabMean.var(dim=0)                                # torch (1, H, W)

    # Compute the average gradient of GrayScale images
    GradTensor = imgradient_torch(videoTensorStabMean)      # torch (N, 1, H, W)
    GradTensorMean = GradTensor.mean(dim=0)                 # torch (1, H, W)
    GradTensorMean[GradTensorMean < GradTensorMean.max()/100] = GradTensorMean.max()/100

    # Calculate the Pseudo-Cn2 Vaule
    constant = 1e4
    Pseudo_cn2_map = var_I / GradTensorMean * constant
    # Pseudo_cn2_val = Pseudo_cn2_map.median()
    # Pseudo_cn2_val = torch.mean(Pseudo_cn2_map)
    
    # Get the Pseudo-Cn2 value at 10th percentile, subsample to reduce computation
    Pseudo_cn2_val = torch.quantile(Pseudo_cn2_map, 0.1)
    # Pseudo_cn2_mean = torch.mean(Pseudo_cn2_map)*0.5
    return np.round(Pseudo_cn2_val.item(), 2)

def computePseudoCn2V2(videoTensorStab):
    assert len(videoTensorStab.shape) == 4, "videoTensorStab must be of shape (N, C, H, W)"
    assert videoTensorStab.shape[1] <= 4, "videoTensorStab must be of shape (N, <=4, H, W)"
    
    # Crop the frames
    croppedVideoTensor = videoTensorStab[:, :, 50:-50, 50:-50]
    
    n = croppedVideoTensor.shape[0]
    cn2_values = []

    for i in range(n - 4):
        # Consider consecutive pairs of frames
        frame_pair = croppedVideoTensor[i:i + 5].mean(dim=1, keepdim=True)  # Average the pair of frames

        # Compute variance and gradient as before
        var_I = frame_pair.var(dim=0)
        GradTensor = imgradient_torch(frame_pair)
        GradTensorMean = GradTensor.mean(dim=0)
        GradTensorMean[GradTensorMean < GradTensorMean.max() / 100] = GradTensorMean.max() / 100

        # Calculate Pseudo-Cn2 for the pair
        constant = 1e4
        Pseudo_cn2_map = var_I / GradTensorMean * constant
        Pseudo_cn2_val = torch.quantile(Pseudo_cn2_map, 0.1)
        cn2_values.append(Pseudo_cn2_val.item())

    # Compute the median of the Cn2 values
    median_cn2 = np.median(cn2_values)
    return np.round(median_cn2, 2)


# Loading the Model and Inference
def load_and_predict(image_numpy, restormer_weight_path):
    # Initialize and load the model
    num_blocks, num_heads, channels, num_refinement, expansion_factor = [4, 6, 6, 8], [1, 2, 4, 8], [48, 96, 192, 384], 4, 2.66
    model = Restormer(num_blocks, num_heads, channels, num_refinement, expansion_factor).cuda()
    model.load_state_dict(torch.load(restormer_weight_path))
    model.eval()

    # Process the image
    rain = torch.from_numpy(image_numpy).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
    h, w = rain.shape[2:]
    pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
    rain = torch.nn.functional.pad(rain, (0, pad_w, 0, pad_h), 'reflect')

    # Predict and return the output
    with torch.no_grad():
        pred_Sharp = torch.clamp(model(rain)[:, :, :h, :w], 0, 1).mul(255).byte()

    pred_Sharp = pred_Sharp.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return pred_Sharp