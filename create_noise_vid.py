from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
from PIL import Image


from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from vid_helper import ctc_2_track


#apply_canny = CannyDetector()

'''def copy_paste_roi(img, coord0, coord1, r):
    # Ensure the coordinates are integers
    coord0 = (int(coord0[0]), int(coord0[1]))
    coord1 = (int(coord1[0]), int(coord1[1]))

    # Define the ranges for the source region
    x0_start, x0_end = max(0, coord0[0] - r), min(img.shape[0], coord0[0] + r)
    y0_start, y0_end = max(0, coord0[1] - r), min(img.shape[1], coord0[1] + r)
    
    # Define the ranges for the destination region
    x1_start, x1_end = max(0, coord1[0] - r), min(img.shape[0], coord1[0] + r)
    y1_start, y1_end = max(0, coord1[1] - r), min(img.shape[1], coord1[1] + r)

    # Extract the region of interest from the source location
    roi = img[x0_start:x0_end, y0_start:y0_end].clone()

    # If the size of the region varies due to edges of the image, we need to handle that
    x1_size = x1_end - x1_start
    y1_size = y1_end - y1_start
    roi = roi[:x1_size, :y1_size]

    # Paste the ROI to the destination location
    img[x1_start:x1_end, y1_start:y1_end] = roi

    return img
'''


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model, ddim_sampler):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape
        print('ips',prompt, a_prompt, n_prompt)

        detected_map = img
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

def make_init_pic(input_dir, output_dir, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    
    resume_path = '/export/data/msturm/CNet/last.ckpt'


    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    
    
    # Get all JPG images from the input directory
    image_paths = [os.path.join(input_dir, img) for img in os.listdir(input_dir) if img.endswith(".jpg")]

    # Sort image paths based on the integer in the file name
    image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Get the image with the highest number
    image_path = image_paths[-1]

    input_image = np.array(Image.open(image_path))  # Read image using Pillow

    result = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model, ddim_sampler)[1]

    output_img = Image.fromarray(result.astype('uint8'))  # Convert result to PIL Image
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + f'.png')
    print('processing', image_path, output_path )
    output_img.save(output_path)  # Save the image to the output directory

    img_number = int(os.path.splitext(os.path.basename(image_path))[0])

    return img_number

def make_vid(num, id_path, res_path, cond_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    resume_path = '/export/data/msturm/CNet_track/last.ckpt'


    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    for i in range(num, 1, -1):

        ctc_2_track(i, res_path, id_path, cond_path)
        image_paths = [os.path.join(cond_path, img) for img in os.listdir(cond_path) if img.endswith(".jpg")]

        # Sort image paths based on the integer in the file name
        image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # Get the image with the lowest number
        image_path = image_paths[0]

        input_image = np.array(Image.open(image_path))  # Read image using Pillow # Read image using Pillow

        result = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model, ddim_sampler)[1]

        output_img = Image.fromarray(result.astype('uint8'))  # Convert result to PIL Image
        output_file_num = int(os.path.splitext(os.path.basename(image_path))[0]) - 1
        output_path = os.path.join(res_path, str(output_file_num) + f'.png')
        print('processing', image_path, output_path )
        output_img.save(output_path)  # Save the image to the output directory





input_dir = './sampling/dots2CNet/id/' # Replace this with your input images directory
output_dir = './sampling/dots2CNet/res_track/' # Replace this with your output images directory

prompt = "fluo_ctc, cell, microscopy image, grayscale"  # Replace this with your prompt
a_prompt =''
n_prompt = ''
num_samples = 1
image_resolution = 512
ddim_steps = 100
guess_mode = False
strength = 1.0
scale = 4.0
seed = 1554547164
eta = 0.0
low_threshold = 100
high_threshold = 200


num = make_init_pic(input_dir, output_dir, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)


prompt = "fluo_ctc, cell, tracking, microscopy image, grayscale"  # Replace this with your prompt

id_path = './sampling/dots2CNet/id_track/'
res_path = './sampling/dots2CNet/res_track/'
cond_path = './sampling/dots2CNet/track_cond/'



make_vid(num, id_path, res_path, cond_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
