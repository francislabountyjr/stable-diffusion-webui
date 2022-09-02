# utility functions
import torch
import math
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as T
import py3d_tools as p3dT
import disco_xform_utils as dxf
from PIL import Image
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from einops import rearrange
from skimage.exposure import match_histograms


def make_callback(sampler, dynamic_threshold=0, static_threshold=0, inpainting=False, mix_with_x0=False, mix_factor=[0.15, 0.30, 0.60, 1.0], x0=None, noise=None, mask=None):  
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image after each step
    def dynamic_thresholding_(img, threshold): # check over implementation to ensure correctness
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
        s = np.max(np.append(s,1.0))
        # torch.clamp_(img, -1*s, s) # this causes images to become grey/brown - investigate
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback(args_dict):
        if static_threshold != 0:
            torch.clamp_(args_dict['x'], -1*static_threshold, static_threshold)
        if dynamic_threshold != 0:
            dynamic_thresholding_(args_dict['x'], dynamic_threshold)
        if inpainting and x0 is not None and mask is not None and noise is not None:
            x = x0 + noise * args_dict['sigma']
            x = x * mask
            torch.FloatTensor.add_(torch.FloatTensor.mul_(args_dict['x'], (1. - mask)), x)
        if mix_with_x0 and x0 is not None and noise is not None:
            x = x0 + noise * args_dict['sigma']
            factor = min(mix_factor[min(args_dict['i'], len(mix_factor)-1)], 1.0)
            torch.FloatTensor.add_(torch.FloatTensor.mul_(args_dict['x'], factor), x * (1.0 - factor))

    # Function that is called on the image (img) and step (i) at each step
    def img_callback(img, i):
        # Thresholding functions
        if dynamic_threshold != 0:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold != 0:
            torch.clamp_(img, -1*static_threshold, static_threshold)

    if sampler in ["PLMS","DDIM"]: 
        # Callback function formated for compvis latent diffusion samplers
        callback = img_callback
    else: 
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback

    return callback


def slerp(val, low, high):
    low_norm = low/torch.norm(low)
    high_norm = high/torch.norm(high)
    omega = torch.acos((low_norm*high_norm).sum())
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so)*low + (torch.sin(val*omega)/so) * high
    return res


def get_slerp_vectors(start, end, device, frames=20):
    factor = 1.0 / (frames - 1)
    out = torch.Tensor(frames, start.shape[0]).to(device)
    for i in range(frames):
        out[i] = slerp(factor*i, start, end)
    return out


def unflatten(l, n):
    res = []
    t = l[:]
    while len(t) > 0:
        res.append(t[:n])
        t = t[n:]
    return res


def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    elif mode == 'RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    else:
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


def add_noise(sample, noise_amt):
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def sample_from_cv2(sample):
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample


def sample_to_cv2(sample):
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255).astype(np.uint8)
    return sample_int8


def sample_from_pil(sample):
    sample = ((np.asarray(sample).astype(float) / 255.0) * 2) - 1
    sample = torch.from_numpy(sample)
    return sample


def sample_to_pil(sample):
    sample_f32 = sample.squeeze().cpu().numpy().astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255).astype(np.uint8)
    return Image.fromarray(sample_int8)


def init_midas_depth_model(device):
    midas_model = None
    net_w = None
    net_h = None
    resize_mode = None
    normalization = None

    print(f"Initializing MiDaS depth model...")
    # load network
    midas_model_path = 'models/depth/midas/model.ckpt'
    midas_model = DPTDepthModel(
        path=midas_model_path,
        backbone="vitl16_384",
        non_negative=True,
    )
    net_w, net_h = 384, 384
    resize_mode = "minimal"
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    midas_transform = T.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    midas_model.eval()

    midas_model = midas_model.to(memory_format=torch.channels_last)  
    midas_model = midas_model.half()

    midas_model.to(device)

    print(f"MiDaS depth model initialized.")
    return midas_model, midas_transform, net_w, net_h, resize_mode, normalization


def do_3d_step(img_filepath, frame_num, midas_model, midas_transform, midas_weight,
               translation_x_series, translation_y_series, translation_z_series,
               rotation_3d_x_series, rotation_3d_y_series, rotation_3d_z_series,
               near_plane, far_plane, fov, padding_mode, sampling_mode, device,
               TRANSLATION_SCALE):
    translation_x = translation_x_series[frame_num]
    translation_y = translation_y_series[frame_num]
    translation_z = translation_z_series[frame_num]
    rotation_3d_x = rotation_3d_x_series[frame_num]
    rotation_3d_y = rotation_3d_y_series[frame_num]
    rotation_3d_z = rotation_3d_z_series[frame_num]
    print(
        f'translation_x: {translation_x}',
        f'translation_y: {translation_y}',
        f'translation_z: {translation_z}',
        f'rotation_3d_x: {rotation_3d_x}',
        f'rotation_3d_y: {rotation_3d_y}',
        f'rotation_3d_z: {rotation_3d_z}',
    )

    translate_xyz = [-translation_x*TRANSLATION_SCALE, translation_y*TRANSLATION_SCALE, -translation_z*TRANSLATION_SCALE]
    rotate_xyz_degrees = [rotation_3d_x, rotation_3d_y, rotation_3d_z]
    print('translation:',translate_xyz)
    print('rotation:',rotate_xyz_degrees)
    rotate_xyz = [math.radians(rotate_xyz_degrees[0]), math.radians(rotate_xyz_degrees[1]), math.radians(rotate_xyz_degrees[2])]
    rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    print("rot_mat: " + str(rot_mat))
    next_step_pil = dxf.transform_image_3d(img_filepath, midas_model, midas_transform, device,
                                          rot_mat, translate_xyz, near_plane, far_plane,
                                          fov, padding_mode=padding_mode,
                                          sampling_mode=sampling_mode, midas_weight=midas_weight)
    return next_step_pil


def parse_key_frames(string, prompt_parser=None):
    """Given a string representing frame numbers paired with parameter values at that frame,
    return a dictionary with the frame numbers as keys and the parameter values as the values.

    Parameters
    ----------
    string: string
        Frame numbers paired with parameter values at that frame number, in the format
        'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
    prompt_parser: function or None, optional
        If provided, prompt_parser will be applied to each string of parameter values.

    Returns
    -------
    dict
        Frame numbers as keys, parameter values at that frame number as values

    Raises
    ------
    RuntimeError
        If the input string does not match the expected format.

    Examples
    --------
    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
    {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
    {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
    """
    import re
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param

    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


def get_inbetweens(key_frames, max_frames, interp_spline, integer=False):
    """Given a dict with frame numbers as keys and a parameter value as values,
    return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
    Any values not provided in the input dict are calculated by linear interpolation between
    the values of the previous and next provided frames. If there is no previous provided frame, then
    the value is equal to the value of the next provided frame, or if there is no next provided frame,
    then the value is equal to the value of the previous provided frame. If no frames are provided,
    all frame values are NaN.

    Parameters
    ----------
    key_frames: dict
        A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
    integer: Bool, optional
        If True, the values of the output series are converted to integers.
        Otherwise, the values are floats.

    Returns
    -------
    pd.Series
        A Series with length max_frames representing the parameter values for each frame.

    Examples
    --------
    >>> max_frames = 5
    >>> get_inbetweens({1: 5, 3: 6})
    0    5.0
    1    5.0
    2    5.5
    3    6.0
    4    6.0
    dtype: float64

    >>> get_inbetweens({1: 5, 3: 6}, integer=True)
    0    5
    1    5
    2    5
    3    6
    4    6
    dtype: int64
    """
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    interp_method = interp_spline

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'

    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(
        method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


def generate_eye_views(trans_scale, filename, frame_num,midas_model, midas_transform, midas_weight, vr_eye_angle, vr_ipd, device, near_plane, far_plane, fov, padding_mode, sampling_mode):
   for i in range(2):
      theta = vr_eye_angle * (math.pi/180)
      ray_origin = math.cos(theta) * vr_ipd / 2 * (-1.0 if i==0 else 1.0)
      ray_rotation = (theta if i==0 else -theta)
      translate_xyz = [-(ray_origin)*trans_scale, 0,0]
      rotate_xyz = [0, (ray_rotation), 0]
      rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
      transformed_image = dxf.transform_image_3d(filename, midas_model, midas_transform, device,
                                                      rot_mat, translate_xyz, near_plane, far_plane,
                                                      fov, padding_mode=padding_mode,
                                                      sampling_mode=sampling_mode, midas_weight=midas_weight,spherical=True)
      eye_file_path = '/'.join(filename.split('/')[:-1])+f"/frame_{frame_num}" + ('_l' if i==0 else '_r')+'.png'
      transformed_image.save(eye_file_path)


# # Video Init

# TAG_CHAR = np.array([202021.25], np.float32)
  
# def writeFlow(filename,uv,v=None):
#     """ 
#     https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py
#     Copyright 2017 NVIDIA CORPORATION

#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at

#         http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
    
#     Write optical flow to file.
    
#     If v is None, uv is assumed to contain both u and v channels,
#     stacked in depth.
#     Original code by Deqing Sun, adapted from Daniel Scharstein.
#     """
#     nBands = 2

#     if v is None:
#         assert(uv.ndim == 3)
#         assert(uv.shape[2] == 2)
#         u = uv[:,:,0]
#         v = uv[:,:,1]
#     else:
#         u = uv

#     assert(u.shape == v.shape)
#     height,width = u.shape
#     f = open(filename,'wb')
#     # write the header
#     f.write(TAG_CHAR)
#     np.array(width).astype(np.int32).tofile(f)
#     np.array(height).astype(np.int32).tofile(f)
#     # arrange into matrix form
#     tmp = np.zeros((height, width*nBands))
#     tmp[:,np.arange(width)*2] = u
#     tmp[:,np.arange(width)*2 + 1] = v
#     tmp.astype(np.float32).tofile(f)
#     f.close()

# def load_img(img, size):
#     img = Image.open(img).convert('RGB').resize(size)
#     return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...].cuda()

# def get_flow(frame1, frame2, model, iters=20):
#     padder = InputPadder(frame1.shape)
#     frame1, frame2 = padder.pad(frame1, frame2)
#     _, flow12 = model(frame1, frame2, iters=iters, test_mode=True)
#     flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

#     return flow12

# def warp_flow(img, flow):
#     h, w = flow.shape[:2]
#     flow = flow.copy()
#     flow[:, :, 0] += np.arange(w)
#     flow[:, :, 1] += np.arange(h)[:, np.newaxis]
#     res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
#     return res

# def makeEven(_x):
#     return _x if (_x % 2 == 0) else _x+1

# def fit(img,maxsize=512):
#     maxdim = max(*img.size)
#     if maxdim>maxsize:
#         # if True:
#         ratio = maxsize/maxdim
#         x,y = img.size
#         size = (makeEven(int(x*ratio)),makeEven(int(y*ratio))) 
#         img = img.resize(size)
#     return img

# def warp(frame1, frame2, flo_path, blend=0.5, weights_path=None):
#     flow21 = np.load(flo_path)
#     frame1pil = np.array(frame1.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
#     frame1_warped21 = warp_flow(frame1pil, flow21)
#     # frame2pil = frame1pil
#     frame2pil = np.array(frame2.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))

#     if weights_path:
#         # TBD
#         pass
#     else:
#         blended_w = frame2pil*(1-blend) + frame1_warped21*(blend)

#     return  Image.fromarray(blended_w.astype('uint8'))
