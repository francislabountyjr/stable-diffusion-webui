import argparse, os, sys, glob, re

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default=None)
parser.add_argument("--outdir_txt2img", type=str, nargs="?", help="dir to write txt2img results to (overrides --outdir)", default=None)
parser.add_argument("--outdir_img2img", type=str, nargs="?", help="dir to write img2img results to (overrides --outdir)", default=None)
parser.add_argument("--outdir_txt_interp", type=str, nargs="?", help="dir to write text_interp results to (overrides --outdir)", default=None)
parser.add_argument("--outdir_disco_anim", type=str, nargs="?", help="dir to write disco_anim results to (overrides --outdir)", default=None)
parser.add_argument("--save-metadata", action='store_true', help="Whether to embed the generation parameters in the sample images", default=False)
parser.add_argument("--skip-grid", action='store_true', help="do not save a grid, only individual samples. Helpful when evaluating lots of samples", default=False)
parser.add_argument("--skip-save", action='store_true', help="do not save indiviual samples. For speed measurements.", default=False)
parser.add_argument("--grid-format", type=str, help="png for lossless png files; jpg:quality for lossy jpeg; webp:quality for lossy webp, or webp:-compression for lossless webp", default="jpg:95")
parser.add_argument("--n_rows", type=int, default=-1, help="rows in the grid; use -1 for autodetect and 0 for n_rows to be same as batch_size (default: -1)",)
parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model",)
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--optimized", action='store_true', help="load the model onto the device piecemeal instead of all at once to reduce VRAM usage at the cost of performance")
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN')) # i disagree with where you're putting it but since all guidefags are doing it this way, there you go
parser.add_argument("--realesrgan-dir", type=str, help="RealESRGAN directory", default=('./src/realesrgan' if os.path.exists('./src/realesrgan') else './RealESRGAN'))
parser.add_argument("--realesrgan-model", type=str, help="Upscaling model for RealESRGAN", default=('RealESRGAN_x4plus'))
parser.add_argument("--no-verify-input", action='store_true', help="do not verify input to check if it's too long", default=False)
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats", default=False)
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)", default=False)
parser.add_argument("--share", action='store_true', help="Should share your server on gradio.app, this allows you to use the UI from your mobile app", default=False)
parser.add_argument("--share-password", type=str, help="Sharing is open by default, use this to set a password. Username: webui", default=None)
parser.add_argument("--defaults", type=str, help="path to configuration file providing UI defaults, uses same format as cli parameter", default='configs/webui/webui.yaml')
parser.add_argument("--gpu", type=int, help="choose which GPU to use if you have multiple", default=int(os.environ.get('CUDA_VISIBLE_DEVICES', 0)))
parser.add_argument("--extra-models-cpu", action='store_true', help="run extra models (GFGPAN/ESRGAN) on cpu", default=False)
parser.add_argument("--esrgan-cpu", action='store_true', help="run ESRGAN on cpu", default=False)
parser.add_argument("--gfpgan-cpu", action='store_true', help="run GFPGAN on cpu", default=False)
parser.add_argument("--cli", type=str, help="don't launch web server, take Python function kwargs from this file.", default=None)
opt = parser.parse_args()

# this should force GFPGAN and RealESRGAN onto the selected gpu as well
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

sys.path.append('E:/Python/k-diffusion')
sys.path.append('E:/Python/stable-diffusion')
sys.path.append('E:/Python/taming-transformers')
sys.path.append('E:/Python/AdaBins')
sys.path.append('E:/Python/MiDaS/midas_utils')
sys.path.append('E:/Python/MiDaS') 
sys.path.append('E:/Python/pytorch3d-lite')
sys.path.append('E:/Python/disco-diffusion')

import gradio as gr
import k_diffusion as K
import math
import shutil
import mimetypes
import numpy as np
import pynvml
import random
import threading, asyncio
import time
import cv2
import gc
import torch
import torch.nn as nn
import yaml
import glob
import imageio
from typing import List, Union
from pathlib import Path
from tqdm.auto import tqdm

from pytorch_lightning import seed_everything
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from itertools import islice
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
import base64
import re
from torch import autocast
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging
    logging.set_verbosity_error()
except:
    pass

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
invalid_filename_chars = '<>:"/\|?*\n'

GFPGAN_dir = opt.gfpgan_dir
RealESRGAN_dir = opt.realesrgan_dir

# should probably be moved to a settings menu in the UI at some point
grid_format = [s.lower() for s in opt.grid_format.split(':')]
grid_lossless = False
grid_quality = 100
if grid_format[0] == 'png':
    grid_ext = 'png'
    grid_format = 'png'
elif grid_format[0] in ['jpg', 'jpeg']:
    grid_quality = int(grid_format[1]) if len(grid_format) > 1 else 100
    grid_ext = 'jpg'
    grid_format = 'jpeg'
elif grid_format[0] == 'webp':
    grid_quality = int(grid_format[1]) if len(grid_format) > 1 else 100
    grid_ext = 'webp'
    grid_format = 'webp'
    if grid_quality < 0: # e.g. webp:-100 for lossless mode
        grid_lossless = True
        grid_quality = abs(grid_quality)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_sd_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def crash(e, s):
    global model
    global device

    print(s, '\n', e)

    del model
    del device

    print('exiting...calling os._exit(0)')
    t = threading.Timer(0.25, os._exit, args=[0])
    t.start()

class MemUsageMonitor(threading.Thread):
    stop_flag = False
    max_usage = 0
    total = -1

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        try:
            pynvml.nvmlInit()
        except:
            print(f"[{self.name}] Unable to initialize NVIDIA management. No memory stats. \n")
            return
        print(f"[{self.name}] Recording max memory usage...\n")
        handle = pynvml.nvmlDeviceGetHandleByIndex(opt.gpu)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        while not self.stop_flag:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.max_usage = max(self.max_usage, m.used)
            # print(self.max_usage)
            time.sleep(0.1)
        print(f"[{self.name}] Stopped recording.\n")
        pynvml.nvmlShutdown()

    def read(self):
        return self.max_usage, self.total

    def stop(self):
        self.stop_flag = True

    def read_and_stop(self):
        self.stop_flag = True
        return self.max_usage, self.total

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class KDiffusionSampler:
    def __init__(self, m, sampler):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler

    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T, img_callback):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)

        samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}, disable=False, callback=img_callback)

        return samples_ddim, None


def create_random_tensors(shape, seeds):
    xs = []
    for seed in seeds:
        torch.manual_seed(seed)
        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this so i do not dare change it for now because
        # it will break everyone's seeds.
        xs.append(torch.randn(shape, device=device))
    x = torch.stack(xs)
    return x

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def load_GFPGAN():
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(GFPGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception("GFPGAN model not found at path "+model_path)

    sys.path.append(os.path.abspath(GFPGAN_dir))
    from gfpgan import GFPGANer
    instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
    if opt.gfpgan_cpu or opt.extra_models_cpu:
        instance.device = torch.device('cpu')
    else:
        instance.device = torch.device(f'cuda:{opt.gpu}') # another way to set gpu device
    return instance

def load_RealESRGAN(model_name: str):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    RealESRGAN_models = {
        'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    }

    model_path = os.path.join(RealESRGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception(model_name+".pth not found at path "+model_path)

    sys.path.append(os.path.abspath(RealESRGAN_dir))
    from realesrgan import RealESRGANer

    if opt.esrgan_cpu or opt.extra_models_cpu:
        instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=False)
        instance.model.name = model_name
        instance.device = torch.device('cpu')
        instance.device = torch.device('cpu')
        instance.model.to('cpu')
    else:
        instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=not opt.no_half)
        instance.model.name = model_name
        instance.device = torch.device(f'cuda:{opt.gpu}') # another way to set gpu device

    return instance

GFPGAN = None
if os.path.exists(GFPGAN_dir):
    try:
        GFPGAN = load_GFPGAN()
        print("Loaded GFPGAN")
    except Exception:
        import traceback
        print("Error loading GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

RealESRGAN = None
def try_loading_RealESRGAN(model_name: str):
    global RealESRGAN
    if os.path.exists(RealESRGAN_dir):
        try:
            RealESRGAN = load_RealESRGAN(model_name) # TODO: Should try to load both models before giving up
            print("Loaded RealESRGAN with model "+RealESRGAN.model.name)
        except Exception:
            import traceback
            print("Error loading RealESRGAN:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
try_loading_RealESRGAN('RealESRGAN_x4plus')

if opt.optimized:
    sd = load_sd_from_config("models/ldm/stable-diffusion-v1/model.ckpt")
    li, lo = [], []
    for key, v_ in sd.items():
        sp = key.split('.')
        if(sp[0]) == 'model':
            if('input_blocks' in sp):
                li.append(key)
            elif('middle_block' in sp):
                li.append(key)
            elif('time_embed' in sp):
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd['model1.' + key[6:]] = sd.pop(key)
    for key in lo:
        sd['model2.' + key[6:]] = sd.pop(key)

    config = OmegaConf.load("optimizedSD/v1-inference.yaml")
    config.modelUNet.params.small_batch = False

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
        
    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model if opt.no_half else model.half()
    modelCS = modelCS if opt.no_half else modelCS.half()
else:
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = (model if opt.no_half else model.half()).to(device)

def load_embeddings(fp):
    if fp is not None and hasattr(model, "embedding_manager"):
        model.embedding_manager.load(fp.name)


def get_font(fontsize):
    fonts = ["arial.ttf", "DejaVuSans.ttf"]
    for font_name in fonts:
        try:
            return ImageFont.truetype(font_name, fontsize)
        except OSError:
           pass

    # ImageFont.load_default() is practically unusable as it only supports
    # latin1, so raise an exception instead if no usable font was found
    raise Exception(f"No usable font found (tried {', '.join(fonts)})")

def image_grid(imgs, batch_size, force_n_rows=None, captions=None):
    if force_n_rows is not None:
        rows = force_n_rows
    elif opt.n_rows > 0:
        rows = opt.n_rows
    elif opt.n_rows == 0:
        rows = batch_size
    else:
        rows = math.sqrt(len(imgs))
        rows = round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    fnt = get_font(30)

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
        if captions:
            d = ImageDraw.Draw( grid )
            size = d.textbbox( (0,0), captions[i], font=fnt, stroke_width=2, align="center" )
            d.multiline_text((i % cols * w + w/2, i // cols * h + h - size[3]), captions[i], font=fnt, fill=(255,0,255), stroke_width=2, stroke_fill=(0,0,0), anchor="mm", align="center")

    return grid

def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == '' or s.lower() == 'none':
        return random.randint(0, 2**32 - 1)
    n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2**32 - 1))
    while n >= 2**32:
        n = n >> 32
    return n

def draw_prompt_matrix(im, width, height, all_prompts):
    def wrap(text, d, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if d.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return '\n'.join(lines)

    def draw_texts(pos, x, y, texts, sizes):
        for i, (text, size) in enumerate(zip(texts, sizes)):
            active = pos & (1 << i) != 0

            if not active:
                text = '\u0336'.join(text) + '\u0336'

            d.multiline_text((x, y + size[1] / 2), text, font=fnt, fill=color_active if active else color_inactive, anchor="mm", align="center")

            y += size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    fnt = get_font(fontsize)
    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_top = height // 4
    pad_left = width * 3 // 4 if len(all_prompts) > 2 else 0

    cols = im.width // width
    rows = im.height // height

    prompts = all_prompts[1:]

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)

    boundary = math.ceil(len(prompts) / 2)
    prompts_horiz = [wrap(x, d, fnt, width) for x in prompts[:boundary]]
    prompts_vert = [wrap(x, d, fnt, pad_left) for x in prompts[boundary:]]

    sizes_hor = [(x[2] - x[0], x[3] - x[1]) for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_horiz]]
    sizes_ver = [(x[2] - x[0], x[3] - x[1]) for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_vert]]
    hor_text_height = sum([x[1] + line_spacing for x in sizes_hor]) - line_spacing
    ver_text_height = sum([x[1] + line_spacing for x in sizes_ver]) - line_spacing

    for col in range(cols):
        x = pad_left + width * col + width / 2
        y = pad_top / 2 - hor_text_height / 2

        draw_texts(col, x, y, prompts_horiz, sizes_hor)

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + height * row + height / 2 - ver_text_height / 2

        draw_texts(row, x, y, prompts_vert, sizes_ver)

    return result


def resize_image(resize_mode, im, width, height):
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res


def check_prompt_length(prompt, comments):
    """this function tests if prompt is too long, and if so, adds a message to comments"""

    tokenizer = (model if not opt.optimized else modelCS).cond_stage_model.tokenizer
    max_length = (model if not opt.optimized else modelCS).cond_stage_model.max_length

    info = (model if not opt.optimized else modelCS).cond_stage_model.tokenizer([prompt], truncation=True, max_length=max_length, return_overflowing_tokens=True, padding="max_length", return_tensors="pt")
    ovf = info['overflowing_tokens'][0]
    overflowing_count = ovf.shape[0]
    if overflowing_count == 0:
        return

    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    overflowing_words = [vocab.get(int(x), "") for x in ovf]
    overflowing_text = tokenizer.convert_tokens_to_string(''.join(overflowing_words))

    comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

def save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode):
    filename_i = os.path.join(sample_path_i, filename)
    if not jpg_sample:
        if opt.save_metadata:
            metadata = PngInfo()
            metadata.add_text("SD:prompt", prompts[i] if prompts is not None else '')
            metadata.add_text("SD:seed", str(seeds[i] if seeds is not None else ''))
            metadata.add_text("SD:width", str(width))
            metadata.add_text("SD:height", str(height))
            metadata.add_text("SD:steps", str(steps))
            metadata.add_text("SD:cfg_scale", str(cfg_scale))
            metadata.add_text("SD:GFPGAN", str(use_GFPGAN and GFPGAN is not None))
            image.save(f"{filename_i}.png", pnginfo=metadata)
        else:
            image.save(f"{filename_i}.png")
    else:
        image.save(f"{filename_i}.jpg", 'jpeg', quality=100, optimize=True)
    if write_info_files:
        # toggles differ for txt2img vs. img2img:
        offset = 0 if init_img is None else 2
        toggles = []
        if prompt_matrix:
            toggles.append(0)
        if init_img is not None:
            if uses_loopback:
                toggles.append(1)
            if uses_random_seed_loopback:
                toggles.append(2)
        if not skip_save:
            toggles.append(1 + offset)
        if not skip_grid:
            toggles.append(2 + offset)
        if sort_samples:
            toggles.append(3 + offset)
        if write_info_files:
            toggles.append(4 + offset)
        if use_GFPGAN:
            toggles.append(5 + offset)
        info_dict = dict(
            target="txt2img" if init_img is None else "img2img",
            prompt=prompts[i], ddim_steps=steps, toggles=toggles, sampler_name=sampler_name,
            ddim_eta=ddim_eta, n_iter=n_iter, batch_size=batch_size, cfg_scale=cfg_scale,
            seed=seeds[i], width=width, height=height
        )
        if init_img is not None:
            # Not yet any use for these, but they bloat up the files:
            #info_dict["init_img"] = init_img
            #info_dict["init_mask"] = init_mask
            info_dict["denoising_strength"] = denoising_strength
            info_dict["resize_mode"] = resize_mode
        with open(f"{filename_i}.yaml", "w", encoding="utf8") as f:
            yaml.dump(info_dict, f, allow_unicode=True)


def get_next_sequence_number(path, prefix=''):
    """
    Determines and returns the next sequence number to use when saving an
    image in the specified directory.

    If a prefix is given, only consider files whose names start with that
    prefix, and strip the prefix from filenames before extracting their
    sequence number.

    The sequence starts at 0.
    """
    result = -1
    for p in Path(path).iterdir():
        if p.name.endswith(('.png', '.jpg')) and p.name.startswith(prefix):
            tmp = p.name[len(prefix):]
            try:
                result = max(int(tmp.split('-')[0]), result)
            except ValueError:
                pass
    return result + 1

def oxlamon_matrix(prompt, seed, batch_size):
    pattern = re.compile(r'(,\s){2,}')

    class PromptItem:
        def __init__(self, text, parts, item):
            self.text = text
            self.parts = parts
            if item:
                self.parts.append( item )

    def clean(txt):
        return re.sub(pattern, ', ', txt)

    def repliter( txt ):
        for data in re.finditer( ".*?\\((.*?)\\).*", txt ):
            if data:
                r = data.span(1)
                for item in data.group(1).split("|"):
                    yield (clean(txt[:r[0]-1] + item.strip() + txt[r[1]+1:]), item.strip())
            break

    def iterlist( items ):
        outitems = []
        for item in items:
            for newitem, newpart in repliter(item.text):
                outitems.append( PromptItem(newitem, item.parts.copy(), newpart) )

        return outitems

    def getmatrix( prompt ):
        dataitems = [ PromptItem( prompt[1:].strip(), [], None ) ]
        while True:
            newdataitems = iterlist( dataitems )
            if len( newdataitems ) == 0:
                return dataitems
            dataitems = newdataitems

    def classToArrays( items ):
        texts = []
        parts = []

        for item in items:
            texts.append( item.text )
            parts.append( "\n".join(item.parts) )        
        return texts, parts

    all_prompts, prompt_matrix_parts = classToArrays(getmatrix( prompt ))
    n_iter = math.ceil(len(all_prompts) / batch_size)
    all_seeds = len(all_prompts) * [seed]
    return all_seeds, n_iter, prompt_matrix_parts, all_prompts



def process_images(
        outpath, func_init, func_sample, prompt, seed, sampler_name, skip_grid, skip_save, batch_size,
        n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN, realesrgan_model_name,
        fp, ddim_eta=0.0, do_not_save_grid=False, init_img=None, init_mask=None,
        keep_mask=False, mask_blur_strength=3, denoising_strength=0.75, resize_mode=None, uses_loopback=False,
        uses_random_seed_loopback=False, sort_samples=True, write_info_files=True, jpg_sample=False, do_interpolation=False,
        project_name='interp', fps=30):
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    assert prompt is not None
    project_name = project_name if project_name != '' else 'interp'
    torch_gc()
    # start time after garbage collection (or before?)
    start_time = time.time()

    mem_mon = MemUsageMonitor('MemMon')
    mem_mon.start()

    if hasattr(model, "embedding_manager"):
        load_embeddings(fp)

    os.makedirs(outpath, exist_ok=True)
    if do_interpolation:
        os.makedirs(outpath + '/frames', exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    if not do_interpolation:
        os.makedirs(sample_path, exist_ok=True)

    comments = []

    prompt_matrix_parts = []
    if prompt_matrix:
        if prompt.startswith("@"):
            all_seeds, n_iter, prompt_matrix_parts, all_prompts = oxlamon_matrix(prompt, seed, batch_size)
        else:
            all_prompts = []
            prompt_matrix_parts = prompt.split("|")
            combination_count = 2 ** (len(prompt_matrix_parts) - 1)
            for combination_num in range(combination_count):
                current = prompt_matrix_parts[0]

                for n, text in enumerate(prompt_matrix_parts[1:]):
                    if combination_num & (2 ** n) > 0:
                        current += ("" if text.strip().startswith(",") else ", ") + text

                all_prompts.append(current)

            n_iter = math.ceil(len(all_prompts) / batch_size)
            all_seeds = len(all_prompts) * [seed]

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {n_iter} batches.")
    else:

        if not opt.no_verify_input and not do_interpolation:
            try:
                check_prompt_length(prompt, comments)
            except:
                import traceback
                print("Error verifying input:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        if not do_interpolation:
            all_prompts = batch_size * n_iter * [prompt]
            all_seeds = [seed + x for x in range(len(all_prompts))]
        elif do_interpolation:
            all_prompts = prompt
            all_seeds = seed

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    output_images = []
    stats = []
    frame = 1
    if do_interpolation and not skip_grid:
        video_out = imageio.get_writer(f"{outpath}/{project_name}.mp4", mode='I', fps=fps, codec='libx264')
    with torch.no_grad(), precision_scope("cuda"), (model.ema_scope() if not opt.optimized else nullcontext()):
        init_data = func_init()
        tic = time.time()

        for n in range(n_iter):
            print(f"Iteration: {n+1}/{n_iter}")
            if not do_interpolation:
                prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
                seeds = all_seeds[n * batch_size:(n + 1) * batch_size]
            elif do_interpolation:
                c = torch.cat(tuple(all_prompts[n]))
                x = torch.cat(tuple(torch.stack(list(all_seeds[n]), dim=0)))

            if opt.optimized:
                modelCS.to(device)
            if not do_interpolation:
                uc = (model if not opt.optimized else modelCS).get_learned_conditioning(len(prompts) * [""])
            elif do_interpolation:
                uc = (model if not opt.optimized else modelCS).get_learned_conditioning(len(all_prompts[n]) * [""])
            if not do_interpolation and isinstance(prompts, tuple):
                prompts = list(prompts)

            if not do_interpolation:
                subprompts,weights = split_weighted_subprompts(prompts[0])
                # sub-prompt weighting used if more than 1
                if len(subprompts) > 1:
                    c = (model if not opt.optimized else modelCS).get_learned_conditioning(subprompts[0])
                    original_c_shape = c.shape
                    c = c.flatten()
                    for i in range(1,len(subprompts)):
                        weight = weights[i-1]
                        # slerp between subprompts by weight between 0-1
                        c = slerp(weight, c, (model if not opt.optimized else modelCS).get_learned_conditioning(subprompts[i]).flatten())
                    c = c.reshape(*original_c_shape)
                else: # just behave like usual
                    c = (model if not opt.optimized else modelCS).get_learned_conditioning(prompts)

            shape = [opt_C, height // opt_f, width // opt_f]

            if opt.optimized:
                mem = torch.cuda.memory_allocated()/1e6
                modelCS.to("cpu")
                while(torch.cuda.memory_allocated()/1e6 >= mem):
                    time.sleep(1)

            # we manually generate all input noises because each one should have a specific seed
            if not do_interpolation:
                x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=seeds)

            samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)

            if opt.optimized:
                modelFS.to(device)

            x_samples_ddim = (model if not opt.optimized else modelFS).decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            for i, x_sample in enumerate(x_samples_ddim):
                if not do_interpolation:
                    sanitized_prompt = prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})
                    if sort_samples:
                        sanitized_prompt = sanitized_prompt[:128] #200 is too long
                        sample_path_i = os.path.join(sample_path, sanitized_prompt)
                        os.makedirs(sample_path_i, exist_ok=True)
                        base_count = get_next_sequence_number(sample_path_i)
                        filename = f"{base_count:05}-{steps}_{sampler_name}_{seeds[i]}"
                    else:
                        sample_path_i = sample_path
                        base_count = get_next_sequence_number(sample_path_i)
                        sanitized_prompt = sanitized_prompt
                        filename = f"{base_count:05}-{steps}_{sampler_name}_{seeds[i]}_{sanitized_prompt}"[:128] #same as before
                elif do_interpolation:
                    sample_path_i = os.path.join(outpath, 'frames')
                    filename = f"{project_name}_{frame}"
                    frame += 1

                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)

                if use_GFPGAN and GFPGAN is not None:
                    torch_gc()
                    original_sample = x_sample
                    original_filename = filename
                    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    x_sample = restored_img[:,:,::-1]
                    image = Image.fromarray(x_sample)
                    filename = filename + '-gfpgan'
                    save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode)
                    filename = original_filename
                    x_sample = original_sample

                if use_RealESRGAN and RealESRGAN is not None:
                    torch_gc()
                    original_sample = x_sample
                    original_filename = filename
                    if RealESRGAN.model.name != realesrgan_model_name:
                        try_loading_RealESRGAN(realesrgan_model_name)
                    output, img_mode = RealESRGAN.enhance(x_sample[:,:,::-1])
                    x_sample = output[:,:,::-1]
                    image = Image.fromarray(x_sample)
                    filename = filename + '-esrgan'
                    save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode)
                    filename = original_filename
                    x_sample = original_sample

                image = Image.fromarray(x_sample)
                # if init_mask:
                #     #init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
                #     init_mask = init_mask.filter(ImageFilter.GaussianBlur(mask_blur_strength))
                #     init_mask = init_mask.convert('L')
                #     init_img = init_img.convert('RGB')
                #     image = image.convert('RGB')

                #     if use_RealESRGAN and RealESRGAN is not None:
                #         if RealESRGAN.model.name != realesrgan_model_name:
                #             try_loading_RealESRGAN(realesrgan_model_name)
                #         output, img_mode = RealESRGAN.enhance(np.array(init_img, dtype=np.uint8))
                #         init_img = Image.fromarray(output)
                #         init_img = init_img.convert('RGB')

                #         output, img_mode = RealESRGAN.enhance(np.array(init_mask, dtype=np.uint8))
                #         init_mask = Image.fromarray(output)
                #         init_mask = init_mask.convert('L')

                #     image = Image.composite(init_img, image, init_mask)
                if not skip_save:
                    if not do_interpolation:
                        save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale, 
    use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
    skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode)
                    else:
                        save_sample(image, sample_path_i, filename, jpg_sample, None, None, width, height, steps, cfg_scale, 
    use_GFPGAN, write_info_files, False, None, False, False, skip_save,
    skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode)

            output_images.append(image)
            if do_interpolation and not skip_grid:
                video_out.append_data(x_sample)

        if (prompt_matrix or not skip_grid) and not do_not_save_grid and not do_interpolation:
            if prompt_matrix:
                grid = image_grid(output_images, batch_size, force_n_rows=1 << ((len(prompt_matrix_parts)-1)//2), captions=prompt_matrix_parts if prompt.startswith("@") else None)
            else:
                grid = image_grid(output_images, batch_size)

            if prompt_matrix and not prompt.startswith("@") and not do_interpolation:
                try:
                    grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                except:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

            output_images.insert(0, grid)
            #else:
            #    grid = image_grid(output_images, batch_size)

            grid_count = get_next_sequence_number(outpath, 'grid-')
            grid_file = f"grid-{grid_count:05}-{seed}_{prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.{grid_ext}"
            grid.save(os.path.join(outpath, grid_file), grid_format, quality=grid_quality, lossless=grid_lossless, optimize=True)

        if opt.optimized:
            mem = torch.cuda.memory_allocated()/1e6
            modelFS.to("cpu")
            while(torch.cuda.memory_allocated()/1e6 >= mem):
                time.sleep(1)

        toc = time.time()

    if do_interpolation and not skip_grid:
        video_out.close()

    mem_max_used, mem_total = mem_mon.read_and_stop()
    time_diff = time.time()-start_time

    info = f"""
{prompt if not do_interpolation else ''}
Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed if not do_interpolation else ''}{', GFPGAN' if use_GFPGAN and GFPGAN is not None else ''}{', '+realesrgan_model_name if use_RealESRGAN and RealESRGAN is not None else ''}{', Prompt Matrix Mode.' if prompt_matrix else ''}""".strip()
    stats = f'''
Took { round(time_diff, 2) }s total ({ round(time_diff/(len(all_prompts)),2) }s per image)
Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

    for comment in comments:
        info += "\n\n" + comment

    #mem_mon.stop()
    #del mem_mon
    torch_gc()

    return output_images, seed, info, stats


def process_disco_anim(outpath, func_init, func_sample, init_image, prompts, seed, sampler_name, animation_mode, start_frame, max_frames,
                       unconditional_guidance_scale, width, height, resume_run, angle_series, zoom_series, translation_x_series,
                       translation_y_series, translation_z_series, rotation_3d_x_series, rotation_3d_y_series, rotation_3d_z_series,
                       color_match, noise_between_frames, turbo_mode, turbo_preroll, turbo_steps, vr_mode, video_init_frames_scale,
                       video_init_flow_warp, videoFramesFolder, flo_folder, video_init_flow_blend, consistent_seed, color_match_mode,
                       interpolate, vr_eye_angle, vr_ipd, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode):
    TRANSLATION_SCALE = 1.0/200.0

    # initialize midas depth model
    if animation_mode == "3D":
        midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(device)

    color_match_sample = None
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    stop_on_next_loop = False
    for frame_num in tqdm(range(start_frame, max_frames), desc='Frames'):
        if stop_on_next_loop:
            break

        if animation_mode == "2D":
            angle = angle_series[frame_num]
            zoom = zoom_series[frame_num]
            translation_x = translation_x_series[frame_num]
            translation_y = translation_y_series[frame_num]
            print(
                f'angle: {angle}',
                f'zoom: {zoom}',
                f'translation_x: {translation_x}',
                f'translation_y: {translation_y}',
            )

            if frame_num > 0:
                if resume_run and frame_num == start_frame:
                    img_0 = cv2.imread(f"{outpath}/Frames/frame_{start_frame-1}.png")
                else:
                    img_0 = cv2.imread('prevFrame.png')
                    center = (1*img_0.shape[1]//2, 1*img_0.shape[0]//2)
                    trans_mat = np.float32(
                        [[1, 0, translation_x],
                        [0, 1, translation_y]]
                    )
                    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
                    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
                    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
                    transformation_matrix = np.matmul(rot_mat, trans_mat)
                    img_0 = cv2.warpPerspective(
                        img_0,
                        transformation_matrix,
                        (img_0.shape[1], img_0.shape[0]),
                        borderMode=cv2.BORDER_WRAP
                    )
                    
                # apply color matching
                if color_match:
                    if color_match_sample is None:
                        color_match_sample = img_0.copy()
                    else:
                        if color_match_mode == 'cycle':
                            img_0 = maintain_colors(img_0, color_match_sample, ['RGB','HSV','LAB'][frame_num % 3])
                        else:
                            img_0 = maintain_colors(img_0, color_match_sample, color_match_mode)

                # apply frame noising
                if noise_between_frames > 0:
                    img_0 = add_noise(sample_from_cv2(img_0), noise_between_frames)

                cv2.imwrite('prevFrameScaled.png', img_0)
                init_image = 'prevFrameScaled.png'
            
        if animation_mode == "3D":
            if frame_num > 0:
                # seed += 1
                if resume_run and frame_num == start_frame:
                    img_filepath = f"{outpath}/frame_{start_frame-1}.png"
                    if turbo_mode and frame_num > turbo_preroll:
                        shutil.copyfile(img_filepath, 'oldFrameScaled.png')
                else:
                    img_filepath = 'prevFrame.png'

                next_step_pil = do_3d_step(
                        img_filepath, frame_num, midas_model, midas_transform, midas_weight,
                        translation_x_series, translation_y_series, translation_z_series,
                        rotation_3d_x_series, rotation_3d_y_series, rotation_3d_z_series,
                        near_plane, far_plane, fov, padding_mode, sampling_mode, device,
                        TRANSLATION_SCALE
                    )
                
                # apply color matching
                if color_match:
                    if color_match_sample is None:
                        color_match_sample = Image.open(img_filepath).copy()
                    else:
                        if color_match_mode == 'cycle':
                            next_step_pil = maintain_colors(np.asarray(next_step_pil).copy(), np.asarray(color_match_sample).copy(), ['RGB','HSV','LAB'][frame_num % 3])
                        else:
                            next_step_pil = maintain_colors(np.asarray(next_step_pil).copy(), np.asarray(color_match_sample).copy(), color_match_mode)
                        next_step_pil = Image.fromarray(next_step_pil)
                
                if turbo_mode and frame_num != turbo_preroll and frame_num > turbo_preroll and frame_num % int(turbo_steps) == 0:
                    next_step_pil.save('oldFrameScaled.png') # to prevent blending in noise for turbo mode

                # apply frame noising
                if noise_between_frames > 0 and ((turbo_mode and not frame_num == turbo_preroll and frame_num % int(turbo_steps) == 0 and frame_num > turbo_preroll) or not turbo_mode):
                    next_step_pil = add_noise(sample_from_pil(next_step_pil), noise_between_frames)
                    next_step_pil = sample_to_pil(next_step_pil)

                next_step_pil.save('prevFrameScaled.png')

                # Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
                if turbo_mode:
                    if frame_num == turbo_preroll:  # start tracking oldframe
                        # stash for later blending
                        next_step_pil.save('oldFrameScaled.png')
                    elif frame_num > turbo_preroll:
                        # set up 2 warped image sequences, old & new, to blend toward new diff image
                        old_frame = do_3d_step(
                                'oldFrameScaled.png', frame_num, midas_model, midas_transform, midas_weight,
                                translation_x_series, translation_y_series, translation_z_series,
                                rotation_3d_x_series, rotation_3d_y_series, rotation_3d_z_series,
                                near_plane, far_plane, fov, padding_mode, sampling_mode, device,
                                TRANSLATION_SCALE
                            )
                        
                        old_frame.save('oldFrameScaled.png')
                        if frame_num % int(turbo_steps) != 0:
                            print(
                                'turbo skip this frame: skipping clip diffusion steps')
                            filename = f"{outpath}/Frames/frame_{frame_num}.png"
                            blend_factor = (
                                (frame_num % int(turbo_steps))+1)/int(turbo_steps)
                            print(
                                'turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                            # this is already updated..
                            newWarpedImg = cv2.imread('prevFrameScaled.png')
                            oldWarpedImg = cv2.imread('oldFrameScaled.png')
                            blendedImage = cv2.addWeighted(
                                newWarpedImg, blend_factor, oldWarpedImg, 1-blend_factor, 0.0)

                            cv2.imwrite(filename, blendedImage)
                            # save it also as prev_frame to feed next iteration
                            # next_step_pil.save(f'{img_filepath}')

                            next_step_pil.save('prevFrame.png')
                            if vr_mode:
                                generate_eye_views(
                                    TRANSLATION_SCALE, filename, frame_num, midas_model, midas_transform, midas_weight,
                                    vr_eye_angle, vr_ipd, device, near_plane, far_plane, fov, padding_mode, sampling_mode)
                            continue
                        else:
                            # if not a skip frame, will run diffusion and need to blend.
                            # oldWarpedImg = cv2.imread('prevFrameScaled.png')
                            # swap in for blending later
                            # cv2.imwrite(f'oldFrameScaled.png', oldWarpedImg)
                            print('clip/diff this frame - generate clip diff image')

                init_image = 'prevFrameScaled.png'

        # if animation_mode == "Video Input":
        #     init_scale = video_init_frames_scale
        #     skip_steps = acalc_frames_skip_steps
        #     if video_init_flow_warp:
        #         if frame_num == 0:
        #             skip_steps = video_init_skip_steps
        #             init_image = f'{videoFramesFolder}/{frame_num+1}.jpg'
        #         if frame_num > 0:
        #             prev = Image.open(f"{outpath}/frame_{frame_num-1}.png")

        #             frame1_path = f'{videoFramesFolder}/{frame_num}.jpg'
        #             frame2 = Image.open(
        #                 f'{videoFramesFolder}/{frame_num+1}.jpg')
        #             flo_path = f"/{flo_folder}/{frame1_path.split('/')[-1]}.npy"

        #             init_image = 'warped.png'
        #             print(video_init_flow_blend)
        #             weights_path = None

        #             warp(prev, frame2, flo_path, blend=video_init_flow_blend,
        #                  weights_path=weights_path).save(init_image)

        #     else:
        #         init_image = f'{videoFramesFolder}/{frame_num+1:04}.jpg'

        if consistent_seed:
            seed_everything(seed)
            torch.manual_seed(seed)
        else:
            seed_everything(seed+frame_num)
            torch.manual_seed(seed+frame_num)


        print(f'Frame {frame_num}')

        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    gc.collect()
                    torch.cuda.empty_cache()
                    init_data = func_init(init_image)
                    prompt = prompts[frame_num%len(prompts)]
                    uc = None
                    if unconditional_guidance_scale != 1.0:
                        uc = model.get_learned_conditioning([""])
                    if not interpolate and isinstance(prompt, tuple):
                        prompt = list(prompt)
                    elif interpolate:
                        c = prompt

                    if not interpolate:
                        subprompts,weights = split_weighted_subprompts(prompt)

                        # sub-prompt weighting used if more than 1
                        if len(subprompts) > 1:
                            c = (model if not opt.optimized else modelCS).get_learned_conditioning(subprompts[0])
                            original_c_shape = c.shape
                            c = c.flatten()
                            for i in range(1,len(subprompts)):
                                weight = weights[i-1]
                                # slerp between subprompts by weight between 0-1
                                c = slerp(weight, c, (model if not opt.optimized else modelCS).get_learned_conditioning(subprompts[i]).flatten())
                            c = c.reshape(*original_c_shape)
                        else: # just behave like usual
                            c = (model if not opt.optimized else modelCS).get_learned_conditioning(prompt)
                    c = torch.cat([c])

                    shape = [opt_C, height // opt_f, width // opt_f]

                    if opt.optimized:
                        mem = torch.cuda.memory_allocated()/1e6
                        modelCS.to("cpu")
                        while(torch.cuda.memory_allocated()/1e6 >= mem):
                            time.sleep(1)

                    if consistent_seed:
                        x = create_random_tensors(shape, seeds=[seed])
                    else:
                        x = create_random_tensors(shape, seeds=[seed+frame_num])

                    samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)

                    if opt.optimized:
                        modelFS.to(device)

                    x_samples_ddim = (model if not opt.optimized else modelFS).decode_first_stage(samples_ddim)[0] # grab first as disco does not batch
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    x_sample = 255. * \
                            rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    if animation_mode != "None":
                        filename = f"{outpath}/frame_{frame_num}.png"
                        img.save('prevFrame.png')
                        img.save(filename)
                        if animation_mode == "3D":
                            # If turbo, save a blended image
                            if turbo_mode and frame_num > 0:
                                # Mix new image with prevFrameScaled
                                blend_factor = (1)/int(turbo_steps)
                                # This is already updated..
                                newFrame = cv2.imread('prevFrame.png')
                                prev_frame_warped = cv2.imread(
                                    'prevFrameScaled.png')
                                blendedImage = cv2.addWeighted(
                                    newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                                cv2.imwrite(filename, blendedImage)

                            if vr_mode:
                                with precision_scope("cuda", enabled=False):
                                    generate_eye_views(
                                        TRANSLATION_SCALE, filename, frame_num, midas_model, midas_transform, midas_weight,
                                        vr_eye_angle, vr_ipd, device, near_plane, far_plane, fov, padding_mode, sampling_mode)

                    if opt.optimized:
                        mem = torch.cuda.memory_allocated()/1e6
                        modelFS.to("cpu")
                        while(torch.cuda.memory_allocated()/1e6 >= mem):
                            time.sleep(1)
    return _


def txt_interp(prompt: str, ddim_steps: int, sampler_name: str, toggles: List[int], realesrgan_model_name: str,
            ddim_eta: float, batch_size: int, cfg_scale: float, dynamic_threshold: float,
            static_threshold: float, degrees_per_second: int, frames_per_second: int, project_name: str,
            seeds: Union[int, str, None], height: int, width: int, fp):
    if opt.outdir_txt_interp != None:
        outpath = f"{opt.outdir_txt_interp}/{project_name}"
    elif opt.outdir != None:
        outpath =  f"{opt.outdir}/{project_name}"
    else:
        outpath = f"outputs/txt_interp/{project_name}"
    err = False
    seeds = [seed_to_int(seed) for seed in seeds.split('\n')]
    prompts = prompt.split('\n')
    while len(seeds) < len(prompts):
        seeds.append(seed_to_int(None))
    if len(seeds) > len(prompts):
        seeds = seeds[:len(prompts)]

    frames_per_degree = frames_per_second / degrees_per_second

    loop = 0 in toggles
    skip_save = 1 in toggles
    skip_save_mp4 = 2 in toggles
    sort_samples = 3 in toggles
    write_info_files = 4 in toggles
    jpg_sample = 5 in toggles
    use_GFPGAN = 6 in toggles
    use_RealESRGAN = 7 in toggles if GFPGAN is None else 8 in toggles # possible index shift

    def get_starting_code_and_conditioning_vector(seed, prompt):
        seed_everything(seed)
        torch.manual_seed(seed)
        start_code = torch.randn([1, opt_C, height // opt_f, width // opt_f], device=device)

        subprompts,weights = split_weighted_subprompts(prompt)

        # sub-prompt weighting used if more than 1
        if len(subprompts) > 1:
            c = (model if not opt.optimized else modelCS).get_learned_conditioning(subprompts[0])
            original_c_shape = c.shape
            c = c.flatten()
            for i in range(1,len(subprompts)): 
                weight = weights[i-1]
                # slerp between subprompts by weight between 0-1
                c = slerp(weight, c, (model if not opt.optimized else modelCS).get_learned_conditioning(subprompts[i]).flatten())
            c = c.reshape(*original_c_shape)
        else: # just behave like usual
            c = (model if not opt.optimized else modelCS).get_learned_conditioning(prompt)
        return (c, start_code)

    # interpolation setup
    previous_c = None
    previous_start_code = None
    slerp_c_vectors = []
    slerp_start_codes = []
    for i, data in enumerate(map(lambda x: get_starting_code_and_conditioning_vector(*x), zip(seeds, prompts))):
        c, start_code = data
        if i == 0:
            slerp_c_vectors.append(c)
            slerp_start_codes.append(start_code)
        else:
            start_norm = previous_c.flatten()/torch.norm(previous_c.flatten())
            end_norm = c.flatten()/torch.norm(c.flatten())
            omega = torch.acos((start_norm*end_norm).sum())
            frames_c = round(omega.item() * frames_per_degree * 57.2957795)
            # start_norm = previous_start_code.flatten()/torch.norm(previous_start_code.flatten())
            # end_norm = start_code.flatten()/torch.norm(start_code.flatten())
            # omega = torch.acos((start_norm*end_norm).sum())
            # frames_start_code = round(omega.item() * frames_per_degree * 57.2957795)

            frames = frames_c # frames = frames_c if frames_c >= frames_start_code else frames_start_code

            original_c_shape = c.shape
            original_start_code_shape = start_code.shape
            c_vectors = get_slerp_vectors(previous_c.flatten(), c.flatten(), device, frames=frames)
            c_vectors = c_vectors.reshape(-1, *original_c_shape)
            slerp_c_vectors.extend(list(c_vectors[1:])) # drop first frame to prevent repeating frames
            start_codes = get_slerp_vectors(previous_start_code.flatten(), start_code.flatten(), device, frames=frames)
            start_codes = start_codes.reshape(-1, *original_start_code_shape)
            slerp_start_codes.extend(list(start_codes[1:])) # drop first frame to prevent repeating frames
            if loop and i == len(prompts) - 1:
                c_vectors = get_slerp_vectors(c.flatten(), slerp_c_vectors[0].flatten(), device, frames=frames)
                c_vectors = c_vectors.reshape(-1, *original_c_shape)
                slerp_c_vectors.extend(list(c_vectors[1:-1])) # drop first and last frame to prevent repeating frames
                start_codes = get_slerp_vectors(start_code.flatten(), slerp_start_codes[0].flatten(), device, frames=frames)
                start_codes = start_codes.reshape(-1, *original_start_code_shape)
                slerp_start_codes.extend(list(start_codes[1:-1])) # drop first and last frame to prevent repeating frames
        previous_c = c
        previous_start_code = start_code

    slerp_c_vectors = unflatten(slerp_c_vectors, batch_size)
    slerp_start_codes = unflatten(slerp_start_codes, batch_size)
    n_iter = len(slerp_c_vectors)

    if sampler_name == 'PLMS':
        sampler = PLMSSampler(model)
    elif sampler_name == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(model,'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(model,'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(model,'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(model,'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(model,'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(model,'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    def init():
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        callback = make_callback(sampler_name, dynamic_threshold=dynamic_threshold, static_threshold=static_threshold)
        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale, unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x, img_callback=callback)
        return samples_ddim

    try:
        output_images, seed, info, stats = process_images(
            outpath=outpath,
            func_init=init,
            func_sample=sample,
            prompt=slerp_c_vectors,
            seed=slerp_start_codes,
            sampler_name=sampler_name,
            skip_save=skip_save,
            skip_grid=skip_save_mp4,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=ddim_steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            prompt_matrix=None,
            use_GFPGAN=use_GFPGAN,
            use_RealESRGAN=use_RealESRGAN,
            realesrgan_model_name=realesrgan_model_name,
            fp=fp,
            ddim_eta=ddim_eta,
            sort_samples=sort_samples,
            write_info_files=write_info_files,
            jpg_sample=jpg_sample,
            do_interpolation=True
        )

        del sampler
        return output_images #, seed, info, stats
    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return [], seed, 'err', stats
    finally:
        if err:
            crash(err, '!!Runtime error (txt_interp)!!')


def txt2img(prompt: str, ddim_steps: int, sampler_name: str, toggles: List[int], realesrgan_model_name: str,
            ddim_eta: float, n_iter: int, batch_size: int, cfg_scale: float, dynamic_threshold: float, 
            static_threshold: float, seed: Union[int, str, None], height: int, width: int, fp):
    if opt.outdir_txt2img != None:
        outpath = opt.outdir_txt2img
    elif opt.outdir != None:
        outpath = opt.outdir
    else:
        outpath = "outputs/txt2img-samples"
    err = False
    seed = seed_to_int(seed)

    prompt_matrix = 0 in toggles
    skip_save = 1 not in toggles
    skip_grid = 2 not in toggles
    sort_samples = 3 in toggles
    write_info_files = 4 in toggles
    jpg_sample = 5 in toggles
    use_GFPGAN = 6 in toggles
    use_RealESRGAN = 6 in toggles if GFPGAN is None else 7 in toggles # possible index shift

    if sampler_name == 'PLMS':
        sampler = PLMSSampler(model)
    elif sampler_name == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(model,'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(model,'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(model,'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(model,'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(model,'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(model,'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    def init():
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        callback = make_callback(sampler_name, dynamic_threshold=dynamic_threshold, static_threshold=static_threshold)
        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale, unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x, img_callback=callback)
        return samples_ddim

    try:
        output_images, seed, info, stats = process_images(
            outpath=outpath,
            func_init=init,
            func_sample=sample,
            prompt=prompt,
            seed=seed,
            sampler_name=sampler_name,
            skip_save=skip_save,
            skip_grid=skip_grid,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=ddim_steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            prompt_matrix=prompt_matrix,
            use_GFPGAN=use_GFPGAN,
            use_RealESRGAN=use_RealESRGAN,
            realesrgan_model_name=realesrgan_model_name,
            fp=fp,
            ddim_eta=ddim_eta,
            sort_samples=sort_samples,
            write_info_files=write_info_files,
            jpg_sample=jpg_sample,
        )

        del sampler

        return output_images, seed, info, stats
    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return [], seed, 'err', stats
    finally:
        if err:
            crash(err, '!!Runtime error (txt2img)!!')


def disco_anim(prompt: str, init_info, project_name: str, ddim_steps: int, sampler_name: str, toggles: List[int], ddim_eta: float, cfg_scale: float, color_match_mode: str,
               dynamic_threshold: float, static_threshold: float, degrees_per_second: int, frames_per_second: int, prev_frame_denoising: float, 
               noise_between_frames: float, mix_factor: float, start_frame: int, max_frames: int, animation_mode: str, interp_spline: str, angle: str, zoom: str,
               translation_x: str, translation_y: str, translation_z: str, rotation_3d_x: str, rotation_3d_y: str, rotation_3d_z: str, midas_weight: float,
               near_plane: int, far_plane: int, fov: int, padding_mode: str, sampling_mode: str, turbo_steps: int, turbo_preroll: int, vr_eye_angle: float,
               vr_ipd: float, seed: Union[int, str, None], height: int, width: int, resize_mode: int, fp):
    if opt.outdir_disco_anim != None:
        outpath = opt.outdir_disco_anim
    if opt.outdir != None:
        outpath = opt.outdir
    else:
        outpath = "outputs/disco_anim"

    start_frame = int(start_frame)
    max_frames = int(max_frames)

    outpath = outpath + '/' + project_name
    os.makedirs(outpath, exist_ok=True)

    # videoFramesFolder = outpath + '/VideoFrames'
    # os.makedirs(videoFramesFolder, exists_ok=True)
    err = False
    seed = seed_to_int(seed)

    prompts = prompt.split('\n')

    frames_per_degree = frames_per_second / degrees_per_second

    interpolate = 0 in toggles
    loop = 1 in toggles
    turbo_mode = 2 in toggles
    vr_mode = 3 in toggles
    resume_run = 4 in toggles
    color_match = 5 in toggles
    consistent_seed = 6 in toggles
    init_latent_mixing = 7 in toggles
    # video_init_flow_warp = 8 in toggles
    # video_init_force_flow = 9 in toggles

    if sampler_name == 'PLMS':
        sampler = DDIMSampler(model)
        print('PLMS not compatible, falling back to DDIM')
    elif sampler_name == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(model,'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(model,'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(model,'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(model,'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(model,'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(model,'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    assert 0. <= prev_frame_denoising <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(prev_frame_denoising * ddim_steps)

    try:
        mix_factor_series = get_inbetweens(parse_key_frames(mix_factor), t_enc, interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `mix_factor` correctly for key frames.\n"
            "Attempting to interpret `mix_factor` as "
            f'"0: ({mix_factor})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        mix_factor = f"0: ({mix_factor})"
        mix_factor_series = get_inbetweens(parse_key_frames(mix_factor), t_enc, interp_spline)
    try:
        angle_series = get_inbetweens(parse_key_frames(mix_factor), max_frames, interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `angle` correctly for key frames.\n"
            "Attempting to interpret `angle` as "
            f'"0: ({angle})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        angle = f"0: ({angle})"
        angle_series = get_inbetweens(parse_key_frames(angle), max_frames, interp_spline)

    try:
        zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames, interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `zoom` correctly for key frames.\n"
            "Attempting to interpret `zoom` as "
            f'"0: ({zoom})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        zoom = f"0: ({zoom})"
        zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames, interp_spline)

    try:
        translation_x_series = get_inbetweens(parse_key_frames(translation_x), max_frames, interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_x` correctly for key frames.\n"
            "Attempting to interpret `translation_x` as "
            f'"0: ({translation_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_x = f"0: ({translation_x})"
        translation_x_series = get_inbetweens(parse_key_frames(translation_x), max_frames, interp_spline)

    try:
        translation_y_series = get_inbetweens(parse_key_frames(translation_y), max_frames, interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_y` correctly for key frames.\n"
            "Attempting to interpret `translation_y` as "
            f'"0: ({translation_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_y = f"0: ({translation_y})"
        translation_y_series = get_inbetweens(parse_key_frames(translation_y), max_frames, interp_spline)

    try:
        translation_z_series = get_inbetweens(parse_key_frames(translation_z), max_frames, interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_z` correctly for key frames.\n"
            "Attempting to interpret `translation_z` as "
            f'"0: ({translation_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_z = f"0: ({translation_z})"
        translation_z_series = get_inbetweens(parse_key_frames(translation_z), max_frames, interp_spline)

    try:
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x), max_frames, interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_x` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_x` as "
            f'"0: ({rotation_3d_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_x = f"0: ({rotation_3d_x})"
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x), max_frames, interp_spline)

    try:
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y), max_frames, interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_y` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_y` as "
            f'"0: ({rotation_3d_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_y = f"0: ({rotation_3d_y})"
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y), max_frames, interp_spline)

    try:
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z), max_frames, interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_z` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_z` as "
            f'"0: ({rotation_3d_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_z = f"0: ({rotation_3d_z})"
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z), max_frames, interp_spline)

    def get_conditioning_vector(prompt):
        subprompts, weights = split_weighted_subprompts(prompt)

        # sub-prompt weighting used if more than 1
        if len(subprompts) > 1:
            c = (model if not opt.optimized else modelCS).get_learned_conditioning(subprompts[0])
            original_c_shape = c.shape
            c = c.flatten()
            for i in range(1,len(subprompts)): 
                weight = weights[i-1]
                # slerp between subprompts by weight between 0-1
                c = slerp(weight, c, (model if not opt.optimized else modelCS).get_learned_conditioning(subprompts[i]).flatten())
            c = c.reshape(*original_c_shape)
        else: # just behave like usual
            c = (model if not opt.optimized else modelCS).get_learned_conditioning(prompt)
        return c

    if interpolate and len(prompts) > 1:
        previous_c = None
        slerp_c_vectors = []
        for i, c in enumerate(map(lambda x: get_conditioning_vector(x), prompts)):
            if i == 0:
                slerp_c_vectors.append(c)
            else:
                start_norm = previous_c.flatten()/torch.norm(previous_c.flatten())
                end_norm = c.flatten()/torch.norm(c.flatten())
                omega = torch.acos((start_norm*end_norm).sum())
                frames = round(omega.item() * frames_per_degree * 57.2957795)

                original_c_shape = c.shape
                c_vectors = get_slerp_vectors(previous_c.flatten(), c.flatten(), device, frames=frames)
                c_vectors = c_vectors.reshape(-1, *original_c_shape)
                slerp_c_vectors.extend(list(c_vectors[1:])) # drop first frame to prevent repeating frames
                if loop and i == len(prompts) - 1:
                    c_vectors = get_slerp_vectors(c.flatten(), slerp_c_vectors[0].flatten(), device, frames=frames)
                    c_vectors = c_vectors.reshape(-1, *original_c_shape)
                    slerp_c_vectors.extend(list(c_vectors[1:-1])) # drop first and last frame to prevent repeating frames
            previous_c = c
        prompts = slerp_c_vectors
    else:
        interpolate = False

    # elif animation_mode == "Video Input":
    #     print(f"Exporting Video Frames (1 every {extract_nth_frame})...")
    #     try:
    #         for f in pathlib.Path(f'{videoFramesFolder}').glob('*.jpg'):
    #             f.unlink()
    #     except:
    #         print('')
    #     vf = f'select=not(mod(n\,{extract_nth_frame}))'
    #     if os.path.exists(video_init_path):
    #         subprocess.run(['ffmpeg', '-i', f'{video_init_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{videoFramesFolder}/%04d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    #     else: 
    #         print(f'\nWARNING!\n\nVideo not found: {video_init_path}.\nPlease check your video path.\n')

    #     force_flow_generation = video_init_force_flow
    #     in_path = videoFramesFolder
    #     flo_folder = f'{in_path}/out_flo_fwd'
    #     flo_fwd_folder = in_path+'/out_flo_fwd'
    #     os.makedirs(flo_fwd_folder, exist_ok=True)
    #     os.makedirs(temp_flo, exist_ok=True)

    #     if not video_init_flow_warp:
    #         print('video_init_flow_warp not set, skipping')

    #     if (animation_mode == 'Video Input') and (video_init_flow_warp):
    #         flows = glob(flo_folder+'/*.*')
    #         if (len(flows)>0) and not force_flow_generation:
    #             print(f'Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {flo_folder}.\nIf you wish to generate new flow files, check force_flow_generation and run this cell again.')
        
    #         if (len(flows)==0) or force_flow_generation:
    #             frames = sorted(glob(in_path+'/*.*'));
    #             if len(frames)<2: 
    #                 print(f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.')
    #             if len(frames)>=2:
            
    #                 raft_model = torch.nn.DataParallel(RAFT(args2))
    #                 raft_model.load_state_dict(torch.load(f'{root_path}/RAFT/models/raft-things.pth'))
    #                 raft_model = raft_model.module.cuda().eval()
            
    #                 for f in pathlib.Path(f'{flo_fwd_folder}').glob('*.*'):
    #                     f.unlink()
            
                    
            
    #                 framecount = 0
    #                 for frame1, frame2 in tqdm(zip(frames[:-1], frames[1:]), total=len(frames)-1):
            
    #                     out_flow21_fn = f"{flo_fwd_folder}/{frame1.split('/')[-1]}"
                
    #                     frame1 = load_img(frame1, width_height)
    #                     frame2 = load_img(frame2, width_height)
                
    #                     flow21 = get_flow(frame2, frame1, raft_model)
    #                     np.save(out_flow21_fn, flow21)

    #                 del raft_model 
    #                 gc.collect()

    def init(init_image):
        if init_image is not None:
            if isinstance(init_image, str):
                image = Image.open(init_image)
            else:
                image = init_image
            image = image.convert("RGB")
            image = resize_image(resize_mode, image, width, height)
            image = np.array(image).astype(np.float32) / 255.0
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)

            if opt.optimized:
                modelFS.to(device)

            init_image = 2. * image - 1.
            init_image = init_image.to(device)
            # init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            init_latent = (model if not opt.optimized else modelFS).get_first_stage_encoding((model if not opt.optimized else modelFS).encode_first_stage(init_image))  # move to latent space
            
            if opt.optimized:
                mem = torch.cuda.memory_allocated()/1e6
                modelFS.to("cpu")
                while(torch.cuda.memory_allocated()/1e6 >= mem):
                    time.sleep(1)
            return init_latent
        else:
            return None

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        if init_data is not None:
            if sampler_name not in ['DDIM', 'PLMS']:
                x0, = init_data

                sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
                noise = x * sigmas[ddim_steps - t_enc - 1]

                callback = make_callback(sampler_name, dynamic_threshold=dynamic_threshold, static_threshold=static_threshold, mix_with_x0=init_latent_mixing, mix_factor=mix_factor_series, x0=x0, noise=x)

                xi = x0 + noise
                sigma_sched = sigmas[ddim_steps - t_enc - 1:]
                model_wrap_cfg = CFGDenoiser(sampler.model_wrap)
                samples_ddim = K.sampling.__dict__[f'sample_{sampler.schedule}'](model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': cfg_scale}, disable=False, callback=callback)
            else:
                callback = make_callback(sampler_name, dynamic_threshold=dynamic_threshold, static_threshold=static_threshold)
                x0, = init_data
                sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
                z_enc = sampler.stochastic_encode(x0, torch.tensor([t_enc]).to(device))
                samples_ddim = sampler.decode(z_enc, conditioning, t_enc,
                                                unconditional_guidance_scale=cfg_scale,
                                                unconditional_conditioning=unconditional_conditioning,
                                                img_callback=callback)
        else:
            callback = make_callback(sampler_name, dynamic_threshold=dynamic_threshold, static_threshold=static_threshold)
            samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale, unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x, img_callback=callback)
        return samples_ddim

    with open(f"{outpath}/{project_name}_settings.yaml", "w", encoding="utf8") as f:
            yaml.dump({
                'prompts': prompt,
                'seed': seed,
                'sampler_name': sampler_name,
                'animation_mode': animation_mode,
                'start_frame': start_frame,
                'max_frames': max_frames,
                'unconditional_guidance_scale': cfg_scale,
                'width': width,
                'height': height,
                'resume_run': resume_run,
                'mix_factor': mix_factor,
                'angle': angle,
                'zoom': zoom,
                'translation_x': translation_x,
                'translation_y': translation_y,
                'translation_z': translation_z,
                'rotation_3d_x': rotation_3d_x,
                'rotation_3d_y': rotation_3d_y,
                'rotation_3d_z': rotation_3d_z,
                'color_match': color_match,
                'noise_between_frames': noise_between_frames,
                'turbo_mode': turbo_mode,
                'turbo_preroll': turbo_preroll,
                'turbo_steps': turbo_steps,
                'vr_mode': vr_mode,
                'consistent_seed': consistent_seed,
                'interpolate': interpolate,
                'vr_eye_angle': vr_eye_angle,
                'vr_ipd': vr_ipd,
                'midas_weight': midas_weight,
                'near_plane': near_plane,
                'far_plane': far_plane,
                'fov': fov,
                'padding_mode': padding_mode,
                'sampling_mode': sampling_mode,
                'color_match_mode': color_match_mode
            }, f, allow_unicode=True)

    try:
        output_images = process_disco_anim(
            outpath=outpath,
            func_init=init,
            func_sample=sample,
            init_image=init_info,
            prompts=prompts,
            seed=seed,
            sampler_name=sampler_name,
            animation_mode=animation_mode,
            start_frame=start_frame,
            max_frames=max_frames,
            unconditional_guidance_scale=cfg_scale,
            width=width,
            height=height,
            resume_run=resume_run,
            angle_series=angle_series,
            zoom_series=zoom_series,
            translation_x_series=translation_x_series,
            translation_y_series=translation_y_series,
            translation_z_series=translation_z_series,
            rotation_3d_x_series=rotation_3d_x_series,
            rotation_3d_y_series=rotation_3d_y_series,
            rotation_3d_z_series=rotation_3d_z_series,
            color_match=color_match,
            noise_between_frames=noise_between_frames,
            turbo_mode=turbo_mode,
            turbo_preroll=turbo_preroll,
            turbo_steps=turbo_steps,
            vr_mode=vr_mode,
            video_init_frames_scale=prev_frame_denoising,
            video_init_flow_warp=None, #video_init_flow_warp,
            videoFramesFolder=None, #videoFramesFolder,
            flo_folder=None, #flo_folder,
            video_init_flow_blend=None, #video_init_flow_blend,
            consistent_seed=consistent_seed,
            interpolate=interpolate,
            vr_eye_angle=vr_eye_angle,
            vr_ipd=vr_ipd,
            midas_weight=midas_weight,
            near_plane=near_plane,
            far_plane=far_plane,
            fov=fov,
            padding_mode=padding_mode,
            sampling_mode=sampling_mode,
            color_match_mode=color_match_mode
        )

        del sampler

        return output_images #, seed, info, stats
    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return [], seed, 'err', stats
    finally:
        if err:
            crash(err, '!!Runtime error (disco_anim)!!')


class Flagging(gr.FlaggingCallback):

    def setup(self, components, flagging_dir: str):
        pass

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        import csv

        os.makedirs("log/images", exist_ok=True)

        # those must match the "txt2img" function !! + images, seed, comment, stats !! NOTE: changes to UI output must be reflected here too
        prompt, ddim_steps, sampler_name, toggles, ddim_eta, n_iter, batch_size, cfg_scale, seed, height, width, fp, images, seed, comment, stats = flag_data

        filenames = []

        with open("log/log.csv", "a", encoding="utf8", newline='') as file:
            import time
            import base64

            at_start = file.tell() == 0
            writer = csv.writer(file)
            if at_start:
                writer.writerow(["sep=,"])
                writer.writerow(["prompt", "seed", "width", "height", "sampler", "toggles", "n_iter", "n_samples", "cfg_scale", "steps", "filename"])

            filename_base = str(int(time.time() * 1000))
            for i, filedata in enumerate(images):
                filename = "log/images/"+filename_base + ("" if len(images) == 1 else "-"+str(i+1)) + ".png"

                if filedata.startswith("data:image/png;base64,"):
                    filedata = filedata[len("data:image/png;base64,"):]

                with open(filename, "wb") as imgfile:
                    imgfile.write(base64.decodebytes(filedata.encode('utf-8')))

                filenames.append(filename)

            writer.writerow([prompt, seed, width, height, sampler_name, toggles, n_iter, batch_size, cfg_scale, ddim_steps, filenames[0]])

        print("Logged:", filenames[0])


def img2img(prompt: str, image_editor_mode: str, init_info, mask_mode: str, mask_blur_strength: int, ddim_steps: int, sampler_name: str,
            toggles: List[int], realesrgan_model_name: str, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float,
            dynamic_threshold: float, static_threshold: float, seed: int, height: int, width: int, resize_mode: int, fp):
    outpath = opt.outdir_img2img or opt.outdir or "outputs/img2img-samples"
    err = False
    seed = seed_to_int(seed)

    width, height = map(lambda x: x - x % 32, (width, height))  # resize to integer multiple of 32

    prompt_matrix = 0 in toggles
    loopback = 1 in toggles
    random_seed_loopback = 2 in toggles
    skip_save = 3 not in toggles
    skip_grid = 4 not in toggles
    sort_samples = 5 in toggles
    write_info_files = 6 in toggles
    jpg_sample = 7 in toggles
    use_GFPGAN = 8 in toggles
    use_RealESRGAN = 8 in toggles if GFPGAN is None else 9 in toggles # possible index shift

    inpainting = False

    if sampler_name == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(model,'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(model,'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(model,'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(model,'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(model,'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(model,'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    if image_editor_mode == 'Mask':
        inpainting = True
        init_img = init_info["image"]
        init_img = init_img.convert("RGB")
        init_img = resize_image(resize_mode, init_img, width, height)
        init_mask = init_info["mask"]
        init_mask = init_mask.convert("L")
        init_mask = init_mask.filter(ImageFilter.GaussianBlur(mask_blur_strength))
        init_mask = resize_image(resize_mode, init_mask, width//opt_f, height//opt_f)
        keep_mask = mask_mode == 0
        init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
        init_mask = np.array(init_mask).astype(np.float32) / 255.0
        init_mask = init_mask[None,None]
        init_mask = torch.from_numpy(init_mask).to(device)
    else:
        init_img = init_info
        init_mask = None
        keep_mask = False

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(denoising_strength * ddim_steps)

    def init():
        image = init_img.convert("RGB")
        image = resize_image(resize_mode, image, width, height)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        if opt.optimized:
            modelFS.to(device)

        init_image = 2. * image - 1.
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = (model if not opt.optimized else modelFS).get_first_stage_encoding((model if not opt.optimized else modelFS).encode_first_stage(init_image))  # move to latent space
        
        if opt.optimized:
            mem = torch.cuda.memory_allocated()/1e6
            modelFS.to("cpu")
            while(torch.cuda.memory_allocated()/1e6 >= mem):
                time.sleep(1)

        return init_latent,

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        if sampler_name != 'DDIM':
            x0, = init_data

            sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
            noise = x * sigmas[ddim_steps - t_enc - 1]

            callback = make_callback(sampler_name, dynamic_threshold=dynamic_threshold, static_threshold=static_threshold, inpainting=inpainting, x0=x0, noise=x, mask=init_mask)

            xi = x0 + noise
            sigma_sched = sigmas[ddim_steps - t_enc - 1:]
            model_wrap_cfg = CFGDenoiser(sampler.model_wrap)
            samples_ddim = K.sampling.__dict__[f'sample_{sampler.schedule}'](model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': cfg_scale}, disable=False, callback=callback)
        else:
            x0, = init_data
            sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
            z_enc = sampler.stochastic_encode(x0, torch.tensor([t_enc]*batch_size).to(device))
                                # decode it
            callback = make_callback(sampler_name, dynamic_threshold=dynamic_threshold, static_threshold=static_threshold)
            samples_ddim = sampler.decode(z_enc, conditioning, t_enc,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=unconditional_conditioning,
                                            img_callback=callback, x0=x0, mask=init_mask)
        return samples_ddim

    try:
        if loopback:
            output_images, info = None, None
            history = []
            initial_seed = None

            for i in range(n_iter):
                output_images, seed, info, stats = process_images(
                    outpath=outpath,
                    func_init=init,
                    func_sample=sample,
                    prompt=prompt,
                    seed=seed,
                    sampler_name=sampler_name,
                    skip_save=skip_save,
                    skip_grid=skip_grid,
                    batch_size=1,
                    n_iter=1,
                    steps=ddim_steps,
                    cfg_scale=cfg_scale,
                    width=width,
                    height=height,
                    prompt_matrix=prompt_matrix,
                    use_GFPGAN=use_GFPGAN,
                    use_RealESRGAN=False, # Forcefully disable upscaling when using loopback
                    realesrgan_model_name=realesrgan_model_name,
                    fp=fp,
                    do_not_save_grid=True,
                    init_img=init_img,
                    init_mask=init_mask,
                    keep_mask=keep_mask,
                    mask_blur_strength=mask_blur_strength,
                    denoising_strength=denoising_strength,
                    resize_mode=resize_mode,
                    uses_loopback=loopback,
                    uses_random_seed_loopback=random_seed_loopback,
                    sort_samples=sort_samples,
                    write_info_files=write_info_files,
                    jpg_sample=jpg_sample
                )

                if initial_seed is None:
                    initial_seed = seed

                init_img = output_images[0]
                if not random_seed_loopback:
                    seed = seed + 1
                else:
                    seed = seed_to_int(None)
                denoising_strength = max(denoising_strength * 0.95, 0.1)
                history.append(init_img)

            if not skip_grid:
                grid_count = get_next_sequence_number(outpath, 'grid-')
                grid = image_grid(history, batch_size, force_n_rows=1)
                grid_file = f"grid-{grid_count:05}-{seed}_{prompt.replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.{grid_ext}"
                grid.save(os.path.join(outpath, grid_file), grid_format, quality=grid_quality, lossless=grid_lossless, optimize=True)


            output_images = history
            seed = initial_seed

        else:
            output_images, seed, info, stats = process_images(
                outpath=outpath,
                func_init=init,
                func_sample=sample,
                prompt=prompt,
                seed=seed,
                sampler_name=sampler_name,
                skip_save=skip_save,
                skip_grid=skip_grid,
                batch_size=batch_size,
                n_iter=n_iter,
                steps=ddim_steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                prompt_matrix=prompt_matrix,
                use_GFPGAN=use_GFPGAN,
                use_RealESRGAN=use_RealESRGAN,
                realesrgan_model_name=realesrgan_model_name,
                fp=fp,
                init_img=init_img,
                init_mask=init_mask,
                keep_mask=keep_mask,
                mask_blur_strength=mask_blur_strength,
                denoising_strength=denoising_strength,
                resize_mode=resize_mode,
                uses_loopback=loopback,
                sort_samples=sort_samples,
                write_info_files=write_info_files,
                jpg_sample=jpg_sample,
            )

        del sampler

        return output_images, seed, info, stats
    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return [], seed, 'err', stats
    finally:
        if err:
            crash(err, '!!Runtime error (img2img)!!')

# grabs all text up to the first occurrence of ':' as sub-prompt
# takes the value following ':' as weight
# if ':' has no value defined, defaults to 1.0
# repeats until no text remaining
# TODO this could probably be done with less code
def split_weighted_subprompts(text):
    print(text)
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":") # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx+1:]
            # find value for weight, assume it is followed by a space or comma
            idx = len(text) # default is read to end of text
            if " " in text:
                idx = min(idx,text.index(" ")) # want the closer idx
            if "," in text:
                idx = min(idx,text.index(",")) # want the closer idx
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except: # couldn't treat as float
                    print(f"Warning: '{text[:idx]}' is not a value, are you missing a space or comma after a value?")
                    weight = 0.5
            else: # no value found
                weight = 0.5
            # remove from main text
            remaining -= idx
            text = text[idx+1:]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(min(max(weight, 0), 1))
        else: # no : found
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(0.5)
            remaining = 0
    return prompts, weights

def run_GFPGAN(image, strength):
    image = image.convert("RGB")

    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(np.array(image, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)
    res = Image.fromarray(restored_img)

    if strength < 1.0:
        res = Image.blend(image, res, strength)

    return res

def run_RealESRGAN(image, model_name: str):
    if RealESRGAN.model.name != model_name:
            try_loading_RealESRGAN(model_name)

    image = image.convert("RGB")

    output, img_mode = RealESRGAN.enhance(np.array(image, dtype=np.uint8))
    res = Image.fromarray(output)

    return res


if opt.defaults is not None and os.path.isfile(opt.defaults):
    try:
        with open(opt.defaults, "r", encoding="utf8") as f:
            user_defaults = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        print(f"Error loading defaults file {opt.defaults}:", e, file=sys.stderr)
        print("Falling back to program defaults.", file=sys.stderr)
        user_defaults = {}
else:
    user_defaults = {}

# make sure these indicies line up at the top of txt2img()
txt2img_toggles = [
    'Create prompt matrix (separate multiple prompts using |, and get all combinations of them)',
    'Save individual images',
    'Save grid',
    'Sort samples by prompt',
    'Write sample info files',
    'jpg samples',
]
if GFPGAN is not None:
    txt2img_toggles.append('Fix faces using GFPGAN')
if RealESRGAN is not None:
    txt2img_toggles.append('Upscale images using RealESRGAN')

txt2img_defaults = {
    'prompt': '',
    'ddim_steps': 50,
    'toggles': [1, 2, 3],
    'sampler_name': 'k_euler_a',
    'ddim_eta': 0.0,
    'n_iter': 1,
    'batch_size': 1,
    'cfg_scale': 7.5,
    'dynamic_threshold': 0,
    'static_threshold': 0,
    'seed': '',
    'height': 512,
    'width': 512,
    'fp': None,
    'submit_on_enter': 'Yes'
}

if 'txt2img' in user_defaults:
    txt2img_defaults.update(user_defaults['txt2img'])

txt2img_toggle_defaults = [txt2img_toggles[i] for i in txt2img_defaults['toggles']]

# make sure these indicies line up at the top of txt_interp()
txt_interp_toggles = [
    'Loop Interplation',
    'Skip Save',
    'Skip Save mp4',
    'Sort Samples',
    'Write Info Files',
    'jpg Samples',
    'use GFPGAN',
    'use RealESRGAN'
]
if GFPGAN is not None:
    txt_interp_toggles.append('Fix faces using GFPGAN')
if RealESRGAN is not None:
    txt_interp_toggles.append('Upscale images using RealESRGAN')

txt_interp_defaults = {
    'prompt': '',
    'ddim_steps': 50,
    'toggles': [0],
    'sampler_name': 'PLMS',
    'ddim_eta': 0.0,
    'n_iter': 1,
    'batch_size': 1,
    'cfg_scale': 7.5,
    'dynamic_threshold': 0,
    'static_threshold': 0,
    'degrees_per_second': 30,
    'frames_per_second': 30,
    'project_name': 'interp',
    'seeds': 'None\nNone',
    'height': 512,
    'width': 512,
    'fp': None,
}

if 'txt_interp' in user_defaults:
    txt_interp_defaults.update(user_defaults['txt_interp'])

txt_interp_toggle_defaults = [txt_interp_toggles[i] for i in txt_interp_defaults['toggles']]

sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# make sure these indicies line up at the top of img2img()
img2img_toggles = [
    'Create prompt matrix (separate multiple prompts using |, and get all combinations of them)',
    'Loopback (use images from previous batch when creating next batch)',
    'Random loopback seed',
    'Save individual images',
    'Save grid',
    'Sort samples by prompt',
    'Write sample info files',
    'jpg samples',
]
if GFPGAN is not None:
    img2img_toggles.append('Fix faces using GFPGAN')
if RealESRGAN is not None:
    img2img_toggles.append('Upscale images using RealESRGAN')

img2img_mask_modes = [
    "Keep masked area",
    "Regenerate only masked area",
]

img2img_resize_modes = [
    "Just resize",
    "Crop and resize",
    "Resize and fill",
]

img2img_defaults = {
    'prompt': '',
    'ddim_steps': 50,
    'toggles': [1, 4, 5],
    'sampler_name': 'k_euler_a',
    'ddim_eta': 0.0,
    'n_iter': 1,
    'batch_size': 1,
    'cfg_scale': 7.5,
    'denoising_strength': 0.75,
    'dynamic_threshold': 0,
    'static_threshold': 0,
    'mask_mode': 0,
    'resize_mode': 0,
    'seed': '',
    'height': 512,
    'width': 512,
    'fp': None,
}

if 'img2img' in user_defaults:
    img2img_defaults.update(user_defaults['img2img'])

img2img_toggle_defaults = [img2img_toggles[i] for i in img2img_defaults['toggles']]
img2img_image_mode = 'sketch'

# make sure these indicies line up at the top of disco_anim()
disco_anim_toggles = [
    'Interpolate Between Prompts',
    'Loop Interpolation',
    'Turbo Mode',
    'VR Mode',
    'Resume Run',
    'Color Match',
    'Consistent Seed',
    'Init Latent Mixing',
    # 'Video Init Flow Warp',
    # 'Video Init Force Flow',
]

disco_anim_resize_modes = [
    "Just resize",
    "Crop and resize",
    "Resize and fill",
]

disco_anim_defaults = {
    'prompt': '',
    'ddim_steps': 100,
    'toggles': [0, 1, 5, 7],
    'sampler_name': 'k_euler_a',
    'ddim_eta': 0.0,
    'n_iter': 1,
    'batch_size': 1,
    'cfg_scale': 7.5,
    'denoising_strength': 0.75,
    'dynamic_threshold': 0,
    'static_threshold': 0,
    'prev_frame_denoising_strength': 0.3,
    'noise_between_frames': 0,
    'color_match_mode': 'LAB',
    'degrees_per_second': 12,
    'frames_per_second': 12,
    'project_name': 'disco',
    'max_frames': 10000,
    'animation_mode': '3D',
    'interp_spline': 'Linear',
    'resize_mode': 0,
    'mix_factor': "0: (0.15) 10: (1.0)",
    'angle': "0:(0)",
    'zoom': "0: (1)",
    'translation_x': "0: (0)",
    'translation_y': "0: (0)",
    'translation_z': "0: (5.0)",
    'rotation_3d_x': "0: (0.2)",
    'rotation_3d_y': "0: (0)",
    'rotation_3d_z': "0: (0)",
    'midas_weight': 0.3,
    'near_plane': 200,
    'far_plane': 10000,
    'fov': 40,
    'padding_mode': 'border',
    'sampling_mode': 'bicubic',
    'turbo_steps': 3,
    'turbo_preroll': 10,
    'vr_eye_angle': 0.5,
    'vr_ipd': 5.0,
    'height': 512,
    'width': 512,
    'fp': None
}

if 'disco_anim' in user_defaults:
    disco_anim_defaults.update(user_defaults['disco_anim'])

disco_anim_toggle_defaults = [disco_anim_toggles[i] for i in disco_anim_defaults['toggles']]

def change_image_editor_mode(choice, cropped_image, resize_mode, width, height):
    if choice == "Mask":
        return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)]
    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]

def update_image_mask(cropped_image, resize_mode, width, height):
    resized_cropped_image = resize_image(resize_mode, cropped_image, width, height) if cropped_image else None
    return gr.update(value=resized_cropped_image)

def copy_img_to_input(selected=1, imgs = []):
    try:
        idx = int(0 if selected - 1 < 0 else selected - 1)
        image_data = re.sub('^data:image/.+;base64,', '', imgs[idx])
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        update = gr.update(selected='img2img_tab')
        return [processed_image, processed_image, update]
    except IndexError:
        return [None, None]

# def stop_anim():
#     stop_on_next_loop = True
#     return

help_text = """
    ## Mask/Crop
    * The masking/cropping is very temperamental.
    * It may take some time for the image to show when switching from Crop to Mask.
    * If the image doesn't appear after switching to Mask, switch back to Crop and then back again to Mask
    * If the mask appears distorted (the brush is weirdly shaped instead of round), switch back to Crop and then back again to Mask.

    ## Advanced Editor
    * For now the button needs to be clicked twice the first time.
    * Once you have edited your image, you _need_ to click the save button for the next step to work.
    * Clear the image from the crop editor (click the x)
    * Click "Get Image from Advanced Editor" to get the image you saved. If it doesn't work, try opening the editor and saving again.

    If it keeps not working, try switching modes again, switch tabs, clear the image or reload.
"""

def show_help():
    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=help_text)]

def hide_help():
    return [gr.update(visible=True), gr.update(visible=False), gr.update(value="")]


css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

styling = """
[data-testid="image"] {min-height: 512px !important}
* #body>.col:nth-child(2){width:250%;max-width:89vw}
#generate{width: 100%; }
#prompt_row input{
 font-size:20px
}
"""

css = styling if opt.no_progressbar_hiding else styling + css_hide_progressbar

with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion WebUI") as demo:
    with gr.Tabs(elem_id='tabss') as tabs:
        with gr.TabItem("Stable Diffusion Text-to-Image Unified", id='txt2img_tab'):
            with gr.Row(elem_id="prompt_row"):
                txt2img_prompt = gr.Textbox(label="Prompt", 
                elem_id='prompt_input',
                placeholder="A corgi wearing a top hat as an oil painting.", 
                lines=1,
                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25, 
                value=txt2img_defaults['prompt'], 
                show_label=False).style()
                
            with gr.Row(elem_id='body').style(equal_height=False):
                with gr.Column():
                    txt2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=txt2img_defaults["height"])
                    txt2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=txt2img_defaults["width"])
                    txt2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=txt2img_defaults['cfg_scale'])
                    txt2img_dynamic_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Dynamic Threshold', value=txt2img_defaults['dynamic_threshold'])
                    txt2img_static_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Static Threshold', value=txt2img_defaults['static_threshold'])
                    txt2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, max_lines=1, value=txt2img_defaults["seed"])                    
                    txt2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1, label='Batch count (how many batches of images to generate)', value=txt2img_defaults['n_iter'])
                    txt2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=txt2img_defaults['batch_size'])
                with gr.Column():
                    output_txt2img_gallery = gr.Gallery(label="Images", elem_id="gallery_output").style(grid=[4,4])
                    with gr.Row():
                        with gr.Group():
                            output_txt2img_seed = gr.Number(label='Seed', interactive=False)
                            output_txt2img_copy_seed = gr.Button("Copy", full_width=True).click(inputs=output_txt2img_seed, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                        with gr.Group():
                            output_txt2img_select_image = gr.Number(label='Image # and click Copy to copy to img2img', value=1, precision=None)
                            output_txt2img_copy_to_input_btn = gr.Button("Push to img2img", full_width=True)
                    with gr.Group():
                        output_txt2img_params = gr.Textbox(label="Copy-paste generation parameters", interactive=False)
                        output_txt2img_copy_params = gr.Button("Copy", full_width=True).click(inputs=output_txt2img_params, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                    output_txt2img_stats = gr.HTML(label='Stats')
                with gr.Column():
                    txt2img_btn = gr.Button("Generate", full_width=True, elem_id="generate", variant="primary")
                    txt2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=txt2img_defaults['ddim_steps'])
                    txt2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=txt2img_defaults['sampler_name'])
                    with gr.Tabs():
                        with gr.TabItem('Simple'):
                            txt2img_submit_on_enter = gr.Radio(['Yes', 'No'], label="Submit on enter? (no means multiline)", value=txt2img_defaults['submit_on_enter'], interactive=True)
                            txt2img_submit_on_enter.change(lambda x: gr.update(max_lines=1 if x == 'Single' else 25) , txt2img_submit_on_enter, txt2img_prompt)
                        with gr.TabItem('Advanced'):
                            txt2img_toggles = gr.CheckboxGroup(label='', choices=txt2img_toggles, value=txt2img_toggle_defaults, type="index")
                            txt2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus', visible=RealESRGAN is not None) # TODO: Feels like I shouldnt slot it in here.
                            txt2img_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=txt2img_defaults['ddim_eta'], visible=False)
                    txt2img_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=hasattr(model, "embedding_manager"))

            txt2img_btn.click(
                txt2img,
                [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_dynamic_threshold, txt2img_static_threshold, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings],
                [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
            )
            txt2img_prompt.submit(
                txt2img,
                [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_dynamic_threshold, txt2img_static_threshold, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings],
                [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
            )

        with gr.TabItem("Stable Diffusion Image-to-Image Unified", id="img2img_tab"):
            with gr.Row(elem_id="prompt_row"):
                img2img_prompt = gr.Textbox(label="Prompt", 
                elem_id='img2img_prompt_input',
                placeholder="A fantasy landscape, trending on artstation.", 
                lines=1,
                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25, 
                value=img2img_defaults['prompt'], 
                show_label=False).style()
                img2img_btn_mask = gr.Button("Generate",variant="primary", visible=False, elem_id="img2img_mask_btn").style(full_width=True)
                img2img_btn_editor = gr.Button("Generate",variant="primary", elem_id="img2img_editot_btn").style(full_width=True)
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    
                    img2img_image_editor_mode = gr.Radio(choices=["Mask", "Crop"], label="Image Editor Mode", value="Crop")
                    img2img_show_help_btn = gr.Button("Show Hints")
                    img2img_hide_help_btn = gr.Button("Hide Hints", visible=False)
                    img2img_help = gr.Markdown(visible=False, value="")
                    with gr.Row():
                        img2img_painterro_btn = gr.Button("Advanced Editor")
                        img2img_copy_from_painterro_btn = gr.Button(value="Get Image from Advanced Editor")
                    img2img_image_editor = gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil", tool="select")
                    img2img_image_mask = gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil", tool="sketch", visible=False)
                    img2img_mask = gr.Radio(choices=["Keep masked area", "Regenerate only masked area"], label="Mask Mode", type="index", value=img2img_mask_modes[img2img_defaults['mask_mode']], visible=False)
                    img2img_mask_blur_strength = gr.Slider(minimum=1, maximum=10, step=1, label="How much blurry should the mask be? (to avoid hard edges)", value=3, visible=False)
                    img2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=img2img_defaults['ddim_steps'])
                    img2img_sampling = gr.Radio(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=img2img_defaults['sampler_name'])
                    img2img_toggles = gr.CheckboxGroup(label='', choices=img2img_toggles, value=img2img_toggle_defaults, type="index")
                    img2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus', visible=RealESRGAN is not None) # TODO: Feels like I shouldnt slot it in here.
                    img2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1, label='Batch count (how many batches of images to generate)', value=img2img_defaults['n_iter'])
                    img2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=img2img_defaults['batch_size'])
                    img2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=img2img_defaults['cfg_scale'])
                    img2img_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength', value=img2img_defaults['denoising_strength'])
                    img2img_dynamic_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Dynamic Threshold', value=img2img_defaults['dynamic_threshold'])
                    img2img_static_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Static Threshold', value=img2img_defaults['static_threshold'])
                    img2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, value=img2img_defaults["seed"])
                    img2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=img2img_defaults["height"])
                    img2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=img2img_defaults["width"])
                    img2img_resize = gr.Radio(label="Resize mode", choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value=img2img_resize_modes[img2img_defaults['resize_mode']])
                    img2img_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=hasattr(model, "embedding_manager"))
                    
                with gr.Column():
                    output_img2img_gallery = gr.Gallery(label="Images")
                    output_img2img_select_image = gr.Number(label='Select image number from results for copying', value=1, precision=None)
                    gr.Markdown("Clear the input image before copying your output to your input. It may take some time to load the image.")
                    output_img2img_copy_to_input_btn = gr.Button("Copy selected image to input")
                    output_img2img_seed = gr.Number(label='Seed')
                    output_img2img_params = gr.Textbox(label="Copy-paste generation parameters")
                    output_img2img_stats = gr.HTML(label='Stats')

            img2img_image_editor_mode.change(
                change_image_editor_mode,
                [img2img_image_editor_mode, img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                [img2img_image_editor, img2img_image_mask, img2img_btn_editor, img2img_btn_mask, img2img_painterro_btn, img2img_copy_from_painterro_btn, img2img_mask, img2img_mask_blur_strength]
            )

            img2img_image_editor.edit(
                update_image_mask,
                [img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                img2img_image_mask
            )

            img2img_show_help_btn.click(
                show_help,
                None,
                [img2img_show_help_btn, img2img_hide_help_btn, img2img_help]
            )

            img2img_hide_help_btn.click(
                hide_help,
                None,
                [img2img_show_help_btn, img2img_hide_help_btn, img2img_help]
            )

            output_img2img_copy_to_input_btn.click(
                copy_img_to_input,
                [output_img2img_select_image, output_img2img_gallery],
                [img2img_image_editor, img2img_image_mask]
            )

            output_txt2img_copy_to_input_btn.click(
                copy_img_to_input,
                [output_txt2img_select_image, output_txt2img_gallery],
                [img2img_image_editor, img2img_image_mask, tabs]
            )

            img2img_btn_mask.click(
                img2img,
                [img2img_prompt, img2img_image_editor_mode, img2img_image_mask, img2img_mask, img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles, img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg, img2img_denoising, img2img_dynamic_threshold, img2img_static_threshold, img2img_seed, img2img_height, img2img_width, img2img_resize, img2img_embeddings],
                [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats]
            )

            img2img_btn_editor.click(
                img2img,
                [img2img_prompt, img2img_image_editor_mode, img2img_image_editor, img2img_mask, img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles, img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg, img2img_denoising, img2img_dynamic_threshold, img2img_static_threshold, img2img_seed, img2img_height, img2img_width, img2img_resize, img2img_embeddings],
                [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats]
            )

            img2img_painterro_btn.click(None, [img2img_image_editor], None, _js="""(img) => {
                try {
                    Painterro({
                        hiddenTools: ['arrow'],
                        saveHandler: function (image, done) {
                            localStorage.setItem('painterro-image', image.asDataURL());
                            done(true);
                        },
                    }).show(Array.isArray(img) ? img[0] : img);
                } catch(e) {
                    const script = document.createElement('script');
                    script.src = 'https://unpkg.com/painterro@1.2.78/build/painterro.min.js';
                    document.head.appendChild(script);
                    const style = document.createElement('style');
                    style.appendChild(document.createTextNode('.ptro-holder-wrapper { z-index: 9999 !important; }'));
                    document.head.appendChild(style);
                }
                return [];
            }""")

            img2img_copy_from_painterro_btn.click(None, None, [img2img_image_editor, img2img_image_mask], _js="""() => {
                const image = localStorage.getItem('painterro-image')
                return [image, image];
            }""")

        with gr.TabItem("Stable Diffusion Text Interpolation Unified", id='txt_interp_tab'):
            with gr.Row(elem_id="prompt_row"):
                txt_interp_prompt = gr.Textbox(label="Prompt", 
                elem_id='prompt_input',
                placeholder="An epic matte painting of a wizards potion room, featured on artstation\nAn epic matte painting of a dragons lair, featured on artstation", 
                lines=1,
                max_lines=100, # if txt_interp_defaults['submit_on_enter'] == 'Yes' else 25, 
                value=txt_interp_defaults['prompt'], 
                show_label=False).style()
                
            with gr.Row(elem_id='body').style(equal_height=False):
                with gr.Column():
                    txt_interp_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=txt_interp_defaults["height"])
                    txt_interp_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=txt_interp_defaults["width"])
                    txt_interp_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=txt_interp_defaults['cfg_scale'])
                    txt_interp_dynamic_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Dynamic Threshold', value=txt_interp_defaults['dynamic_threshold'])
                    txt_interp_static_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Static Threshold', value=txt_interp_defaults['static_threshold'])
                    txt_interp_degrees_per_second = gr.Slider(minimum=1, maximum=360, step=1, label='Degrees Per Second', value=txt_interp_defaults['degrees_per_second'])
                    txt_interp_frames_per_second = gr.Slider(minimum=1, maximum=360, step=1, label='Frames Per Second', value=txt_interp_defaults['frames_per_second'])
                    txt_interp_project_name = gr.Textbox(label="Project Name", lines=1, max_lines=1, value=txt_interp_defaults["project_name"])
                    txt_interp_seeds = gr.Textbox(label="Seeds (blank or None to randomize, seperate with newline)", lines=1, max_lines=100, value=txt_interp_defaults["seeds"])
                    # txt_interp_batch_count = gr.Slider(minimum=1, maximum=250, step=1, label='Batch count (how many batches of images to generate)', value=txt_interp_defaults['n_iter'])
                    txt_interp_batch_size = gr.Slider(minimum=1, maximum=20, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=txt_interp_defaults['batch_size'])
                with gr.Column():
                    output_txt_interp_gallery = gr.Gallery(label="Images", elem_id="gallery_output").style(grid=[4,4])
                    with gr.Row():
                        # with gr.Group():
                            # output_txt_interp_seed = gr.Number(label='Seed', interactive=False)
                            # output_txt_interp_copy_seed = gr.Button("Copy", full_width=True).click(inputs=output_txt_interp_seed, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                        with gr.Group():
                            output_txt_interp_select_image = gr.Number(label='Image # and click Copy to copy to img2img', value=1, precision=None)
                            output_txt_interp_copy_to_input_btn = gr.Button("Push to img2img", full_width=True)
                    # with gr.Group():
                    #     output_txt_interp_params = gr.Textbox(label="Copy-paste generation parameters", interactive=False)
                    #     output_txt_interp_copy_params = gr.Button("Copy", full_width=True).click(inputs=output_txt_interp_params, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                    # output_txt_interp_stats = gr.HTML(label='Stats')
                with gr.Column():
                    txt_interp_btn = gr.Button("Generate", full_width=True, elem_id="generate", variant="primary")
                    txt_interp_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=txt_interp_defaults['ddim_steps'])
                    txt_interp_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=txt_interp_defaults['sampler_name'])
                    with gr.Tabs():
                        # with gr.TabItem('Simple'):
                        #     txt_interp_submit_on_enter = gr.Radio(['Yes', 'No'], label="Submit on enter? (no means multiline)", value=txt_interp_defaults['submit_on_enter'], interactive=True)
                        #     txt_interp_submit_on_enter.change(lambda x: gr.update(max_lines=1 if x == 'Single' else 25) , txt_interp_submit_on_enter, txt_interp_prompt)
                        with gr.TabItem('Advanced'):
                            txt_interp_toggles = gr.CheckboxGroup(label='', choices=txt_interp_toggles, value=txt_interp_toggle_defaults, type="index")
                            txt_interp_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus', visible=RealESRGAN is not None) # TODO: Feels like I shouldnt slot it in here.
                            txt_interp_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=txt_interp_defaults['ddim_eta'], visible=False)
                    txt_interp_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=hasattr(model, "embedding_manager"))

            txt_interp_btn.click(
                txt_interp,
                [txt_interp_prompt, txt_interp_steps, txt_interp_sampling, txt_interp_toggles, txt_interp_realesrgan_model_name, txt_interp_ddim_eta, txt_interp_batch_size, txt_interp_cfg, txt_interp_dynamic_threshold, txt_interp_static_threshold, txt_interp_degrees_per_second, txt_interp_frames_per_second, txt_interp_project_name, txt_interp_seeds, txt_interp_height, txt_interp_width, txt_interp_embeddings],
                [output_txt_interp_gallery] #, output_txt_interp_seed, output_txt_interp_params, output_txt_interp_stats]
            )
            txt_interp_prompt.submit(
                txt_interp,
                [txt_interp_prompt, txt_interp_steps, txt_interp_sampling, txt_interp_toggles, txt_interp_realesrgan_model_name, txt_interp_ddim_eta, txt_interp_batch_size, txt_interp_cfg, txt_interp_dynamic_threshold, txt_interp_static_threshold, txt_interp_degrees_per_second, txt_interp_frames_per_second, txt_interp_project_name, txt_interp_seeds, txt_interp_height, txt_interp_width, txt_interp_embeddings],
                [output_txt_interp_gallery] #, output_txt_interp_seed, output_txt_interp_params, output_txt_interp_stats]
            )

        with gr.TabItem("Stable Diffusion Disco Animation Unified", id='disco_anim_tab'):
            with gr.Row(elem_id="prompt_row"):
                disco_anim_prompt = gr.Textbox(label="Prompt", 
                elem_id='prompt_input',
                placeholder="An epic matte painting of a wizards potion room, featured on artstation\nAn epic matte painting of a dragons lair, featured on artstation", 
                lines=1,
                max_lines=100, # if disco_anim_defaults['submit_on_enter'] == 'Yes' else 25, 
                value=disco_anim_defaults['prompt'], 
                show_label=False).style()
                
            with gr.Row(elem_id='body').style(equal_height=False):
                with gr.Column():
                    disco_anim_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=disco_anim_defaults["height"])
                    disco_anim_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=disco_anim_defaults["width"])
                    disco_anim_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=disco_anim_defaults['cfg_scale'])
                    disco_anim_dynamic_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Dynamic Threshold', value=disco_anim_defaults['dynamic_threshold'])
                    disco_anim_static_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Static Threshold', value=disco_anim_defaults['static_threshold'])
                    disco_anim_prev_frame_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Previous Frame Denoising Strength', value=disco_anim_defaults['prev_frame_denoising_strength'])
                    disco_anim_noise_between_frames = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label='Amount of noise to inject in between frames', value=disco_anim_defaults['noise_between_frames'])
                    disco_anim_degrees_per_second = gr.Slider(minimum=1, maximum=360, step=1, label='Degrees Per Second (if interpolating between prompts)', value=disco_anim_defaults['degrees_per_second'])
                    disco_anim_frames_per_second = gr.Slider(minimum=1, maximum=360, step=1, label='Frames Per Second  (if interpolating between prompts)', value=disco_anim_defaults['frames_per_second'])
                    disco_anim_project_name = gr.Textbox(label="Project Name", lines=1, max_lines=1, value=disco_anim_defaults["project_name"])
                    disco_anim_start_frame = gr.Number(precision=None, label="Start Frame (will use 0 if not resuming an animation)", value=0)
                    disco_anim_max_frames = gr.Number(precision=None, label="Max Frames in Animation", value=disco_anim_defaults['max_frames'])
                    disco_anim_seed = gr.Textbox(label="Seed (blank or None to randomize)", lines=1, max_lines=1, value='')
                    disco_anim_animation_mode = gr.Dropdown(label='Animation Mode', choices=["3D", "2D"], value=disco_anim_defaults['animation_mode']) # video mode WIP
                    disco_anim_interp_spline = gr.Dropdown(label='Spline Interpolation (Linear Recommended)', choices=["Linear", "Quadratic", "Cubic"], value=disco_anim_defaults['interp_spline'])
                    disco_anim_resize_mode = gr.Radio(label="Resize mode", choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value=img2img_resize_modes[disco_anim_defaults['resize_mode']])
                    disco_anim_color_match_mode = gr.Dropdown(label='Color Match Mode (if enabled)', choices=["RGB", "HSV", "LAB", "cycle"], value=disco_anim_defaults['color_match_mode'])
                    disco_anim_mix_factor = gr.Textbox(label="Amount of previous frame's latent to inject in between timesteps (1.0 - mix factor = amount mixed in)", lines=1, max_lines=1, value=disco_anim_defaults['mix_factor'])
                    with gr.Group():
                        disco_anim_angle = gr.Textbox(label='Angle', lines=1, max_lines=1, value=disco_anim_defaults['angle'])
                        disco_anim_zoom = gr.Textbox(label='Zoom', lines=1, max_lines=1, value=disco_anim_defaults['zoom'])
                        disco_anim_translation_x = gr.Textbox(label='Translation x', lines=1, max_lines=1, value=disco_anim_defaults['translation_x'])
                        disco_anim_translation_y = gr.Textbox(label='Translation y', lines=1, max_lines=1, value=disco_anim_defaults['translation_y'])
                        disco_anim_translation_z = gr.Textbox(label='Translation z', lines=1, max_lines=1, value=disco_anim_defaults['translation_z'])
                        disco_anim_rotation_3d_x = gr.Textbox(label='Rotation 3D x', lines=1, max_lines=1, value=disco_anim_defaults['rotation_3d_x'])
                        disco_anim_rotation_3d_y = gr.Textbox(label='Rotation 3D y', lines=1, max_lines=1, value=disco_anim_defaults['rotation_3d_y'])
                        disco_anim_rotation_3d_z = gr.Textbox(label='Rotation 3D z', lines=1, max_lines=1, value=disco_anim_defaults['rotation_3d_z'])
                        disco_anim_midas_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Angle', value=disco_anim_defaults['midas_weight'])
                        disco_anim_near_plane = gr.Slider(minimum=1, maximum=1000, step=1, label='Near Plane', value=disco_anim_defaults['near_plane'])
                        disco_anim_far_plane = gr.Slider(minimum=1, maximum=50000, step=1, label='Far Plane', value=disco_anim_defaults['far_plane'])
                        disco_anim_fov = gr.Slider(minimum=1, maximum=360, step=1, label='Field of View', value=disco_anim_defaults['fov'])
                        disco_anim_padding_mode= gr.Dropdown(label='Padding Mode', choices=["border"], value=disco_anim_defaults['padding_mode'])
                        disco_anim_sampling_mode = gr.Dropdown(label='Sampling Mode', choices=["bicubic"], value=disco_anim_defaults['sampling_mode'])
                        disco_anim_turbo_steps = gr.Slider(minimum=1, maximum=5, step=1, label='Turbo Steps', value=disco_anim_defaults['turbo_steps'])
                        disco_anim_turbo_preroll = gr.Slider(minimum=1, maximum=15, step=1, label='Turbo Preroll', value=disco_anim_defaults['turbo_preroll'])
                        disco_anim_vr_eye_angle = gr.Slider(minimum=1.0, maximum=10.0, step=0.01, label='vr Eye Angle', value=disco_anim_defaults['vr_eye_angle'])
                        disco_anim_vr_ipd = gr.Slider(minimum=1.0, maximum=10.0, step=0.01, label='vr IPD', value=disco_anim_defaults['vr_ipd'])
                    disco_anim_init_info = gr.Image(value=None, source="upload", interactive=True, type="pil", tool="select")
                    # with gr.Group():
                    #     disco_anim_extract_nth_frame = gr.Slider(minimum=1, maximum=100, step=1, label='Extract nth Frame', value=disco_anim_defaults['extract_nth_frame'])
                    #     disco_anim_video_init_flow_blend = gr.Slider(minimum=0, maximum=1, step=.01, label='Video Init Flow Blend', value=disco_anim_defaults['video_init_flow_blend'])
                    #     disco_anim_video_init_blend_mode= gr.Dropdown(label='Video Init Blend Mode', choices=['None', 'linear', 'optical flow'], value=disco_anim_defaults['video_init_blend_mode'])
                with gr.Column():
                    output_disco_anim_gallery = gr.Gallery(label="Images", elem_id="gallery_output").style(grid=[4,4])
                    with gr.Row():
                        # with gr.Group():
                        #     output_disco_anim_seed = gr.Number(label='Seed', interactive=False)
                        #     output_disco_anim_copy_seed = gr.Button("Copy", full_width=True).click(inputs=output_disco_anim_seed, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                        with gr.Group():
                            output_disco_anim_select_image = gr.Number(label='Image # and click Copy to copy to img2img', value=1, precision=None)
                            output_disco_anim_copy_to_input_btn = gr.Button("Push to img2img", full_width=True)
                    # with gr.Group():
                    #     output_disco_anim_params = gr.Textbox(label="Copy-paste generation parameters", interactive=False)
                    #     output_disco_anim_copy_params = gr.Button("Copy", full_width=True).click(inputs=output_disco_anim_params, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                    # output_disco_anim_stats = gr.HTML(label='Stats')
                with gr.Column():
                    disco_anim_btn = gr.Button("Generate", full_width=True, elem_id="generate", variant="primary")
                    # disco_anim_stop_anim = gr.Button("Stop Animation", full_width=True, elem_id="stop_animation", variant='primary')
                    disco_anim_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=disco_anim_defaults['ddim_steps'])
                    disco_anim_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=disco_anim_defaults['sampler_name'])
                    with gr.Tabs():
                        with gr.Group():
                            disco_anim_toggles = gr.CheckboxGroup(label='', choices=disco_anim_toggles, value=disco_anim_toggle_defaults, type="index")
                            disco_anim_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=disco_anim_defaults['ddim_eta'], visible=False)
                    disco_anim_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=hasattr(model, "embedding_manager"))

            disco_anim_btn.click(
                disco_anim,
                [disco_anim_prompt, disco_anim_init_info, disco_anim_project_name, disco_anim_steps, disco_anim_sampling, disco_anim_toggles, disco_anim_ddim_eta, disco_anim_cfg, disco_anim_color_match_mode, disco_anim_dynamic_threshold, disco_anim_static_threshold, disco_anim_degrees_per_second, disco_anim_frames_per_second, disco_anim_prev_frame_denoising, disco_anim_noise_between_frames, disco_anim_mix_factor, disco_anim_start_frame, disco_anim_max_frames, disco_anim_animation_mode, disco_anim_interp_spline, disco_anim_angle, disco_anim_zoom, disco_anim_translation_x, disco_anim_translation_y, disco_anim_translation_z, disco_anim_rotation_3d_x, disco_anim_rotation_3d_y, disco_anim_rotation_3d_z, disco_anim_midas_weight, disco_anim_near_plane, disco_anim_far_plane, disco_anim_fov, disco_anim_padding_mode, disco_anim_sampling_mode, disco_anim_turbo_steps, disco_anim_turbo_preroll, disco_anim_vr_eye_angle, disco_anim_vr_ipd, disco_anim_seed, disco_anim_height, disco_anim_width, disco_anim_resize_mode, disco_anim_embeddings],
                [output_disco_anim_gallery] #, output_disco_anim_seed, output_disco_anim_params, output_disco_anim_stats]
            )
            disco_anim_prompt.submit(
                disco_anim,
                [disco_anim_prompt, disco_anim_init_info, disco_anim_project_name, disco_anim_steps, disco_anim_sampling, disco_anim_toggles, disco_anim_ddim_eta, disco_anim_cfg, disco_anim_color_match_mode, disco_anim_dynamic_threshold, disco_anim_static_threshold, disco_anim_degrees_per_second, disco_anim_frames_per_second, disco_anim_prev_frame_denoising, disco_anim_noise_between_frames, disco_anim_mix_factor, disco_anim_start_frame, disco_anim_max_frames, disco_anim_animation_mode, disco_anim_interp_spline, disco_anim_angle, disco_anim_zoom, disco_anim_translation_x, disco_anim_translation_y, disco_anim_translation_z, disco_anim_rotation_3d_x, disco_anim_rotation_3d_y, disco_anim_rotation_3d_z, disco_anim_midas_weight, disco_anim_near_plane, disco_anim_far_plane, disco_anim_fov, disco_anim_padding_mode, disco_anim_sampling_mode, disco_anim_turbo_steps, disco_anim_turbo_preroll, disco_anim_vr_eye_angle, disco_anim_vr_ipd, disco_anim_seed, disco_anim_height, disco_anim_width, disco_anim_resize_mode, disco_anim_embeddings],
                [output_disco_anim_gallery] #, output_disco_anim_seed, output_disco_anim_params, output_disco_anim_stats]
            )

            # disco_anim_stop_anim.click(
            #     stop_anim,
            #     [],
            #     []
            # )

        if GFPGAN is not None:
            gfpgan_defaults = {
                'strength': 100,
            }

            if 'gfpgan' in user_defaults:
                gfpgan_defaults.update(user_defaults['gfpgan'])

            with gr.TabItem("GFPGAN"):
                gr.Markdown("Fix faces on images")
                with gr.Row():
                    with gr.Column():
                        gfpgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                        gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength", value=gfpgan_defaults['strength'])
                        gfpgan_btn = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        gfpgan_output = gr.Image(label="Output")
                gfpgan_btn.click(
                    run_GFPGAN,
                    [gfpgan_source, gfpgan_strength],
                    [gfpgan_output]
                )
        if RealESRGAN is not None:
            with gr.TabItem("RealESRGAN"):
                gr.Markdown("Upscale images")
                with gr.Row():
                    with gr.Column():
                        realesrgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                        realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus')
                        realesrgan_btn = gr.Button("Generate")
                    with gr.Column():
                        realesrgan_output = gr.Image(label="Output")
                realesrgan_btn.click(
                    run_RealESRGAN,
                    [realesrgan_source, realesrgan_model_name],
                    [realesrgan_output]
                )


class ServerLauncher(threading.Thread):
    def __init__(self, demo):
        threading.Thread.__init__(self)
        self.name = 'Gradio Server Thread'
        self.demo = demo

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gradio_params = {
            'show_error': True, 
            'server_name': '0.0.0.0', 
            'share': opt.share
        }
        if not opt.share:
            demo.queue(concurrency_count=1)
        if opt.share and opt.share_password:
            gradio_params['auth'] = ('webui', opt.share_password)    
        self.demo.launch(**gradio_params)

    def stop(self):
        self.demo.close() # this tends to hang

def launch_server():
    server_thread = ServerLauncher(demo)
    server_thread.start()

    try:
        while server_thread.is_alive():
            time.sleep(60)
    except (KeyboardInterrupt, OSError) as e:
        crash(e, 'Shutting down...')

def run_headless():
    with open(opt.cli, 'r', encoding='utf8') as f:
        kwargs = yaml.safe_load(f)
    target = kwargs.pop('target')
    if target == 'txt2img':
        target_func = txt2img
    elif target == 'img2img':
        target_func = img2img
        raise NotImplementedError()
    else:
        raise ValueError(f'Unknown target: {target}')
    prompts = kwargs.pop("prompt")
    prompts = prompts if type(prompts) is list else [prompts]
    for i, prompt_i in enumerate(prompts):
        print(f"===== Prompt {i+1}/{len(prompts)}: {prompt_i} =====")
        output_images, seed, info, stats = target_func(prompt=prompt_i, **kwargs)
        print(f'Seed: {seed}')
        print(info)
        print(stats)
        print()

if __name__ == '__main__':
    if opt.cli is None:
        launch_server()
    else:
        run_headless()
