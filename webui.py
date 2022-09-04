import argparse, os, sys, glob, re

# sys.path.append(f'{os.path.abspath(os.getcwd())}/k-diffusion')
# sys.path.append(f'{os.path.abspath(os.getcwd())}/stable-diffusion')
# sys.path.append(f'{os.path.abspath(os.getcwd())}/taming-transformers')
# sys.path.append(f'{os.path.abspath(os.getcwd())}/AdaBins')
# sys.path.append(f'{os.path.abspath(os.getcwd())}/MiDaS/midas_utils')
# sys.path.append(f'{os.path.abspath(os.getcwd())}/MiDaS') 
# sys.path.append(f'{os.path.abspath(os.getcwd())}/pytorch3d-lite')
# sys.path.append(f'{os.path.abspath(os.getcwd())}/disco-diffusion')

sys.path.append('E:/Python/k-diffusion')
sys.path.append('E:/Python/stable-diffusion')
sys.path.append('E:/Python/taming-transformers')
sys.path.append('E:/Python/AdaBins')
sys.path.append('E:/Python/MiDaS/midas_utils')
sys.path.append('E:/Python/MiDaS') 
sys.path.append('E:/Python/pytorch3d-lite')
sys.path.append('E:/Python/disco-diffusion')

from utils import *
from frontend.frontend import draw_gradio_ui
from frontend.job_manager import JobManager, JobInfo
from frontend.ui_functions import resize_image


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model",)
parser.add_argument("--cli", type=str, help="don't launch web server, take Python function kwargs from this file.", default=None)
parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model",)
parser.add_argument("--defaults", type=str, help="path to configuration file providing UI defaults, uses same format as cli parameter", default='configs/webui/webui.yaml')
parser.add_argument("--esrgan-cpu", action='store_true', help="run ESRGAN on cpu", default=False)
parser.add_argument("--esrgan-gpu", type=int, help="run ESRGAN on specific gpu (overrides --gpu)", default=0)
parser.add_argument("--extra-models-cpu", action='store_true', help="run extra models (GFGPAN/ESRGAN) on cpu", default=False)
parser.add_argument("--extra-models-gpu", action='store_true', help="run extra models (GFGPAN/ESRGAN) on cpu", default=False)
parser.add_argument("--gfpgan-cpu", action='store_true', help="run GFPGAN on cpu", default=False)
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN')) # i disagree with where you're putting it but since all guidefags are doing it this way, there you go
parser.add_argument("--gfpgan-gpu", type=int, help="run GFPGAN on specific gpu (overrides --gpu) ", default=0)
parser.add_argument("--gpu", type=int, help="choose which GPU to use if you have multiple", default=0)
parser.add_argument("--grid-format", type=str, help="png for lossless png files; jpg:quality for lossy jpeg; webp:quality for lossy webp, or webp:-compression for lossless webp", default="jpg:95")
parser.add_argument("--inbrowser", action='store_true', help="automatically launch the interface in a new tab on the default browser", default=False)
parser.add_argument("--ldsr-dir", type=str, help="LDSR directory", default=('./src/latent-diffusion' if os.path.exists('./src/latent-diffusion') else './LDSR'))
parser.add_argument("--n_rows", type=int, default=-1, help="rows in the grid; use -1 for autodetect and 0 for n_rows to be same as batch_size (default: -1)",)
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats", default=False)
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)", default=False)
parser.add_argument("--no-verify-input", action='store_true', help="do not verify input to check if it's too long", default=False)
parser.add_argument("--optimized-turbo", action='store_true', help="alternative optimization mode that does not save as much VRAM but runs siginificantly faster")
parser.add_argument("--optimized", action='store_true', help="load the model onto the device piecemeal instead of all at once to reduce VRAM usage at the cost of performance")
parser.add_argument("--outdir_img2img", type=str, nargs="?", help="dir to write img2img results to (overrides --outdir)", default=None)
parser.add_argument("--outdir_imglab", type=str, nargs="?", help="dir to write imglab results to (overrides --outdir)", default=None)
parser.add_argument("--outdir_txt2img", type=str, nargs="?", help="dir to write txt2img results to (overrides --outdir)", default=None)
parser.add_argument("--outdir_txt_interp", type=str, nargs="?", help="dir to write text_interp results to (overrides --outdir)", default=None)
parser.add_argument("--outdir_disco_anim", type=str, nargs="?", help="dir to write disco_anim results to (overrides --outdir)", default=None)
parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default=None)
parser.add_argument("--port", type=int, help="choose the port for the gradio webserver to use", default=7860)
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--realesrgan-dir", type=str, help="RealESRGAN directory", default=('./src/realesrgan' if os.path.exists('./src/realesrgan') else './RealESRGAN'))
parser.add_argument("--realesrgan-model", type=str, help="Upscaling model for RealESRGAN", default=('RealESRGAN_x4plus'))
parser.add_argument("--save-metadata", action='store_true', help="Store generation parameters in the output png. Drop saved png into Image Lab to read parameters", default=False)
parser.add_argument("--share-password", type=str, help="Sharing is open by default, use this to set a password. Username: webui", default=None)
parser.add_argument("--share", action='store_true', help="Should share your server on gradio.app, this allows you to use the UI from your mobile app", default=False)
parser.add_argument("--skip-grid", action='store_true', help="do not save a grid, only individual samples. Helpful when evaluating lots of samples", default=False)
parser.add_argument("--skip-save", action='store_true', help="do not save indiviual samples. For speed measurements.", default=False)
parser.add_argument('--no-job-manager', action='store_true', help="Don't use the experimental job manager on top of gradio", default=False)
parser.add_argument("--max-jobs", type=int, help="Maximum number of concurrent 'generate' commands", default=1)
opt = parser.parse_args()

#Should not be needed anymore
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# all selected gpus, can probably be done nicer
#if opt.extra_models_gpu:
#    gpus = set([opt.gpu, opt.esrgan_gpu, opt.gfpgan_gpu])
#    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(g) for g in set(gpus))
#else:
#    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

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
from typing import List, Union, Dict
from pathlib import Path
from tqdm.auto import tqdm
from collections import namedtuple

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

stop_on_next_loop = False

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
invalid_filename_chars = '<>:"/\|?*\n'

GFPGAN_dir = opt.gfpgan_dir
RealESRGAN_dir = opt.realesrgan_dir
LDSR_dir = opt.ldsr_dir

if opt.optimized_turbo:
    opt.optimized = True

if opt.no_job_manager:
    job_manager = None
else:
    job_manager = JobManager(opt.max_jobs)
    opt.max_jobs += 1 # Leave a free job open for button clicks

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


def stop_anim():
    global stop_on_next_loop
    stop_on_next_loop = True


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
    try:
        del model
        del device
    except:
        try:
            del device
        except:
            pass
        pass

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

class CFGMaskedDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
        x_in = x
        x_in = torch.cat([x_in] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        denoised = uncond + (cond - uncond) * cond_scale

        if mask is not None:
            assert x0 is not None
            img_orig = x0
            mask_inv = 1. - mask
            denoised = (img_orig * mask_inv) + (mask * denoised)

        return denoised

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


    def get_sampler_name(self):
        return self.schedule
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
def load_LDSR(checking=False):
    model_name = 'model'
    yaml_name = 'project'
    model_path = os.path.join(LDSR_dir, 'experiments/pretrained_models', model_name + '.ckpt')
    yaml_path = os.path.join(LDSR_dir, 'experiments/pretrained_models', yaml_name + '.yaml')
    if not os.path.isfile(model_path):
        raise Exception("LDSR model not found at path "+model_path)
    if not os.path.isfile(yaml_path):
        raise Exception("LDSR model not found at path "+yaml_path)
    if checking == True:
        return True

    sys.path.append(os.path.abspath(LDSR_dir))
    from LDSR import LDSR
    LDSRObject = LDSR(model_path, yaml_path)
    return LDSRObject
def load_GFPGAN(checking=False):
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(GFPGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception("GFPGAN model not found at path "+model_path)
    if checking == True:
        return True
    sys.path.append(os.path.abspath(GFPGAN_dir))
    from gfpgan import GFPGANer

    if opt.gfpgan_cpu or opt.extra_models_cpu:
        instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=torch.device('cpu'))
    elif opt.extra_models_gpu:
        instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=torch.device(f'cuda:{opt.gfpgan_gpu}'))
    else:
        instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=torch.device(f'cuda:{opt.gpu}'))
    return instance

def load_RealESRGAN(model_name: str, checking = False):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    RealESRGAN_models = {
        'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    }

    model_path = os.path.join(RealESRGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception(model_name+".pth not found at path "+model_path)
    if checking == True:
        return True
    sys.path.append(os.path.abspath(RealESRGAN_dir))
    from realesrgan import RealESRGANer

    if opt.esrgan_cpu or opt.extra_models_cpu:
        instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=False) # cpu does not support half
        instance.device = torch.device('cpu')
        instance.model.to('cpu')
    elif opt.extra_models_gpu:
        instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=not opt.no_half, gpu_id=opt.esrgan_gpu)
    else:
        instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=not opt.no_half)
    instance.model.name = model_name
    return instance

GFPGAN = None
if os.path.exists(GFPGAN_dir):
    try:
        GFPGAN = load_GFPGAN(checking=True)
        print("Found GFPGAN")
    except Exception:
        import traceback
        print("Error loading GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

RealESRGAN = None
def try_loading_RealESRGAN(model_name: str,checking=False):
    global RealESRGAN
    if os.path.exists(RealESRGAN_dir):
        try:
            RealESRGAN = load_RealESRGAN(model_name,checking) # TODO: Should try to load both models before giving up
            if checking == True:
                print("Found RealESRGAN")
                return True
            print("Loaded RealESRGAN with model "+RealESRGAN.model.name)
        except Exception:
            import traceback
            print("Error loading RealESRGAN:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
try_loading_RealESRGAN('RealESRGAN_x4plus',checking=True)

LDSR = None
def try_loading_LDSR(model_name: str,checking=False):
    global LDSR
    if os.path.exists(LDSR_dir):
        try:
            LDSR = load_LDSR(checking=True) # TODO: Should try to load both models before giving up
            if checking == True:
                print("Found LDSR")
                return True
            print("Latent Diffusion Super Sampling (LDSR) model loaded")
        except Exception:
            import traceback
            print("Error loading LDSR:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    else:
        print("LDSR not found at path, please make sure you have cloned the LDSR repo to ./src/latent-diffusion/")
try_loading_LDSR('model',checking=True)

def load_SD_model():
    if opt.optimized:
        sd = load_sd_from_config(opt.ckpt)
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
        device = torch.device(f"cuda:{opt.gpu}") if torch.cuda.is_available() else torch.device("cpu")

        model = instantiate_from_config(config.modelUNet)
        _, _ = model.load_state_dict(sd, strict=False)
        model.cuda()
        model.eval()
        model.turbo = opt.optimized_turbo

        modelCS = instantiate_from_config(config.modelCondStage)
        _, _ = modelCS.load_state_dict(sd, strict=False)
        modelCS.cond_stage_model.device = device
        modelCS.eval()

        modelFS = instantiate_from_config(config.modelFirstStage)
        _, _ = modelFS.load_state_dict(sd, strict=False)
        modelFS.eval()

        del sd

        if not opt.no_half:
            model = model.half()
            modelCS = modelCS.half()
            modelFS = modelFS.half()
        return model,modelCS,modelFS,device, config
    else:
        config = OmegaConf.load(opt.config)
        model = load_model_from_config(config, opt.ckpt)

        device = torch.device(f"cuda:{opt.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        model = (model if opt.no_half else model.half()).to(device)
    return model, device,config

if opt.optimized:
    model,modelCS,modelFS,device, config = load_SD_model()
else:
    model, device,config = load_SD_model()


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
        if captions and i<len(captions):
            d = ImageDraw.Draw( grid )
            size = d.textbbox( (0,0), captions[i], font=fnt, stroke_width=2, align="center" )
            d.multiline_text((i % cols * w + w/2, i // cols * h + h - size[3]), captions[i], font=fnt, fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0), anchor="mm", align="center")

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
use_GFPGAN, write_info_files, write_sample_info_to_log_file, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, skip_metadata=True):
    filename_i = os.path.join(sample_path_i, filename)
    if not jpg_sample:
        if opt.save_metadata and not skip_metadata:
            metadata = PngInfo()
            metadata.add_text("SD:prompt", prompts[i] if prompts is not None else '')
            metadata.add_text("SD:seed", str(seeds[i] if seeds is not None else ''))
            metadata.add_text("SD:width", str(width))
            metadata.add_text("SD:height", str(height))
            metadata.add_text("SD:steps", str(steps))
            metadata.add_text("SD:cfg_scale", str(cfg_scale))
            if init_img is not None:
                metadata.add_text("SD:denoising_strength", str(denoising_strength))
            metadata.add_text("SD:GFPGAN", str(use_GFPGAN and GFPGAN is not None))
            image.save(f"{filename_i}.png", pnginfo=metadata)
        else:
            image.save(f"{filename_i}.png")
    else:
        image.save(f"{filename_i}.jpg", 'jpeg', quality=100, optimize=True)
    if write_info_files or write_sample_info_to_log_file:
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
        if write_info_files:
            with open(f"{filename_i}.yaml", "w", encoding="utf8") as f:
                yaml.dump(info_dict, f, allow_unicode=True, width=10000)

        if write_sample_info_to_log_file:
            ignore_list = ["prompt", "target", "toggles", "ddim_eta", "batch_size"]
            rename_dict = {"ddim_steps": "steps", "n_iter": "number", "sampler_name": "sampler"} #changes the name of parameters to match with dynamic parameters
            sample_log_path = os.path.join(sample_path_i, "log.yaml")
            log_dump = info_dict.get("prompt") # making sure the first item that is listed in the txt is the prompt text
            for key, value in info_dict.items():
                if key in ignore_list:
                    continue
                found_key = rename_dict.get(key)

                if key == "cfg_scale": #adds zeros to to cfg_scale necessary for dynamic params
                    value = str(value).zfill(2)

                if found_key:
                    key = found_key
                log_dump += f" {key} {value}"

            log_dump = log_dump + " \n" #space at the end for dynamic params to accept the last param
            with open(sample_log_path, "a", encoding="utf8") as log_file:
                log_file.write(log_dump)



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


def oxlamon_matrix(prompt, seed, n_iter, batch_size):
    pattern = re.compile(r'(,\s){2,}')

    class PromptItem:
        def __init__(self, text, parts, item):
            self.text = text
            self.parts = parts
            if item:
                self.parts.append( item )

    def clean(txt):
        return re.sub(pattern, ', ', txt)

    def getrowcount( txt ):
        for data in re.finditer( ".*?\\((.*?)\\).*", txt ):
            if data:
                return len(data.group(1).split("|"))
            break
        return None

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

    def classToArrays( items, seed, n_iter ):
        texts = []
        parts = []
        seeds = []

        for item in items:
            itemseed = seed
            for i in range(n_iter):
                texts.append( item.text )
                parts.append( f"Seed: {itemseed}\n" + "\n".join(item.parts) )
                seeds.append( itemseed )
                itemseed += 1

        return seeds, texts, parts

    all_seeds, all_prompts, prompt_matrix_parts = classToArrays(getmatrix( prompt ), seed, n_iter)
    n_iter = math.ceil(len(all_prompts) / batch_size)

    needrows = getrowcount(prompt)
    if needrows:
        xrows = math.sqrt(len(all_prompts))
        xrows = round(xrows)
        # if columns is to much
        cols = math.ceil(len(all_prompts) / xrows)
        if cols > needrows*4:
            needrows *= 2

    return all_seeds, n_iter, prompt_matrix_parts, all_prompts, needrows


# THIS FUNCTION IS A TEMPORARY WORKAROUND FOR IN-PROGRESS ANIMS
def _process_images(
        outpath, func_init, func_sample, prompt, seed, sampler_name, skip_grid, skip_save, batch_size,
        n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN, realesrgan_model_name,
        fp, ddim_eta=0.0, do_not_save_grid=False, init_info=None, denoising_strength=0.75, resize_mode=None, uses_loopback=False,
        uses_random_seed_loopback=False, sort_samples=True, write_info_files=True, write_sample_info_to_log_file=False, jpg_sample=False, do_interpolation=False,
        project_name='interp', fps=30, variant_amount=0.0, variant_seed=None, imgProcessorTask=False, job_info: JobInfo = None):
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    prompt = prompt or ''
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

    if isinstance(prompt, str):
        if not ("|" in prompt) and prompt.startswith("@"):
            prompt = prompt[1:]

    comments = []

    prompt_matrix_parts = []
    simple_templating = False
    add_original_image = True
    if prompt_matrix and not do_interpolation:
        if prompt.startswith("@"):
            simple_templating = True
            add_original_image = not (use_RealESRGAN or use_GFPGAN)
            all_seeds, n_iter, prompt_matrix_parts, all_prompts, frows = oxlamon_matrix(prompt, seed, n_iter, batch_size)
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
            all_seeds = seed_everything()
    original_seeds = all_seeds.copy()

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    if job_info:
        output_images = job_info.images
    else:
        output_images = []
    grid_captions = []
    stats = []
    frame = 1
    if do_interpolation and not skip_grid:
        video_out = imageio.get_writer(f"{outpath}/{project_name}.mp4", mode='I', fps=fps, codec='libx264')
    with torch.no_grad(), precision_scope("cuda"), (model.ema_scope() if not opt.optimized else nullcontext()):
        init_data = func_init(init_info)
        tic = time.time()


        # if variant_amount > 0.0 create noise from base seed
        base_x = None
        if variant_amount > 0.0:
            target_seed_randomizer = seed_to_int('') # random seed
            torch.manual_seed(seed) # this has to be the single starting seed (not per-iteration)
            base_x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=[seed])
            # we don't want all_seeds to be sequential from starting seed with variants,
            # since that makes the same variants each time,
            # so we add target_seed_randomizer as a random offset
            for si in range(len(all_seeds)):
                all_seeds[si] += target_seed_randomizer

        for n in range(n_iter):
            if job_info and job_info.should_stop.is_set():
                print("Early exit requested")
                break

            print(f"Iteration: {n+1}/{n_iter}")

            if not do_interpolation:
                prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
                captions = prompt_matrix_parts[n * batch_size:(n + 1) * batch_size]
                seeds = all_seeds[n * batch_size:(n + 1) * batch_size]
                current_seeds = original_seeds[n * batch_size:(n + 1) * batch_size]

                if job_info:
                    job_info.job_status = f"Processing Iteration {n+1}/{n_iter}. Batch size {batch_size}"
                    for idx,(p,s) in enumerate(zip(prompts,seeds)):
                        job_info.job_status += f"\nItem {idx}: Seed {s}\nPrompt: {p}"
            elif do_interpolation:
                c = torch.cat(tuple(all_prompts[n]))
                x = torch.cat(tuple(torch.stack(list(all_seeds[n]), dim=0)))

                if job_info:
                    job_info.job_status = f"Processing Iteration {n+1}/{n_iter}. Batch size {batch_size}"

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
                cur_variant_amount = variant_amount 
                if variant_amount == 0.0:
                    # we manually generate all input noises because each one should have a specific seed
                    x = create_random_tensors(shape, seeds=seeds)
                else: # we are making variants
                    # using variant_seed as sneaky toggle,
                    # when not None or '' use the variant_seed
                    # otherwise use seeds
                    if variant_seed != None and variant_seed != '':
                        specified_variant_seed = seed_to_int(variant_seed)
                        torch.manual_seed(specified_variant_seed)
                        target_x = create_random_tensors(shape, seeds=[specified_variant_seed])
                        # with a variant seed we would end up with the same variant as the basic seed
                        # does not change. But we can increase the steps to get an interesting result
                        # that shows more and more deviation of the original image and let us adjust
                        # how far we will go (using 10 iterations with variation amount set to 0.02 will
                        # generate an icreasingly variated image which is very interesting for movies)
                        cur_variant_amount += n*variant_amount
                    else:
                        target_x = create_random_tensors(shape, seeds=seeds)
                    # finally, slerp base_x noise to target_x noise for creating a variant
                    x = slerp(device, max(0.0, min(1.0, cur_variant_amount)), base_x, target_x)

            samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)

            if opt.optimized:
                modelFS.to(device)

            x_samples_ddim = (model if not opt.optimized else modelFS).decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            for i, x_sample in enumerate(x_samples_ddim):
                if not do_interpolation:
                    sanitized_prompt = prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})
                    if variant_seed != None and variant_seed != '':
                        if variant_amount == 0.0:
                            seed_used = f"{current_seeds[i]}-{variant_seed}"
                        else:
                            seed_used = f"{seed}-{variant_seed}"
                    else:
                        seed_used = f"{current_seeds[i]}"
                    if sort_samples:
                        sanitized_prompt = sanitized_prompt[:128] #200 is too long
                        sample_path_i = os.path.join(sample_path, sanitized_prompt)
                        os.makedirs(sample_path_i, exist_ok=True)
                        base_count = get_next_sequence_number(sample_path_i)
                        filename = f"{base_count:05}-{steps}_{sampler_name}_{seed_used}_{cur_variant_amount:.2f}"
                    else:
                        sample_path_i = sample_path
                        base_count = get_next_sequence_number(sample_path_i)
                        sanitized_prompt = sanitized_prompt
                        filename = f"{base_count:05}-{steps}_{sampler_name}_{seed_used}_{cur_variant_amount:.2f}_{sanitized_prompt}"[:128] #same as before
                elif do_interpolation:
                    sample_path_i = outpath.strip('\\') + '/frames'
                    filename = f"{project_name}_{frame}"
                    frame += 1

                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)
                original_sample = x_sample
                original_filename = filename
                if use_GFPGAN and GFPGAN is not None and not use_RealESRGAN:
                    skip_save = True # #287 >_>
                    torch_gc()
                    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(original_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    gfpgan_sample = restored_img[:,:,::-1]
                    gfpgan_image = Image.fromarray(gfpgan_sample)
                    gfpgan_filename = original_filename + '-gfpgan'
                    save_sample(gfpgan_image, sample_path_i, gfpgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
normalize_prompt_weights, use_GFPGAN, write_info_files, write_sample_info_to_log_file, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, skip_metadata=True)
                    output_images.append(gfpgan_image) #287
                    #if simple_templating:
                    #    grid_captions.append( captions[i] + "\ngfpgan" )

                if use_RealESRGAN and RealESRGAN is not None and not use_GFPGAN:
                    skip_save = True # #287 >_>
                    torch_gc()
                    output, img_mode = RealESRGAN.enhance(original_sample[:,:,::-1])
                    esrgan_filename = original_filename + '-esrgan4x'
                    esrgan_sample = output[:,:,::-1]
                    esrgan_image = Image.fromarray(esrgan_sample)
                    save_sample(esrgan_image, sample_path_i, esrgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
normalize_prompt_weights, use_GFPGAN,write_info_files, write_sample_info_to_log_file, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, skip_metadata=True)
                    output_images.append(esrgan_image) #287
                    #if simple_templating:
                    #    grid_captions.append( captions[i] + "\nesrgan" )

                if use_RealESRGAN and RealESRGAN is not None and use_GFPGAN and GFPGAN is not None:
                    skip_save = True # #287 >_>
                    torch_gc()
                    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    gfpgan_sample = restored_img[:,:,::-1]
                    output, img_mode = RealESRGAN.enhance(gfpgan_sample[:,:,::-1])
                    gfpgan_esrgan_filename = original_filename + '-gfpgan-esrgan4x'
                    gfpgan_esrgan_sample = output[:,:,::-1]
                    gfpgan_esrgan_image = Image.fromarray(gfpgan_esrgan_sample)
                    save_sample(gfpgan_esrgan_image, sample_path_i, gfpgan_esrgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
normalize_prompt_weights, use_GFPGAN, write_info_files, write_sample_info_to_log_file, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, skip_metadata=True)
                    output_images.append(gfpgan_esrgan_image) #287
                    #if simple_templating:
                    #    grid_captions.append( captions[i] + "\ngfpgan_esrgan" )

                image = Image.fromarray(x_sample)

                # this flag is used for imgProcessorTasks like GoBig, will return the image without saving it
                if imgProcessorTask == True:
                    output_images.append(image)

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
    use_GFPGAN, write_info_files, write_sample_info_to_log_file, prompt_matrix, init_info, uses_loopback, uses_random_seed_loopback, skip_save,
    skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, False)
                    if add_original_image or not simple_templating:
                            output_images.append(image)
                            if simple_templating:
                                grid_captions.append(captions[i])
                    else:
                        save_sample(image, sample_path_i, filename, jpg_sample, None, None, width, height, steps, cfg_scale, 
    use_GFPGAN, write_info_files, False, False, None, False, False, skip_save,
    skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, False)

                output_images.append(image)
                # if do_interpolation:
                #     yield image, f"Frame: {n+1}/{n_iter}\nDirectory: {sample_path_i}" if n+1 != n_iter else f"Completed! Frames are available at {sample_path_i}"
                if do_interpolation and not skip_grid:
                    video_out.append_data(x_sample)

        if (prompt_matrix or not skip_grid) and not do_not_save_grid and not do_interpolation:
            if prompt_matrix:
                if simple_templating:
                    grid = image_grid(output_images, batch_size, force_n_rows=frows, captions=grid_captions)
                else:
                    grid = image_grid(output_images, batch_size, force_n_rows=1 << ((len(prompt_matrix_parts)-1)//2))
                    try:
                        grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                    except:
                        import traceback
                        print("Error creating prompt_matrix text:", file=sys.stderr)
                        print(traceback.format_exc(), file=sys.stderr)
            elif batch_size > 1  or n_iter > 1:
                grid = image_grid(output_images, batch_size)

            if prompt_matrix and not prompt.startswith("@") and not do_interpolation:
                try:
                    grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                except:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

            else:
               grid = image_grid(output_images, batch_size)

            output_images.insert(0, grid)

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

    args_and_names = {
        "seed": seed if not do_interpolation else '',
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler": sampler_name,
    }

    full_string = f"{prompt if not do_interpolation else ''}\n"+ " ".join([f"{k}:" for k,v in args_and_names.items()])
    info = {
        'text': full_string,
        'entities': [{'entity':str(v), 'start': full_string.find(f"{k}:"),'end': full_string.find(f"{k}:") + len(f"{k} ")} for k,v in args_and_names.items()]
     }
    stats = f'''
Took { round(time_diff, 2) }s total ({ round(time_diff/(len(all_prompts)),2) }s per image)
Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

    for comment in comments:
        info['text'] += "\n\n" + comment

    #mem_mon.stop()
    #del mem_mon
    torch_gc()

    if not do_interpolation:
        return output_images, seed, info, stats


# THIS FUNCTION IS A TEMPORARY WORKAROUND FOR IN-PROGRESS ANIMS
def process_images(
        outpath, func_init, func_sample, prompt, seed, sampler_name, skip_grid, skip_save, batch_size,
        n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN, realesrgan_model_name,
        fp, ddim_eta=0.0, do_not_save_grid=False, init_info=None, denoising_strength=0.75, resize_mode=None, uses_loopback=False,
        uses_random_seed_loopback=False, sort_samples=True, write_info_files=True, write_sample_info_to_log_file=False, jpg_sample=False, do_interpolation=False,
        project_name='interp', fps=30, variant_amount=0.0, variant_seed=None, imgProcessorTask=False, job_info: JobInfo = None):
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    prompt = prompt or ''
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

    if isinstance(prompt, str):
        if not ("|" in prompt) and prompt.startswith("@"):
            prompt = prompt[1:]

    comments = []

    prompt_matrix_parts = []
    simple_templating = False
    add_original_image = True
    if prompt_matrix and not do_interpolation:
        if prompt.startswith("@"):
            simple_templating = True
            add_original_image = not (use_RealESRGAN or use_GFPGAN)
            all_seeds, n_iter, prompt_matrix_parts, all_prompts, frows = oxlamon_matrix(prompt, seed, n_iter, batch_size)
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
            all_seeds = seed_everything()
    original_seeds = all_seeds.copy()

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    if job_info:
        output_images = job_info.images
    else:
        output_images = []
    grid_captions = []
    stats = []
    frame = 1
    if do_interpolation and not skip_grid:
        video_out = imageio.get_writer(f"{outpath}/{project_name}.mp4", mode='I', fps=fps, codec='libx264')
    with torch.no_grad(), precision_scope("cuda"), (model.ema_scope() if not opt.optimized else nullcontext()):
        init_data = func_init(init_info)
        tic = time.time()


        # if variant_amount > 0.0 create noise from base seed
        base_x = None
        if variant_amount > 0.0:
            target_seed_randomizer = seed_to_int('') # random seed
            torch.manual_seed(seed) # this has to be the single starting seed (not per-iteration)
            base_x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=[seed])
            # we don't want all_seeds to be sequential from starting seed with variants,
            # since that makes the same variants each time,
            # so we add target_seed_randomizer as a random offset
            for si in range(len(all_seeds)):
                all_seeds[si] += target_seed_randomizer

        for n in range(n_iter):
            if job_info and job_info.should_stop.is_set():
                print("Early exit requested")
                break

            print(f"Iteration: {n+1}/{n_iter}")

            if not do_interpolation:
                prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
                captions = prompt_matrix_parts[n * batch_size:(n + 1) * batch_size]
                seeds = all_seeds[n * batch_size:(n + 1) * batch_size]
                current_seeds = original_seeds[n * batch_size:(n + 1) * batch_size]

                if job_info:
                    job_info.job_status = f"Processing Iteration {n+1}/{n_iter}. Batch size {batch_size}"
                    for idx,(p,s) in enumerate(zip(prompts,seeds)):
                        job_info.job_status += f"\nItem {idx}: Seed {s}\nPrompt: {p}"
            elif do_interpolation:
                c = torch.cat(tuple(all_prompts[n]))
                x = torch.cat(tuple(torch.stack(list(all_seeds[n]), dim=0)))

                if job_info:
                    job_info.job_status = f"Processing Iteration {n+1}/{n_iter}. Batch size {batch_size}"

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
                cur_variant_amount = variant_amount 
                if variant_amount == 0.0:
                    # we manually generate all input noises because each one should have a specific seed
                    x = create_random_tensors(shape, seeds=seeds)
                else: # we are making variants
                    # using variant_seed as sneaky toggle,
                    # when not None or '' use the variant_seed
                    # otherwise use seeds
                    if variant_seed != None and variant_seed != '':
                        specified_variant_seed = seed_to_int(variant_seed)
                        torch.manual_seed(specified_variant_seed)
                        target_x = create_random_tensors(shape, seeds=[specified_variant_seed])
                        # with a variant seed we would end up with the same variant as the basic seed
                        # does not change. But we can increase the steps to get an interesting result
                        # that shows more and more deviation of the original image and let us adjust
                        # how far we will go (using 10 iterations with variation amount set to 0.02 will
                        # generate an icreasingly variated image which is very interesting for movies)
                        cur_variant_amount += n*variant_amount
                    else:
                        target_x = create_random_tensors(shape, seeds=seeds)
                    # finally, slerp base_x noise to target_x noise for creating a variant
                    original_shape = base_x.shape
                    x = slerp( max(0.0, min(1.0, cur_variant_amount)), base_x.flatten(), target_x.flatten()).reshape(*original_shape)

            samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)

            if opt.optimized:
                modelFS.to(device)

            x_samples_ddim = (model if not opt.optimized else modelFS).decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            for i, x_sample in enumerate(x_samples_ddim):
                if not do_interpolation:
                    sanitized_prompt = prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})
                    if variant_seed != None and variant_seed != '':
                        if variant_amount == 0.0:
                            seed_used = f"{current_seeds[i]}-{variant_seed}"
                        else:
                            seed_used = f"{seed}-{variant_seed}"
                    else:
                        seed_used = f"{current_seeds[i]}"
                    if sort_samples:
                        sanitized_prompt = sanitized_prompt[:128] #200 is too long
                        sample_path_i = os.path.join(sample_path, sanitized_prompt)
                        os.makedirs(sample_path_i, exist_ok=True)
                        base_count = get_next_sequence_number(sample_path_i)
                        filename = f"{base_count:05}-{steps}_{sampler_name}_{seed_used}_{cur_variant_amount:.2f}"
                    else:
                        sample_path_i = sample_path
                        base_count = get_next_sequence_number(sample_path_i)
                        sanitized_prompt = sanitized_prompt
                        filename = f"{base_count:05}-{steps}_{sampler_name}_{seed_used}_{cur_variant_amount:.2f}_{sanitized_prompt}"[:128] #same as before
                elif do_interpolation:
                    sample_path_i = outpath.strip('\\') + '/frames'
                    filename = f"{project_name}_{frame}"
                    frame += 1

                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)
                original_sample = x_sample
                original_filename = filename
                if use_GFPGAN and GFPGAN is not None and not use_RealESRGAN:
                    skip_save = True # #287 >_>
                    torch_gc()
                    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(original_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    gfpgan_sample = restored_img[:,:,::-1]
                    gfpgan_image = Image.fromarray(gfpgan_sample)
                    gfpgan_filename = original_filename + '-gfpgan'
                    save_sample(gfpgan_image, sample_path_i, gfpgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
normalize_prompt_weights, use_GFPGAN, write_info_files, write_sample_info_to_log_file, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, skip_metadata=True)
                    output_images.append(gfpgan_image) #287
                    #if simple_templating:
                    #    grid_captions.append( captions[i] + "\ngfpgan" )

                if use_RealESRGAN and RealESRGAN is not None and not use_GFPGAN:
                    skip_save = True # #287 >_>
                    torch_gc()
                    output, img_mode = RealESRGAN.enhance(original_sample[:,:,::-1])
                    esrgan_filename = original_filename + '-esrgan4x'
                    esrgan_sample = output[:,:,::-1]
                    esrgan_image = Image.fromarray(esrgan_sample)
                    save_sample(esrgan_image, sample_path_i, esrgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
normalize_prompt_weights, use_GFPGAN,write_info_files, write_sample_info_to_log_file, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, skip_metadata=True)
                    output_images.append(esrgan_image) #287
                    #if simple_templating:
                    #    grid_captions.append( captions[i] + "\nesrgan" )

                if use_RealESRGAN and RealESRGAN is not None and use_GFPGAN and GFPGAN is not None:
                    skip_save = True # #287 >_>
                    torch_gc()
                    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    gfpgan_sample = restored_img[:,:,::-1]
                    output, img_mode = RealESRGAN.enhance(gfpgan_sample[:,:,::-1])
                    gfpgan_esrgan_filename = original_filename + '-gfpgan-esrgan4x'
                    gfpgan_esrgan_sample = output[:,:,::-1]
                    gfpgan_esrgan_image = Image.fromarray(gfpgan_esrgan_sample)
                    save_sample(gfpgan_esrgan_image, sample_path_i, gfpgan_esrgan_filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
normalize_prompt_weights, use_GFPGAN, write_info_files, write_sample_info_to_log_file, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, skip_metadata=True)
                    output_images.append(gfpgan_esrgan_image) #287
                    #if simple_templating:
                    #    grid_captions.append( captions[i] + "\ngfpgan_esrgan" )

                image = Image.fromarray(x_sample)

                # this flag is used for imgProcessorTasks like GoBig, will return the image without saving it
                if imgProcessorTask == True:
                    output_images.append(image)

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
    use_GFPGAN, write_info_files, write_sample_info_to_log_file, prompt_matrix, init_info, uses_loopback, uses_random_seed_loopback, skip_save,
    skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, False)
                    if add_original_image or not simple_templating:
                            output_images.append(image)
                            if simple_templating:
                                grid_captions.append(captions[i])
                    else:
                        save_sample(image, sample_path_i, filename, jpg_sample, None, None, width, height, steps, cfg_scale, 
    use_GFPGAN, write_info_files, False, False, None, False, False, skip_save,
    skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode, False)

                output_images.append(image)
                if do_interpolation:
                    yield image, f"Frame: {n+1}/{n_iter}\nDirectory: {sample_path_i}" if n+1 != n_iter else f"Completed! Frames are available at {sample_path_i}"
                if do_interpolation and not skip_grid:
                    video_out.append_data(x_sample)

        if (prompt_matrix or not skip_grid) and not do_not_save_grid and not do_interpolation:
            if prompt_matrix:
                if simple_templating:
                    grid = image_grid(output_images, batch_size, force_n_rows=frows, captions=grid_captions)
                else:
                    grid = image_grid(output_images, batch_size, force_n_rows=1 << ((len(prompt_matrix_parts)-1)//2))
                    try:
                        grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                    except:
                        import traceback
                        print("Error creating prompt_matrix text:", file=sys.stderr)
                        print(traceback.format_exc(), file=sys.stderr)
            elif batch_size > 1  or n_iter > 1:
                grid = image_grid(output_images, batch_size)

            if prompt_matrix and not prompt.startswith("@") and not do_interpolation:
                try:
                    grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                except:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

            else:
               grid = image_grid(output_images, batch_size)

            output_images.insert(0, grid)

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

    args_and_names = {
        "seed": seed if not do_interpolation else '',
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler": sampler_name,
    }

    full_string = f"{prompt if not do_interpolation else ''}\n"+ " ".join([f"{k}:" for k,v in args_and_names.items()])
    info = {
        'text': full_string,
        'entities': [{'entity':str(v), 'start': full_string.find(f"{k}:"),'end': full_string.find(f"{k}:") + len(f"{k} ")} for k,v in args_and_names.items()]
     }
    stats = f'''
Took { round(time_diff, 2) }s total ({ round(time_diff/(len(all_prompts)),2) }s per image)
Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

    for comment in comments:
        info['text'] += "\n\n" + comment

    #mem_mon.stop()
    #del mem_mon
    torch_gc()

    if not do_interpolation:
        return output_images, seed, info, stats


def process_disco_anim(outpath, func_init, func_sample, init_image, prompts, seed, sampler_name, animation_mode, start_frame, max_frames,
                       unconditional_guidance_scale, width, height, resume_run, angle_series, zoom_series, translation_x_series,
                       translation_y_series, translation_z_series, rotation_3d_x_series, rotation_3d_y_series, rotation_3d_z_series,
                       color_match, noise_between_frames, turbo_mode, turbo_preroll, turbo_steps, vr_mode, video_init_frames_scale,
                       video_init_flow_warp, videoFramesFolder, flo_folder, video_init_flow_blend, consistent_seed, color_match_mode,
                       interpolate, vr_eye_angle, vr_ipd, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode):
    global stop_on_next_loop

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
                        near_plane, far_plane, fov, padding_mode, sampling_mode, torch.device('cuda'), # using device variable throws error - investigate
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
                                near_plane, far_plane, fov, padding_mode, sampling_mode, torch.device('cuda'),
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
                                    vr_eye_angle, vr_ipd, torch.device('cuda'), near_plane, far_plane, fov, padding_mode, sampling_mode)
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
                                        vr_eye_angle, vr_ipd, torch.device('cuda'), near_plane, far_plane, fov, padding_mode, sampling_mode)

                    if opt.optimized:
                        mem = torch.cuda.memory_allocated()/1e6
                        modelFS.to("cpu")
                        while(torch.cuda.memory_allocated()/1e6 >= mem):
                            time.sleep(1)
                    yield img, f"Frame: {frame_num}/{max_frames}\nDirectory: {outpath}" if frame_num != max_frames else f"Completed! Frames are available at {outpath}"


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

    def init(init_info):
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        callback = make_callback(sampler_name, dynamic_threshold=dynamic_threshold, static_threshold=static_threshold)
        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale, unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x, img_callback=callback)
        return samples_ddim

    try:
        yield from process_images(
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

    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return []
    finally:
        if err:
            crash(err, '!!Runtime error (txt_interp)!!')


def txt2img(prompt: str, ddim_steps: int, sampler_name: str, toggles: List[int], realesrgan_model_name: str,
            ddim_eta: float, n_iter: int, batch_size: int, cfg_scale: float, dynamic_threshold: float, 
            static_threshold: float, seed: Union[int, str, None], variant_amount: float, variant_seed: int,
            height: int, width: int, fp, job_info: JobInfo = None):
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
    write_to_one_file = 5 in toggles
    jpg_sample = 6 in toggles
    use_GFPGAN = 7 in toggles
    use_RealESRGAN = 8 in toggles
    ModelLoader(['model'],True,False)
    if use_GFPGAN and not use_RealESRGAN:
        ModelLoader(['GFPGAN'],True,False)
        ModelLoader(['RealESRGAN'],False,True)
    if use_RealESRGAN and not use_GFPGAN:
        ModelLoader(['GFPGAN'],False,True)
        ModelLoader(['RealESRGAN'],True,False,realesrgan_model_name)
    if use_RealESRGAN and use_GFPGAN:
        ModelLoader(['GFPGAN','RealESRGAN'],True,False,realesrgan_model_name)

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

    def init(init_info):
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        callback = make_callback(sampler_name, dynamic_threshold=dynamic_threshold, static_threshold=static_threshold)
        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale, unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x, img_callback=callback)
        return samples_ddim

    try:
        output_images, seed, info, stats = _process_images(
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
            write_sample_info_to_log_file=write_to_one_file,
            jpg_sample=jpg_sample,
            variant_amount=variant_amount,
            variant_seed=variant_seed,
            job_info=job_info
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
                x0 = init_data

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
        yield from process_disco_anim(
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

        # del sampler

        # return output_images #, seed, info, stats
    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return []
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
        prompt, ddim_steps, sampler_name, toggles, ddim_eta, n_iter, batch_size, cfg_scale, seed, height, width, fp, variant_amount, variant_seed, images, seed, comment, stats = flag_data

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


def img2img(prompt: str, image_editor_mode: str, init_info: any, init_info_mask: any, mask_mode: str, mask_blur_strength: int, ddim_steps: int, sampler_name: str,
            toggles: List[int], realesrgan_model_name: str, n_iter: int, cfg_scale: float, denoising_strength: float,
            dynamic_threshold: float, static_threshold: float, seed: int, height: int, width: int, resize_mode: int, fp, job_info: JobInfo = None):

    outpath = opt.outdir_img2img or opt.outdir or "outputs/img2img-samples"
    err = False
    seed = seed_to_int(seed)

    batch_size = 1
    width, height = map(lambda x: x - x % 32, (width, height))  # resize to integer multiple of 32

    prompt_matrix = 0 in toggles
    loopback = 1 in toggles
    random_seed_loopback = 2 in toggles
    skip_save = 3 not in toggles
    skip_grid = 4 not in toggles
    sort_samples = 5 in toggles
    write_info_files = 6 in toggles
    write_sample_info_to_log_file = 7 in toggles
    jpg_sample = 8 in toggles
    use_GFPGAN = 9 in toggles
    use_RealESRGAN = 10 in toggles
    ModelLoader(['model'],True,False)
    if use_GFPGAN and not use_RealESRGAN:
        ModelLoader(['GFPGAN'],True,False)
        ModelLoader(['RealESRGAN'],False,True)
    if use_RealESRGAN and not use_GFPGAN:
        ModelLoader(['GFPGAN'],False,True)
        ModelLoader(['RealESRGAN'],True,False,realesrgan_model_name)
    if use_RealESRGAN and use_GFPGAN:
        ModelLoader(['GFPGAN','RealESRGAN'],True,False,realesrgan_model_name)

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

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(denoising_strength * ddim_steps)

    if image_editor_mode == 'Mask':
        init_info = init_info_mask

    def init(init_img_):
        if init_img_ != None:
            init_img = init_img_
        else:
            if isinstance(init_info, Image.Image):
                init_img = init_info
            elif isinstance(init_info, str):
                init_img = Image.open(init_info)
            else:
                init_img = init_info["image"]
        image = init_img.convert("RGB")
        image = resize_image(resize_mode, image, width, height)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        mask_channel = None
        if image_editor_mode == "Uncrop":
            alpha = init_img.convert("RGB")
            alpha = resize_image(resize_mode, alpha, width // 8, height // 8)
            mask_channel = alpha.split()[-1]
            mask_channel = mask_channel.filter(ImageFilter.GaussianBlur(4))
            mask_channel = np.array(mask_channel)
            mask_channel[mask_channel >= 255] = 255
            mask_channel[mask_channel < 255] = 0
            mask_channel = Image.fromarray(mask_channel).filter(ImageFilter.GaussianBlur(2))

        init_mask = None
        if mask_channel is not None:
            init_mask = np.array(mask_channel).astype(np.float32) / 255.0
            init_mask = (1 - init_mask)
            init_mask = np.tile(init_mask, (4, 1, 1))
            init_mask = init_mask[None].transpose(0, 1, 2, 3)
            init_mask = torch.from_numpy(init_mask).to(device)
        if opt.optimized:
            modelFS.to(device)

        init_image = 2. * image - 1.
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = (model if not opt.optimized else modelFS).get_first_stage_encoding((model if not opt.optimized else modelFS).encode_first_stage(init_image))  # move to latent space

        if image_editor_mode == 'Mask':
            init_mask = init_info["mask"]
            init_mask = init_mask.convert("L")
            init_mask = init_mask.filter(ImageFilter.GaussianBlur(mask_blur_strength))
            init_mask = resize_image(resize_mode, init_mask, width//opt_f, height//opt_f)
            keep_mask = mask_mode == 0
            # init_mask = init_mask.convert("RGB")
            init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
            init_mask = np.array(init_mask).astype(np.float32) / 255.0
            init_mask = init_mask[None,None]
            init_mask = torch.from_numpy(init_mask).to(device)
        if opt.optimized:
            mem = torch.cuda.memory_allocated()/1e6
            modelFS.to("cpu")
            while(torch.cuda.memory_allocated()/1e6 >= mem):
                time.sleep(1)

        return init_latent, init_mask

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        t_enc_steps = t_enc
        obliterate = False
        if ddim_steps == t_enc_steps:
            t_enc_steps = t_enc_steps - 1
            obliterate = True

        if sampler_name != 'DDIM':
            x0, init_mask = init_data

            inpainting = True if init_mask is not None else False

            sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
            noise = x * sigmas[ddim_steps - t_enc_steps - 1]

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

    init_info_ = None
    try:
        if loopback:
            output_images, info = None, None
            history = []
            initial_seed = None

            for i in range(n_iter):
                output_images, seed, info, stats = _process_images(
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
                    init_info=init_info_,
                    denoising_strength=denoising_strength,
                    resize_mode=resize_mode,
                    uses_loopback=loopback,
                    uses_random_seed_loopback=random_seed_loopback,
                    sort_samples=sort_samples,
                    write_info_files=write_info_files,
                    write_sample_info_to_log_file=write_sample_info_to_log_file,
                    jpg_sample=jpg_sample,
                    job_info=job_info
                )

                if initial_seed is None:
                    initial_seed = seed

                init_info_ = output_images[0]
                if not random_seed_loopback:
                    seed = seed + 1
                else:
                    seed = seed_to_int(None)
                denoising_strength = max(denoising_strength * 0.95, 0.1)
                history.append(init_info_)
                yield output_images, seed, info, stats

            if not skip_grid:
                grid_count = get_next_sequence_number(outpath, 'grid-')
                grid = image_grid(history, batch_size, force_n_rows=1)
                grid_file = f"grid-{grid_count:05}-{seed}_{prompt.replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.{grid_ext}"
                grid.save(os.path.join(outpath, grid_file), grid_format, quality=grid_quality, lossless=grid_lossless, optimize=True)

            output_images = history
            seed = initial_seed

        else:
            output_images, seed, info, stats = _process_images(
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
                init_info=init_info,
                denoising_strength=denoising_strength,
                resize_mode=resize_mode,
                uses_loopback=loopback,
                sort_samples=sort_samples,
                write_info_files=write_info_files,
                write_sample_info_to_log_file=write_sample_info_to_log_file,
                jpg_sample=jpg_sample,
                job_info=job_info
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


prompt_parser = re.compile("""
    (?P<prompt>     # capture group for 'prompt'
    (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
    )               # end 'prompt'
    (?:             # non-capture group
    :+              # match one or more ':' characters
    (?P<weight>     # capture group for 'weight'
    -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
    )?              # end weight capture group, make optional
    \s*             # strip spaces after weight
    |               # OR
    $               # else, if no ':' then match end of line
    )               # end non-capture group
""", re.VERBOSE)

# grabs all text up to the first occurrence of ':' as sub-prompt
# takes the value following ':' as weight
# if ':' has no value defined, defaults to 0.5
# repeats until no text remaining
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


def imgproc(image,image_batch,imgproc_prompt,imgproc_toggles, imgproc_upscale_toggles,imgproc_realesrgan_model_name,imgproc_sampling,
 imgproc_steps, imgproc_height, imgproc_width, imgproc_cfg, imgproc_denoising, imgproc_seed,imgproc_gfpgan_strength,imgproc_ldsr_steps,imgproc_ldsr_pre_downSample,imgproc_ldsr_post_downSample):

    outpath = opt.outdir_imglab or opt.outdir or "outputs/imglab-samples"
    output = []
    images = []
    def processGFPGAN(image,strength):
        image = image.convert("RGB")
        cropped_faces, restored_faces, restored_img = GFPGAN.enhance(np.array(image, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)
        result = Image.fromarray(restored_img)
        if strength < 1.0:
            result = Image.blend(image, result, strength)

        return result
    def processRealESRGAN(image):
        if 'x2' in imgproc_realesrgan_model_name:
            # downscale to 1/2 size
            modelMode = imgproc_realesrgan_model_name.replace('x2','x4')
        else:
            modelMode = imgproc_realesrgan_model_name
        image = image.convert("RGB")
        RealESRGAN = load_RealESRGAN(modelMode)
        result, res = RealESRGAN.enhance(np.array(image, dtype=np.uint8))
        result = Image.fromarray(result)
        if 'x2' in imgproc_realesrgan_model_name:
            # downscale to 1/2 size
            result = result.resize((result.width//2, result.height//2), LANCZOS)

        return result
    def processGoBig(image):
        result = processRealESRGAN(image,)
        if 'x4' in imgproc_realesrgan_model_name:
            #downscale to 1/2 size
            result = result.resize((result.width//2, result.height//2), LANCZOS)



        #make sense of parameters
        n_iter = 1
        batch_size = 1
        seed = seed_to_int(imgproc_seed)
        ddim_steps = int(imgproc_steps)
        resize_mode = 0 #need to add resize mode to form, or infer correct resolution from file name
        width = int(imgproc_width)
        height = int(imgproc_height)
        cfg_scale = float(imgproc_cfg)
        denoising_strength = float(imgproc_denoising)
        skip_save = True
        skip_grid = True
        prompt = imgproc_prompt
        t_enc = int(denoising_strength * ddim_steps)
        sampler_name = imgproc_sampling


        if sampler_name == 'DDIM':
            sampler = DDIMSampler(model)
        elif sampler_name == 'k_dpm_2_a':
            sampler = KDiffusionSampler(model,'dpm_2_ancestral')
        elif sampler_name == 'k_dpm_2':
            sampler_name = KDiffusionSampler(model,'dpm_2')
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
            pass
        init_img = result
        init_mask = None
        keep_mask = False
        assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

        def init(init_info):
            if init_info != None:
                image = init_info
                image = image.convert("RGB")
            else:
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

                xi = x0 + noise
                sigma_sched = sigmas[ddim_steps - t_enc - 1:]
                model_wrap_cfg = CFGDenoiser(sampler.model_wrap)
                samples_ddim = K.sampling.__dict__[f'sample_{sampler.get_sampler_name()}'](model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': cfg_scale}, disable=False)
            else:
                x0, = init_data
                sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
                z_enc = sampler.stochastic_encode(x0, torch.tensor([t_enc]*batch_size).to(device))
                                    # decode it
                samples_ddim = sampler.decode(z_enc, conditioning, t_enc,
                                                unconditional_guidance_scale=cfg_scale,
                                                unconditional_conditioning=unconditional_conditioning,)
            return samples_ddim
        def split_grid(image, tile_w=512, tile_h=512, overlap=64):
            Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])
            w = image.width
            h = image.height

            now = tile_w - overlap  # non-overlap width
            noh = tile_h - overlap

            cols = math.ceil((w - overlap) / now)
            rows = math.ceil((h - overlap) / noh)

            grid = Grid([], tile_w, tile_h, w, h, overlap)
            for row in range(rows):
                row_images = []

                y = row * noh

                if y + tile_h >= h:
                    y = h - tile_h

                for col in range(cols):
                    x = col * now

                    if x+tile_w >= w:
                        x = w - tile_w

                    tile = image.crop((x, y, x + tile_w, y + tile_h))

                    row_images.append([x, tile_w, tile])

                grid.tiles.append([y, tile_h, row_images])

            return grid


        def combine_grid(grid):
            def make_mask_image(r):
                r = r * 255 / grid.overlap
                r = r.astype(np.uint8)
                return Image.fromarray(r, 'L')

            mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
            mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

            combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
            for y, h, row in grid.tiles:
                combined_row = Image.new("RGB", (grid.image_w, h))
                for x, w, tile in row:
                    if x == 0:
                        combined_row.paste(tile, (0, 0))
                        continue

                    combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
                    combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

                if y == 0:
                    combined_image.paste(combined_row, (0, 0))
                    continue

                combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
                combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

            return combined_image

        grid = split_grid(result, tile_w=width, tile_h=height, overlap=64)
        work = []
        work_results = []

        for y, h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])
        batch_count = math.ceil(len(work) / batch_size)
        print(f"GoBig upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} in a total of {batch_count} batches.")
        for i in range(batch_count):
            init_img = work[i*batch_size:(i+1)*batch_size][0]
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
                    prompt_matrix=None,
                    use_GFPGAN=None,
                    use_RealESRGAN=None,
                    realesrgan_model_name=None,
                    fp=None,
                    normalize_prompt_weights=False,
                    init_info=init_img,
                    mask_blur_strength=None,
                    denoising_strength=denoising_strength,
                    resize_mode=resize_mode,
                    uses_loopback=False,
                    sort_samples=True,
                    write_info_files=True,
                    write_sample_info_to_log_file=False,
                    jpg_sample=False,
                    imgProcessorTask=True
                )
            #if initial_seed is None:
            #    initial_seed = seed
            #seed = seed + 1

            work_results.append(output_images[0])
        image_index = 0
        for y, h, row in grid.tiles:
            for tiledata in row:
                tiledata[2] = work_results[image_index]
                image_index += 1

        combined_image = combine_grid(grid)
        grid_count = len(os.listdir(outpath)) - 1
        del sampler

        torch.cuda.empty_cache()
        return combined_image
    def processLDSR(image):
        result = LDSR.superResolution(image,int(imgproc_ldsr_steps),str(imgproc_ldsr_pre_downSample),str(imgproc_ldsr_post_downSample))
        return result   
    

    if image_batch != None:
        if image != None:
            print("Batch detected and single image detected, please only use one of the two. Aborting.")
            return None
        #convert file to pillow image
        for img in image_batch:
            image = Image.fromarray(np.array(Image.open(img)))
            images.append(image)

    elif image != None:
        if image_batch != None:
            print("Batch detected and single image detected, please only use one of the two. Aborting.")
            return None
        else:
            images.append(image)

    if len(images) > 0:
        print("Processing images...")
        #pre load models not in loop
        if 0 in imgproc_toggles:
            ModelLoader(['RealESGAN','LDSR'],False,True) # Unload unused models
            ModelLoader(['GFPGAN'],True,False) # Load used models
        if 1 in imgproc_toggles:
                if imgproc_upscale_toggles == 0:
                     ModelLoader(['GFPGAN','LDSR'],False,True) # Unload unused models
                     ModelLoader(['RealESGAN'],True,False,imgproc_realesrgan_model_name) # Load used models 
                elif imgproc_upscale_toggles == 1:
                        ModelLoader(['GFPGAN','LDSR'],False,True) # Unload unused models
                        ModelLoader(['RealESGAN','model'],True,False) # Load used models
                elif imgproc_upscale_toggles == 2:

                    ModelLoader(['model','GFPGAN','RealESGAN'],False,True) # Unload unused models
                    ModelLoader(['LDSR'],True,False) # Load used models
                elif imgproc_upscale_toggles == 3:
                    ModelLoader(['GFPGAN','LDSR'],False,True) # Unload unused models
                    ModelLoader(['RealESGAN','model'],True,False,imgproc_realesrgan_model_name) # Load used models
        for image in images:
            if 0 in imgproc_toggles:
                #recheck if GFPGAN is loaded since it's the only model that can be loaded in the loop as well
                ModelLoader(['GFPGAN'],True,False) # Load used models
                image = processGFPGAN(image,imgproc_gfpgan_strength)
                outpathDir = os.path.join(outpath,'GFPGAN')
                os.makedirs(outpathDir, exist_ok=True)
                batchNumber = get_next_sequence_number(outpathDir)
                outFilename = str(batchNumber)+'-'+'result'

                if 1 not in imgproc_toggles:
                    output.append(image)
                    save_sample(image, outpathDir, outFilename, False, None, None, None, None, None, None, None, None, None, None, None, None, None, False, None, None, None, None, None, None, None, None, None, True)
            if 1 in imgproc_toggles:
                if imgproc_upscale_toggles == 0:
                    image = processRealESRGAN(image)
                    outpathDir = os.path.join(outpath,'RealESRGAN')
                    os.makedirs(outpathDir, exist_ok=True)
                    batchNumber = get_next_sequence_number(outpathDir)
                    outFilename = str(batchNumber)+'-'+'result'
                    output.append(image)
                    save_sample(image, outpathDir, outFilename, False, None, None, None, None, None, None, None, None, None, None, None, None, None, False, None, None, None, None, None, None, None, None, None, True)

                elif imgproc_upscale_toggles == 1:
                    image = processGoBig(image)
                    outpathDir = os.path.join(outpath,'GoBig')
                    os.makedirs(outpathDir, exist_ok=True)
                    batchNumber = get_next_sequence_number(outpathDir)
                    outFilename = str(batchNumber)+'-'+'result'
                    output.append(image)
                    save_sample(image, outpathDir, outFilename, False, None, None, None, None, None, None, None, None, None, None, None, None, None, False, None, None, None, None, None, None, None, None, None, True)

                elif imgproc_upscale_toggles == 2:
                    image = processLDSR(image)
                    outpathDir = os.path.join(outpath,'LDSR')
                    os.makedirs(outpathDir, exist_ok=True)
                    batchNumber = get_next_sequence_number(outpathDir)
                    outFilename = str(batchNumber)+'-'+'result'
                    output.append(image)
                    save_sample(image, outpathDir, outFilename, False, None, None, None, None, None, None, None, None, None, None, None, None, None, False, None, None, None, None, None, None, None, None, None, True)

                elif imgproc_upscale_toggles == 3:
                    image = processGoBig(image)
                    ModelLoader(['model','GFPGAN','RealESGAN'],False,True) # Unload unused models
                    ModelLoader(['LDSR'],True,False) # Load used models
                    image = processLDSR(image)
                    outpathDir = os.path.join(outpath,'GoLatent')
                    os.makedirs(outpathDir, exist_ok=True)
                    batchNumber = get_next_sequence_number(outpathDir)
                    outFilename = str(batchNumber)+'-'+'result'
                    output.append(image)

                    save_sample(image, outpathDir, outFilename, None, None, None, None, None, None, None, None, None, None, None, None, None, None, False, None, None, None, None, None, None, None, None, None, True)

    #LDSR is always unloaded to avoid memory issues
    #ModelLoader(['LDSR'],False,True)
    #print("Reloading default models...")
    #ModelLoader(['model','RealESGAN','GFPGAN'],True,False) # load back models
    print("Done.")
    return output

def ModelLoader(models,load=False,unload=False,imgproc_realesrgan_model_name='RealESRGAN_x4plus'):
    #get global variables
    global_vars = globals()
    #check if m is in globals
    if unload:
        for m in models:
            if m in global_vars:
                #if it is, delete it
                del global_vars[m]
                if opt.optimized:
                    if m == 'model':
                        del global_vars[m+'FS']
                        del global_vars[m+'CS']
                if m =='model':
                    m='Stable Diffusion'
                print('Unloaded ' + m)
    if load:
        for m in models:
            if m not in global_vars or m in global_vars and type(global_vars[m]) == bool:
                #if it isn't, load it
                if m == 'GFPGAN':
                    global_vars[m] = load_GFPGAN()
                elif m == 'model':
                    sdLoader = load_SD_model()
                    global_vars[m] = sdLoader[0]
                    if opt.optimized:
                        global_vars[m+'CS'] = sdLoader[1]
                        global_vars[m+'FS'] = sdLoader[2]
                elif m == 'RealESRGAN':
                    global_vars[m] = load_RealESRGAN(imgproc_realesrgan_model_name)
                elif m == 'LDSR':
                    global_vars[m] = load_LDSR()
                if m =='model':
                    m='Stable Diffusion'
                print('Loaded ' + m)
    torch_gc()

def run_GFPGAN(image, strength):
    ModelLoader(['LDSR','RealESRGAN'],False,True)
    ModelLoader(['GFPGAN'],True,False)
    image = image.convert("RGB")

    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(np.array(image, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)
    res = Image.fromarray(restored_img)

    if strength < 1.0:
        res = Image.blend(image, res, strength)

    return res

def run_RealESRGAN(image, model_name: str):
    ModelLoader(['GFPGAN','LDSR'],False,True)
    ModelLoader(['RealESRGAN'],True,False)
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
    'write sample info to log file',
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
    'variant_amount': 0.0,
    'variant_seed': '',
    'submit_on_enter': 'Yes',
}

if 'txt2img' in user_defaults:
    txt2img_defaults.update(user_defaults['txt2img'])

txt2img_toggle_defaults = [txt2img_toggles[i] for i in txt2img_defaults['toggles']]

# make sure these indicies line up at the top of txt_interp()
txt_interp_toggles = [
    'Loop Interpolation',
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

imgproc_defaults = {
    'prompt': '',
    'ddim_steps': 50,
    'sampler_name': 'k_lms',
    'cfg_scale': 7.5,
    'seed': '',
    'height': 512,
    'width': 512,
    'denoising_strength': 0.30
}
imgproc_mode_toggles = [
    'Fix Faces',
    'Upscale'
]

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
    'Write sample info one file',
    'jpg samples',
]
# removed for now becuase of Image Lab implementation
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
    'toggles': [3, 4],
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
    'mix_factor': "0: (0.15), 10: (1.0)",
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


def copy_img_to_input(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.update(selected='img2img_tab')
        img_update = gr.update(value=processed_image)
        return {img2img_image_mask: processed_image, img2img_image_editor: img_update, tabs: tab_update}
    except IndexError:
        return [None, None]


def copy_img_to_upscale_esrgan(img):
    update = gr.update(selected='realesrgan_tab')
    image_data = re.sub('^data:image/.+;base64,', '', img)
    processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return {realesrgan_source: processed_image, tabs: update}


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

demo = draw_gradio_ui(opt,
                      user_defaults=user_defaults,
                      txt2img=txt2img,
                      img2img=img2img,
                      txt_interp=txt_interp,
                      disco_anim=disco_anim,
                      imgproc=imgproc,
                      txt_interp_defaults=txt_interp_defaults,
                      txt_interp_toggles=txt_interp_toggles,
                      txt_interp_toggle_defaults=txt_interp_toggle_defaults,
                      disco_anim_defaults=disco_anim_defaults,
                      disco_anim_toggles=disco_anim_toggles,
                      disco_anim_toggle_defaults=disco_anim_toggle_defaults,
                      txt2img_defaults=txt2img_defaults,
                      txt2img_toggles=txt2img_toggles,
                      txt2img_toggle_defaults=txt2img_toggle_defaults,
                      show_embeddings=hasattr(model, "embedding_manager"),
                      img2img_defaults=img2img_defaults,
                      img2img_toggles=img2img_toggles,
                      img2img_toggle_defaults=img2img_toggle_defaults,
                      img2img_mask_modes=img2img_mask_modes,
                      img2img_resize_modes=img2img_resize_modes,
                      sample_img2img=sample_img2img,
                      imgproc_defaults=imgproc_defaults,
                      imgproc_mode_toggles=imgproc_mode_toggles,
                      RealESRGAN=RealESRGAN,
                      GFPGAN=GFPGAN,
                      LDSR=LDSR,
                      run_GFPGAN=run_GFPGAN,
                      run_RealESRGAN=run_RealESRGAN,
                      stop_anim=stop_anim,
                      job_manager=job_manager
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
            'inbrowser': opt.inbrowser,
            'server_name': '0.0.0.0',
            'server_port': opt.port,
            'share': opt.share, 
            'show_error': True
        }
        if not opt.share:
            demo.queue(concurrency_count=opt.max_jobs)
        if opt.share and opt.share_password:
            gradio_params['auth'] = ('webui', opt.share_password)

        # Check to see if Port 7860 is open
        port_status = 1
        while port_status != 0:
            try:
                self.demo.launch(**gradio_params)
            except (OSError) as e:
                print (f'Error: Port: {opt.port} is not open yet. Please wait, this may take upwards of 60 seconds...')
                time.sleep(10)
            else:
                port_status = 0

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
