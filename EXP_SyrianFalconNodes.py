
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import PIL
import torch.nn.functional as F

import os
import torch
import sys
import json
import hashlib
import traceback
import time
import re
import glob
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch
import random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils

import comfy.clip_vision

import comfy.model_management
import importlib

import folder_paths
import latent_preview

import math
import struct


def alternatingtokenfunc(prompt):
    newprompt = prompt
    occurslist = []
    while len(newprompt)>0:
        occurs = re.search("\<[^<]+\|[^<]+\>",newprompt)
        if occurs:
            oldoccurs = re.search(re.escape(newprompt[occurs.span()[0]:occurs.span()[1]]),prompt)
            occurslist.append([newprompt[occurs.span()[0]:occurs.span()[1]],oldoccurs.span()])
            newprompt = newprompt[occurs.span()[1]+1:]
        else:
            newprompt = ""
    for i in occurslist:
        i.append(i[0][1:-1].split("|"))
        i.append(0)
    for i in occurslist:
        if len(i[0])<=1:
            occurslist.remove(i)
    return occurslist


def promptscreator(text,steps):
    prompt = text
    checkpoints = []
    replaces = {}
    additions = {}
    removes = {}
    initials = []
    numloop = []
    total = steps
    prompts = []
    occs = re.findall("\[[^\[]*]",prompt)
    #for i in occs:
        #checkpoints.append(re.search(occs[1][1:-1],prompt).span())
    for idx,i in enumerate(occs):
        #c = int(re.findall("[\d][^\]]*",i)[0][:]) if float(re.findall("[\d][^\]]*",i)[0][:])>=1 else int(float(re.findall("[\d][^\]]*",i)[0][:])*total)
        if len(re.findall(":",i)) > 1:
            if re.search("::",i):
                c = int(i[re.search("::",i).span()[1]:-1]) if float(i[re.search("::",i).span()[1]:-1])>=1 else int(float(i[re.search("::",i).span()[1]:-1])*total)
                if c not in numloop:
                    numloop.append(c)
                removes[c] = [i,re.findall(".*:",i)[0][1:-2],re.findall(".*:",i)[0][1:-2]]
                initials.append([i,re.findall(".*:",i)[0][1:-2]])
            else:
                i2 = i[re.search(":",i).span()[0]+1:]   
                     
                c = int(i2[re.search(":",i2).span()[0]+1:-1]) if float(i2[re.search(":",i2).span()[0]+1:-1])>=1 else int(float(i2[re.search(":",i2).span()[0]+1:-1])*total)
                if c not in numloop:
                    numloop.append(c)
                   
                replaces[c] = [i,re.findall(":.*:",i)[0][1:-1],re.findall("[^:]*:",i)[0][1:-1]]
                initials.append([i,re.findall("[^:]*:",i)[0][1:-1]])
        elif len(re.findall("\|",i)) >= 1:
            continue
        else:
            c = int(i[re.search(":",i).span()[0]+1:-1]) if float(i[re.search(":",i).span()[0]+1:-1])>=1 else int(float(i[re.search(":",i).span()[0]+1:-1])*total)
            if c not in numloop:
                numloop.append(c)
       
            additions[c] = [i,re.findall(".*:",i)[0][1:-1],re.findall(".*:",i)[0][1:-1],re.search(i[1:-1],prompt).span()[0]]
            #initials.append([i,""])


  
    numloop.sort()

    for i in initials:
        prompt = re.sub("."+re.escape(i[0][1:-1])+".",i[1],prompt,1)

    prompts.append(prompt)
    for x in numloop:
        if x in replaces:
            prompt = re.sub(re.escape(replaces[x][2]),replaces[x][1],prompt,1)
            
        if x in removes:
            prompt = re.sub(removes[x][2],"",prompt,1)
        prompts.append(prompt)
    if len(prompts)==0:
        prompts.append(prompt)
        #print(prompts)
        numloop.append(steps-1)
    keys = []
    for j in additions:
        keys.append(j)
        keys.sort()
        print(keys)
    for a in keys:
        prompts[0] = re.sub(re.escape(additions[a][0]),"",prompts[0])
        for idx,j in enumerate(numloop):
            if a>j:
                prompts[idx+1] = re.sub(re.escape(additions[a][0]),"",prompts[idx+1])
            else:
                prompts[idx+1] = re.sub(re.escape(additions[a][0]),additions[a][2],prompts[idx+1])
    return prompts,numloop

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False,flag=False):
    
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = latent_preview.get_previewer(device)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback)
    out = latent.copy()
    out["samples"] = samples
    if not flag:
        return out
    else: 
        return (out, )
    	

        
        
class KSamplerPromptEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "control_net": ("CONTROL_NET", ),
                    "image": ("IMAGE", ),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "return_with_leftover_noise": (["disable", "enable"], ),"clip": ("CLIP", ),"text": ("STRING", {"multiline": True})
                    
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "SyrianFalcon/nodes"
    def sample(self, clip,model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, text, return_with_leftover_noise,control_net,image,strength, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        prompts,numloop = promptscreator(text,steps)
        if len(prompts) ==0:
            prompts.append(text)
        if len(numloop) < 1:
            numloop.append(steps-1)
        positive = ([[clip.encode(prompts[0]), {}]] )
        try:
            if strength == 0:
                pass
            else:
                c = []
                control_hint = image.movedim(-1,1)
                for t in positive:
                    n = [t[0], t[1].copy()]
                    c_net = control_net.copy().set_cond_hint(control_hint, strength)
                    if 'control' in t[1]:
                        c_net.set_previous_controlnet(t[1]['control'])
                    n[1]['control'] = c_net
                    c.append(n)
                positive = c
        except:
            pass
        c1 = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=0, last_step=numloop[0], force_full_denoise=force_full_denoise)

        for i in range(0,len(numloop)-1):
            positive = ([[clip.encode(prompts[i]), {}]] )
            try:
                if strength == 0:
                    pass
                else:
                    c = []
                    control_hint = image.movedim(-1,1)
                    for t in positive:
                        n = [t[0], t[1].copy()]
                        c_net = control_net.copy().set_cond_hint(control_hint, strength)
                        if 'control' in t[1]:
                            c_net.set_previous_controlnet(t[1]['control'])
                        n[1]['control'] = c_net
                        c.append(n)
                    positive = c
            except:
                pass
            c1 = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, c1, denoise=denoise, disable_noise=disable_noise, start_step=numloop[i], last_step=numloop[i+1], force_full_denoise=force_full_denoise)
        positive = ([[clip.encode(prompts[len(prompts)-1]), {}]] )
        try:
            if strength == 0:
                pass
            else:
                c = []
                control_hint = image.movedim(-1,1)
                for t in positive:
                    n = [t[0], t[1].copy()]
                    c_net = control_net.copy().set_cond_hint(control_hint, strength)
                    if 'control' in t[1]:
                        c_net.set_previous_controlnet(t[1]['control'])
                    n[1]['control'] = c_net
                    c.append(n)
                positive = c
        except:
            pass
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, c1, denoise=denoise, disable_noise=disable_noise, start_step=numloop[len(numloop)-1], last_step=steps, force_full_denoise=force_full_denoise,flag=True)
        
        
class KSamplerAlternate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "control_net": ("CONTROL_NET", ),
                    "image": ("IMAGE", ),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "return_with_leftover_noise": (["disable", "enable"], ),"clip": ("CLIP", ),"text": ("STRING", {"multiline": True})
                    
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "SyrianFalcon/nodes"
    def sample(self, clip,model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, text, return_with_leftover_noise,control_net,image,strength, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        arr = alternatingtokenfunc(text)
        newprompt = text
        for j in arr:
            if j[3]>=len(j[2]):
                j[3] = 0
            newerprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
            newprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
            j[3]+=1
        positive = ([[clip.encode(newerprompt), {}]] )
        c1 = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=0, last_step=1, force_full_denoise=force_full_denoise)
        for i in range(0,steps-1):
            newprompt = text
            for j in arr:
                if j[3]>=len(j[2]):
                    j[3] = 0
                newerprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
                newprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
                j[3]+=1
            positive = ([[clip.encode(newerprompt), {}]] )
            c1 = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, c1, denoise=denoise, disable_noise=disable_noise, start_step=i, last_step=i+1, force_full_denoise=force_full_denoise)
        newprompt = text
        for j in arr:
            if j[3]>=len(j[2]):
                j[3] = 0
            newerprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
            newprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
            j[3]+=1
        positive = ([[clip.encode(newerprompt), {}]] )
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, c1, denoise=denoise, disable_noise=disable_noise, start_step=steps-1, last_step=steps, force_full_denoise=force_full_denoise,flag=True)


class KSamplerPromptEditAndAlternate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "control_net": ("CONTROL_NET", ),
                    "image": ("IMAGE", ),
                    "ControlNet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "return_with_leftover_noise": (["disable", "enable"], ),"clip": ("CLIP", ),"text": ("STRING", {"multiline": True})
                    
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "SyrianFalcon/nodes"
    def sample(self, clip,model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, text, return_with_leftover_noise,control_net,image,ControlNet_strength, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        prevseed = noise_seed
        strength = ControlNet_strength
        try:
            prompts,numloop = promptscreator(text,steps)
        
            if len(prompts) ==0:
                prompts.append(text)
            if len(numloop) < 1:
                numloop.append(steps-1)
        except:
            prompts = [text,0]
            numloop = []
        print(prompts)
        arr = alternatingtokenfunc(prompts[0])
        newprompt = prompts[0]
        newerprompt = newprompt
        for j in arr:
            if j[3]>=len(j[2]):
                j[3] = 0
            newerprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
            newprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
            j[3]+=1
        positive = ([[clip.encode(newerprompt), {}]] )
        try:
            if strength == 0:
                pass
            else:
                c = []
                control_hint = image.movedim(-1,1)
                for t in positive:
                    n = [t[0], t[1].copy()]
                    c_net = control_net.copy().set_cond_hint(control_hint, strength)
                    if 'control' in t[1]:
                        c_net.set_previous_controlnet(t[1]['control'])
                    n[1]['control'] = c_net
                    c.append(n)
                positive = c
        except:
            pass
        print(newerprompt)
        c1 = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=0, last_step=1, force_full_denoise=force_full_denoise)
        noise_seed = prevseed
        lastprompt = prompts[0]
        for i in range(0,steps-1):
            path = False
            for idx0,el in enumerate(numloop):
                if el== i+1:
                    path = True
                    idxx = idx0
            if path:
                if len(prompts)==1:
                    print("no:  "+str(prompts[idxx]))
                    lastprompt = prompts[idxx]
                    newarr = alternatingtokenfunc(prompts[idxx])
                    newprompt = prompts[idxx]
                else:
                    print("yes:  "+str(prompts[idxx+1]))
                    lastprompt = prompts[idxx+1]
                    newarr = alternatingtokenfunc(prompts[idxx+1])
                    newprompt = prompts[idxx+1]
                if len(newarr)>=1:
                    for idx,j in enumerate(newarr):
                        if arr[idx][3]>=len(j[2]):
                            arr[idx][3] = 0
                        newerprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
                        newprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
                        arr[idx][3]+=1
                else:
                    newerprompt = newprompt
                positive = ([[clip.encode(newerprompt), {}]] )            
            else:
                newprompt = lastprompt
                for j in arr:
                    if j[3]>=len(j[2]):
                        j[3] = 0
                    newerprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
                    newprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
                    j[3]+=1
                print(newerprompt)
                positive = ([[clip.encode(newerprompt), {}]] )
            try:
                if strength == 0:
                    pass
                else:
                    c = []
                    control_hint = image.movedim(-1,1)
                    for t in positive:
                        n = [t[0], t[1].copy()]
                        c_net = control_net.copy().set_cond_hint(control_hint, strength)
                        if 'control' in t[1]:
                            c_net.set_previous_controlnet(t[1]['control'])
                        n[1]['control'] = c_net
                        c.append(n)
                    positive = c
            except:
                pass
            print(newerprompt)
            prevseed = random.randint(0,10000000000)
            c1 = common_ksampler(model, prevseed, steps, cfg, sampler_name, scheduler, positive, negative, c1, denoise=denoise, disable_noise=disable_noise, start_step=i, last_step=i+1, force_full_denoise=force_full_denoise)
        newarr = alternatingtokenfunc(prompts[-1])
        for idx,j in enumerate(newarr):
            if arr[idx][3]>=len(j[2]):
                arr[idx][3] = 0
            newerprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
            newprompt = re.sub(re.escape(j[0]),j[2][j[3]],newprompt)
            arr[idx][3]+=1
        positive = ([[clip.encode(newerprompt), {}]] )
        try:
            if strength == 0:
                pass
            else:
                c = []
                control_hint = image.movedim(-1,1)
                for t in positive:
                    n = [t[0], t[1].copy()]
                    c_net = control_net.copy().set_cond_hint(control_hint, strength)
                    if 'control' in t[1]:
                        c_net.set_previous_controlnet(t[1]['control'])
                    n[1]['control'] = c_net
                    c.append(n)
                positive = c
        except:
            pass
        print(newerprompt)
        return common_ksampler(model, prevseed, steps, cfg, sampler_name, scheduler, positive, negative, c1, denoise=denoise, disable_noise=disable_noise, start_step=steps-1, last_step=steps, force_full_denoise=force_full_denoise,flag=True)
        


NODE_CLASS_MAPPINGS = {
    "KSamplerPromptEdit":KSamplerPromptEdit,
    "KSamplerAlternate":KSamplerAlternate,
    "KSamplerPromptEditAndAlternate":KSamplerPromptEditAndAlternate,
}
