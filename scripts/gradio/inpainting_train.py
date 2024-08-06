import sys
from random import randint

import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path
import argparse, os
import sys

import torch.utils.data as data
from torchvision.utils import save_image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


torch.set_grad_enabled(False)


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def initialize_model(config, ckpt,ckpt_unet= None):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
    # model.model.diffusion_model.load_state_dict(torch.load(ckpt_unet)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)
    image = image
    mask = mask
    with torch.no_grad():
            #torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)
        mask = batch["masked_image"]
        #cv2.imwrite("masked_img.jpg", mask)
        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            print(ck)
            cc = batch[ck].float()
           #print(cc+"````````````````````````")
            if ck != model.masked_image_key: # 不是masked_image
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])# size:64*64 插值
            else:#masked_image
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))#过VAE的encoder
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]} #图像和文本

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]} # 只有图像，文本不过crossatten

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

class FT_dataset(data.Dataset):

    def __init__(self, image_root, mask_root, gen_root,prompt):
        super(FT_dataset, self).__init__()
        self.image_list=[]
        self.txt_list = []
        self.mask_list = []
        self.masked_image_list = []
        self.gen_list = []
        for img_name in os.listdir(os.path.join(image_root)):
            img = Image.open(os.path.join(image_root, img_name)).resize([512,512])
            mask_name = img_name.replace('GT_', '')
            mask = Image.open(os.path.join(mask_root, mask_name)).resize([512,512])
            gen_name = img_name.replace('GT', 'Out')
            gen = Image.open(os.path.join(gen_root, gen_name)).resize([512,512])

            image = np.array(img.convert("RGB"))
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
            image = image.squeeze(dim=0)
            mask = np.array(mask.convert("L"))
            mask = mask[None][None]
            mask = mask.astype(np.float32) / 255.0
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)
            mask = mask.squeeze(dim=0)
            masked_image = image * (mask < 0.5)

            gen = np.array(gen.convert("RGB"))
            gen = gen[None].transpose(0, 3, 1, 2)
            gen = torch.from_numpy(gen).to(dtype=torch.float32) / 127.5 - 1.0
            gen = gen.squeeze(dim=0)

            self.image_list.append(image)
            self.txt_list.append(prompt)
            self.mask_list.append(mask)
            self.masked_image_list.append(masked_image)
            self.gen_list.append(gen)


    def __getitem__(self, item):
        return self.image_list[item],self.txt_list[item],self.mask_list[item],self.masked_image_list[item],self.gen_list[item]

    def __len__(self):
        return len(self.image_list)


# sampler = initialize_model("../../configs/stable-diffusion/v2-inpainting-inference.yaml"
#                            ,"../../experiment/finetune_AAA_FUZA最终结果3张图100个epoch就能效果很好.ckpt")
sampler = initialize_model("../../configs/stable-diffusion/v2-inpainting-inference.yaml","../../experiment/inpainting.ckpt")
# sampler = initialize_model("../../configs/stable-diffusion/v2-inpainting-inference.yaml","../../experiment/SAM100NOTEXT.ckpt")


def test(ddim_steps=45, num_samples=4, scale=9, seed=42,img = None, mask = None, root= None):

    # img = Image.open(os.path.join("../../data/finetune_letter", "02_05.png")).resize([512, 512])
    init_image = img.convert("RGB")
    # mask = Image.open(os.path.join("../../data/finetune_letter", "mask2_02_05.png")).resize([512, 512])
    init_mask = mask.convert("RGB")
    image = pad_image(init_image)  # resize to integer multiple of 32
    mask = pad_image(init_mask)  # resize to integer multiple of 32
    width, height = image.size
    print("Inpainting...", width, height)
    # prompt = "remove the letter and restore the background like the arround"
    prompt = " white background,no word"
    # prompt = "a photo of card with solid color background"
    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width
    )
    # a = result[0].resize([247,44])
    # b = image.resize([247,44])
    # from torchvision import transforms
    # totensor = transforms.ToTensor()
    # a = totensor(a)
    # b = totensor(b)
    # ab = [a , b]
    # save_image(torch.cat(ab, dim=1),root, normalize=True)
    result[0].resize([512,512]).save(root)

def main():

    # for i in range(341):
    #     source_path = '../../data/part_test/part_images/{}.jpg'.format(i)
    #     mask_path = '../../data/part_test/part_masks/{}.jpg'.format(i)
    #     target_path = '../../data/part_test/part_gen1/{}.jpg'.format(i)
    #     try:
    #         f = open(source_path)
    #     except Exception:
    #         continue
    #     img = Image.open(source_path).resize([512, 512])
    #     mask = Image.open(mask_path).resize([512, 512])
    #     test(45, 4, 9, seed, img, mask, target_path)

    for i in range(10):
        # seed = 42
        seed = randint(1, 100000)

        image_root = "../../data/AAA_FUZA_test/img_origin"
        mask_root =  "../../data/AAA_FUZA_test/img_mask"
        save_root = "../../data/AAA_FUZA_test/output_inpaint/"
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for img_name in os.listdir(os.path.join(image_root)):
            img = Image.open(os.path.join(image_root, img_name)).resize([512, 512])
            mask_name = img_name
            mask = Image.open(os.path.join(mask_root, mask_name)).resize([512, 512])
            test(45, 4, 9, seed, img, mask, save_root+img_name+str(i)+".jpg")
        # seed = randint(1, 100000)
        # save_root = "../../data/test_sam"
        # if not os.path.exists(save_root):
        #     os.makedirs(save_root)
        # img_name = "bbb.jpg"
        # mask_name = 'bbb.png'
        # for i in range(2):
        #     img2 = Image.open(os.path.join("../../data",img_name)).resize([512, 512])
        #     mask2 = Image.open(os.path.join("../../data", mask_name)).resize([512, 512])
        #     test(45, 4, 9, seed, img2, mask2, "../../data/" +str(i)+img_name)

main()


