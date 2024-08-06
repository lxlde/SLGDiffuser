import os
import logging
import time
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.utils.data as data
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from models.diffusion import Model
from pren_main.Nets.model import Model as OCRModel
from pren_main.recog import Recognizer
from trocr.use import Recognizer1
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu

from pren_main.Configs.testConf import configs as ocr_configs

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # print(self.logvar)
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt_1780000.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (img1, img2, mask, text) in enumerate(train_loader):
                n = img1.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                # import torchvision.transforms as transforms
                # img_test = img1.squeeze()
                # a2 = transforms.ToPILImage()
                # img_test = a2(img_test)
                # img_test.show()

                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                mask = mask.to(self.device)


                img1 = data_transform(self.config, img1)
                img2 = data_transform(self.config, img2)
                mask = data_transform(self.config, mask)

                e_img = torch.randn_like(img1)
                e_mask = torch.randn_like(mask)
                b = self.betas

                # antithetic sampling
                x = img1
                guide = img2

                t = torch.randint(
                    low=1, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss, x_0_hat, mask_0_hat = loss_registry[config.model.type](model, x, guide, mask, text, t, e_img, e_mask, b)
                # mask_0_hat = mask_0_hat.repeat(1, 3, 1, 1)
                # print(mask_0_hat[0].shape)
                #
                # print(x[0].shape)
                # print(x_0_hat[0].shape)
                if step % 5000 == 0:
                    mask_0_hat = mask_0_hat.repeat(1,3,1,1)
                    # print(mask_0_hat[0].shape)
                    # print(x[0].shape)
                    tvu.save_image(torch.cat([(x[0]+1)*2, (x_0_hat[0]+1)*2, (mask_0_hat[0]+1)*2], dim=2), os.path.join(args.train_image_path, f"{step}.png"))
                    print("save train results")

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"epoch: {epoch}, step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        checkpoint = torch.load(ocr_configs.model_path)
        ocr_model = OCRModel(checkpoint['model_config'])
        ocr_model.load_state_dict(checkpoint['state_dict'])
        print('[Info] Load model from {}'.format(ocr_configs.model_path))

        tester = Recognizer(ocr_model)
        # processor = TrOCRProcessor.from_pretrained('trocr/trocr-base-str')
        # ocr_model = VisionEncoderDecoderModel.from_pretrained('trocr/trocr-base-str')
        #
        # tester = Recognizer1(ocr_model, processor)

        model = Model(self.config)
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)
            # model = model.module

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model,tester = tester)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, tester = None):

        args = self.args
        config = self.config
        dataset = get_dataset(args, config)
        test_loader = data.DataLoader(
            dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        img_id = 0
        print(f"starting from image {img_id}")
        total_n_samples = config.sampling.total_sample_num
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for i in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                import torchvision.transforms as transforms
                from PIL import Image
                transform = transforms.Compose([
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor()]
                )
            # for (image_guide,text) in enumerate(test_loader):
                use_stand = 3
                try:
                    if use_stand ==3:
                        image_guide = dataset[i][0]
                        image_guide = torch.unsqueeze(image_guide, dim=0)
                        text = dataset[i][1]
                        self.args.text = text
                        img_source_path = dataset[i][2]
                    elif use_stand == 0:
                        image_guide = dataset[i][0]
                        image_guide = torch.unsqueeze(image_guide, dim=0)
                        text = dataset[i][1]
                        self.args.text = text
                        x_t = torch.randn(
                            1,
                            config.data.channels,
                            config.data.image_size[0],
                            config.data.image_size[1],
                            device=self.device,
                        )
                        print("")
                    elif config.data.dataset == "WORD4":   
                        image_guide = dataset[i][0]
                        image_guide = torch.unsqueeze(image_guide, dim=0)
                        text = dataset[i][1]
                        self.args.text = text
                        stand_image = dataset[i][2]
                        stand_mask = dataset[i][3]
                        img_source = Image.open(stand_image).convert("RGB")
                        # img_source = Image.open(img_source_path).convert("RGB")
                        # img_source.show()
                        img_source = transform(img_source)
                        img_source = img_source.to(self.device)
                        img_source_mask = Image.open(stand_mask).convert("L")
                        img_source_mask = transform(img_source_mask)
                        img_source_mask = img_source_mask.to(self.device)
                        x0_mask = data_transform(self.config, img_source_mask)
                        x_0 = torch.cat((img_source, x0_mask), dim=0)
                        e_img = torch.randn_like(x_0).to(self.device)  #
                        t1 = torch.IntTensor([699]).to(self.device)
                        b = self.betas
                        a_t = (1 - b).cumprod(dim=0).index_select(0, t1).view(-1, 1, 1, 1)
                        x_t = x_0 * a_t.sqrt() + e_img * (1.0 - a_t).sqrt()
                        x_t = x_t.to(self.device)
                    elif config.data.dataset == "WORD1":
                        image_guide = dataset[i][0]
                        image_guide = torch.unsqueeze(image_guide, dim=0)
                        text = dataset[i][1]
                        self.args.text = text
                        stand_image = dataset[i][2]
                        img_source = torch.randn(1, 3, config.data.image_size[0], config.data.image_size[1],
                                                 device=self.device, )
                        img_source_mask = Image.open(stand_image).convert("L")
                        img_source_mask = transform(img_source_mask)
                        img_source_mask = img_source_mask.to(self.device)
                        x0_mask = data_transform(self.config, img_source_mask)
                        e_img = torch.randn_like(x0_mask).to(self.device)  #
                        t1 = torch.IntTensor([1]).to(self.device)
                        b = self.betas
                        a_t = (1 - b).cumprod(dim=0).index_select(0, t1).view(-1, 1, 1, 1)
                        x_t_mask = x0_mask * a_t.sqrt() + e_img * (1.0 - a_t).sqrt()
                        import torchvision.transforms as transforms
                        a2 = transforms.ToPILImage()
                        x = x_t_mask.squeeze(0)
                        image = a2(x)
                        image.show()
                        x_t = torch.cat((img_source, x_t_mask), dim=1)
                        x_t = x_t.to(self.device)
                except:
                    if i<dataset.__len__(): 
                        image_guide = dataset[i][0]
                        image_guide = torch.unsqueeze(image_guide, dim=0)
                        text = dataset[i][1]
                        self.args.text = text
                        x_t = torch.randn(
                            1,
                            config.data.channels,
                            config.data.image_size[0],
                            config.data.image_size[1],
                            device=self.device,
                        )
                    else:
                        image_guide = dataset[i%dataset.__len__()][0]
                        image_guide = torch.unsqueeze(image_guide, dim=0)
                        text = dataset[i%dataset.__len__()][1]
                        self.args.text = text
                        x_t = torch.randn(
                            1,
                            config.data.channels,
                            config.data.image_size[0],
                            config.data.image_size[1],
                            device=self.device,
                        )


                n = config.sampling.batch_size
                torch.cuda.manual_seed(int(time.time()))


                flag = 1
                if flag ==3:
                    img_source = torch.randn(3,config.data.image_size[0],config.data.image_size[1],
                        device=self.device,)
                    x0_mask = torch.randn(1,config.data.image_size[0],config.data.image_size[1],
                        device=self.device,)

                x, x0_pred = self.sample_image(x_t, self.args.text, tester, model, image_guide, last=False)
                # x, x0_pred = self.sample_image(x_t,text, tester, model, image_guide, last=False)
                for i, img in enumerate(x):
                    x[i] = inverse_data_transform(config, img)

                for i in range(n):
                    tvu.save_image(x0_pred[len(x)-1][i,:3],os.path.join(self.args.image_folder, f"text/{str(img_id).zfill(2)}.png"))
                    # tvu.save_image(x[len(x) - 1][i, :3],
                    #                os.path.join(self.args.image_folder, f"{img_id}_{i}_x111.png"))
                    tvu.save_image(x0_pred[len(x) - 1][i, 3:4], os.path.join(self.args.image_folder, f"text_mask/{str(img_id).zfill(2)}.png"))
                    # tvu.save_image(
                    #     torch.cat([x0_pred[j][i, :3] for j in range(len(x0_pred))], dim=2), os.path.join(self.args.image_folder, f"{img_id}_{i}_x0.png")
                    # )
                    # tvu.save_image(
                    #     torch.cat([x0_pred[j][i, 3:] for j in range(len(x0_pred))], dim=2), os.path.join(self.args.image_folder, f"{img_id}_{i}_mask.png")
                    # )
                    # tvu.save_image(
                    #     torch.cat([x[j][i, :3] for j in range(len(x))], dim=2),
                    #     os.path.join(self.args.image_folder, f"{img_id}_{i}_x0_process.png")
                    # )
                    # tvu.save_image(
                    #     torch.cat([x[j][i, 3:] for j in range(len(x))], dim=2),
                    #     os.path.join(self.args.image_folder, f"{img_id}_{i}_mask_process.png")
                    # )
                    tvu.save_image(
                        image_guide,
                        os.path.join(self.args.image_folder, f"guide/{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, text, tester, model, image_guide, last=True,):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            #下面的参数决定DDIM采样均匀跳步或者平方跳步
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps  #timesteps=1000 num_timesteps=1000, skip=1, num_timesteps可在main改
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs, x0_pred = generalized_steps(x, seq, model, self.betas, text, image_guide, tester, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x, x0_pred = ddpm_steps(x, text, image_guide, tester, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x, x0_pred

    def sample_image_conditional(self, x, model, last=True):
        pass

    def test(self, t):
        model = Model(self.config)
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        args = self.args
        config = self.config
        dataset, _ = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        iters = iter(train_loader)
        from functions.denoising import test_sample
        for i in range(10):
            content, target = next(iters)
            content = content.to('cuda')
            target = target.to('cuda')
            n = content.size(0)

            x0_hat, xt_next, xt_list = test_sample(target, content, model, t, self.betas)
            tvu.save_image(torch.cat([target[0], content[0], x0_hat[0], xt_next[0]], dim=2), os.path.join('exp', 'image_samples',self.args.image_folder, f"{t}_{i}_testsample.png"))
            tvu.save_image(torch.cat([xt_list[i][0] for i in range(len(xt_list))], dim=2),
                           os.path.join('exp', 'image_samples', self.args.image_folder, f"{t}_{i}_process.png"))

