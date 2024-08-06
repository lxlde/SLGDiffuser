import numpy as np
import torch
from PIL.Image import Image


def compute_alpha(beta, t):
    beta_1 = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta_1).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, text=None, image_guide=None, tester=None, **kwargs):
    from torchvision  import transforms
    toPIL = transforms.ToPILImage()
    # pic =toPIL(image_guide[0])
    # pic.show()

    # image_guide = image_guide.to('cuda')
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []

        xs = [x]
        # guide_noise = torch.randn_like(image_guide).to('cuda')
        # 上面测试用，测试随机取的z与初始图片扩散的z有什么区别
        # t = (torch.ones(1) * 999).to(x.device)
        for i, j in zip(reversed(seq), reversed(seq_next)):

            t = (torch.ones(n) * i).to(x.device)       #当前时间步
            next_t = (torch.ones(n) * j).to(x.device)  #下一个时间步
            at = compute_alpha(b, t.long())            #计算alpha_t
            at_next = compute_alpha(b, next_t.long())  #计算alphg_{t-1}
            xt = xs[-1].to('cuda')                     #取出目前最新的x_t

            # if i > 600:
            #     guide = at.sqrt() * image_guide + (1 - at).sqrt() * guide_noise
            #     xt[:,:3,...] = (image_guide + xt[:,:3,...]) / 2.

            xt = xt.detach().requires_grad_()
            et = model(xt[:,:3,...], xt[:,3:4,...], image_guide, t, text)                        #逆扩散步骤
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() #计算当前步骤估计的x_0
            x0_t = x0_t.detach().requires_grad_()
            x0_preds.append(x0_t.to('cpu'))            #保存每一步估计的x_0
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()  #eta是采样方差,为0既是DDIM，可取0-1之间
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            if i < 0:
                with torch.enable_grad():
                    pred, loss = tester.ocr_loss(x0_t[:,:3,...], text)
                grad = -torch.autograd.grad(loss, x0_t)[0]
                xt_next = at_next.sqrt() * x0_t + 2*grad + c1 * torch.randn_like(x) + c2 * et
                print("iter{}: {}".format(i, loss.item()))
            else:
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                import torchvision.utils as tvu
                import os
                if i %100 == 0 :
                    tvu.save_image(xt_next[0,3],f"test_slgnoise/800_{i}.png")
                print("iter{}".format(i))



            xt_next = xt - (at.sqrt() * x0_t + (1 - at).sqrt() * et) + xt_next
            xs.append(xt_next.to('cpu'))
        if len(xs)>10:
            xs = [xs[i] for i in range(len(xs)) if (i+1)%100 == 0]
            x0_preds = [x0_preds[i] for i in range(len(x0_preds)) if (i + 1) % 100 == 0]
        #xs[0] = xs[0].to('cpu')

    return xs, x0_preds


def ddpm_steps(x, text, image_guide, tester, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            #下面计算参数同上
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1

            x = xs[-1].to('cuda')     #取出最新的x_t
            x = x.detach().requires_grad_()

            output = model(x, t.float())  #计算估计epsilon
            e = output

            x0_from_e = (1.0 / at).sqrt() *( x - (1.0 / 1 - at).sqrt() * e)  #这里我改了一下，原本的计算公式与论文中不符，不知道是不是他故意为之，总之先改了
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))

            #下面暂时没看懂

            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()

            with torch.enable_grad():
                pred, loss = tester.ocr_loss(x, text)
            if i <= 300:
                grad = -torch.autograd.grad(loss, x)[0]
                sample = mean + 10*grad + mask * torch.exp(0.5 * logvar) * noise
            else:
                sample = mean + mask * torch.exp(0.5 * logvar) * noise

            print("iter{}: {}".format(i, loss.item()))

            xs.append(sample.to('cpu'))
        if len(xs)>10:
            xs = [xs[i] for i in range(len(xs)) if (i+1)%100 == 0]
            x0_preds = [x0_preds[i] for i in range(len(x0_preds)) if (i + 1) % 100 == 0]
    return xs, x0_preds



# def test_sample(x0, condition, model, t, b):
#     with torch.no_grad():
#         e = torch.randn_like(x0)
#         e = e.to('cuda')
#         n = x0.size(0)
#         now_t = (torch.ones(n) * t).to('cuda')  # 当前时间步
#         next_t = (torch.ones(n) * (t - 1)).to('cuda')
#
#         beta_1 = torch.cat([torch.zeros(1).to(b.device), b], dim=0)
#         at = (1 - beta_1).cumprod(dim=0).index_select(0, now_t.long() + 1).view(-1, 1, 1, 1)
#
#         at_next = (1 - beta_1).cumprod(dim=0).index_select(0, next_t.long() + 1).view(-1, 1, 1, 1)
#
#         condition = condition.to('cuda')
#         condition_t = condition * at.sqrt() + e * (1.0 - at).sqrt()
#         x0 = x0.to('cuda')
#         x_t = x0 * at.sqrt() + e * (1.0 - at).sqrt()
#
#         output = model(x_t, now_t, condition, condition_t)
#         x_0_hat = (x_t - output * (1.0 - at).sqrt()) / at.sqrt()
#         c2 = (1 - at_next).sqrt()
#         xt_next = at_next.sqrt() * x_0_hat + + c2 * output
#
#         return x_0_hat, xt_next


def test_sample(x0, condition, model, t, b):

    step = t+1
    with torch.no_grad():
        e = torch.randn_like(x0)
        e = e.to('cuda')
        n = x0.size(0)
        #以下计算当前T时刻的alpha_t与alpha_{t-1}
        now_t = (torch.ones(n) * t).to('cuda')  # 当前时间步
        next_t = (torch.ones(n) * (t - 1)).to('cuda')
        beta_1 = torch.cat([torch.zeros(1).to(b.device), b], dim=0)
        at = (1 - beta_1).cumprod(dim=0).index_select(0, now_t.long() + 1).view(-1, 1, 1, 1)
        at_next = (1 - beta_1).cumprod(dim=0).index_select(0, next_t.long() + 1).view(-1, 1, 1, 1)
        #以下是最大T时的初次加噪
        condition = condition.to('cuda')
        condition_t = condition * at.sqrt() + e * (1.0 - at).sqrt()
        x0 = x0.to('cuda')
        x_t = x0 * at.sqrt() + e * (1.0 - at).sqrt()

        xt_list = []

        for i in reversed(range(step)):
            now_t = (torch.ones(n) * t).to('cuda')
            next_t = (torch.ones(n) * (t - 1)).to('cuda')
            at = compute_alpha(b, now_t.long())
            at_next = compute_alpha(b, next_t.long())
            condition_t = condition * at.sqrt() + e * (1.0 - at).sqrt()
            empty = torch.zeros_like(condition).to('cuda')

            output = model(x_t, now_t, condition, empty)
            x_0_hat = (x_t - output * (1.0 - at).sqrt()) / at.sqrt()
            c2 = (1 - at_next).sqrt()
            xt_next = at_next.sqrt() * x_0_hat + + c2 * output
            if i % 10 == 0:
                xt_list.append(xt_next)
            x_t = xt_next
            t = t - 1

        return x_0_hat, xt_next, xt_list

