import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          guide: torch.Tensor,
                          mask: torch.Tensor,
                          text: str,
                          t: torch.LongTensor,
                          e_img: torch.Tensor,
                          e_mask: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a_t = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x_t = x0 * a_t.sqrt() + e_img * (1.0 - a_t).sqrt()
    mask_t = mask * a_t.sqrt() + e_mask * (1.0 - a_t).sqrt()  #a_t可能会出现和mask单通道图像无法相乘的问题，debug时注意

    output = model(x_t, mask_t, guide, t.float(), text) #(B,4,H,W)通道数为4，前三个为图片，最后一个为mask

    x_0_hat = (x_t - output[:,:3,...] * (1.0 - a_t).sqrt()) / a_t.sqrt()
    mask_0_hat = (mask_t - output[:,3:4,...] * (1.0 - a_t).sqrt()) / a_t.sqrt()
    # print(x_0_hat.shape)
    # print(mask_0_hat.shape)
    # print(mask_t.shape)
    # print(output[:,3:4, ...].shape)
    # print(output[:,-1,...].shape)
    # print((output[:,-1,...] * (1.0 - a_t).sqrt()).shape)
    # print(a_t.shape)
    e = torch.cat([e_img,e_mask],dim=1)
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3)), x_0_hat, mask_0_hat
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0), x_0_hat, mask_0_hat

def l1_loss(model,
            x0: torch.Tensor,
            cond: torch.Tensor,
            t: torch.LongTensor,
            e: torch.Tensor,
            b: torch.Tensor, keepdim=False):
    pass



loss_registry = {
    'simple': noise_estimation_loss,
    'l1': l1_loss,
}
