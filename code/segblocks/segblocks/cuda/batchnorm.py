import torch

def block_batch_norm(dualres, running_mean, running_var, weight, bias, training, momentum, eps):
    assert training

    exponential_average_factor = 0.0 if momentum is None else momentum
    hr, lr = dualres.highres, dualres.lowres

    if training:
        if dualres.metadata.nhighres > 0:
            n_hr = hr.shape[0]*hr.shape[2]*hr.shape[3]
            var_hr, mean_hr = torch.var_mean(hr, [0, 2, 3], unbiased=False)
        else:
            n_hr, var_hr, mean_hr = 0, 0, 0

        if dualres.metadata.nlowres > 0:
            n_lr = lr.shape[0]*lr.shape[2]*lr.shape[3]
            var_lr, mean_lr = torch.var_mean(lr, [0, 2, 3], unbiased=False)
        else:
            n_lr, var_lr, mean_lr = 0, 0, 0

        n = n_hr + n_lr
        var =  var_hr*(float(n_hr)/n) + var_lr*(float(n_lr)/n)
        mean = mean_hr*(float(n_hr)/n) + mean_lr*(float(n_lr)/n)

        with torch.no_grad():
            running_mean[:] = exponential_average_factor * mean + (1 - exponential_average_factor) * running_mean
            running_var[:] = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * running_var
    else:
        mean = running_mean
        var = running_var

    if dualres.metadata.nhighres > 0:
        hr = apply_bn_affine(hr, mean, var, weight, bias, eps)
    if dualres.metadata.nlowres > 0:
        lr = apply_bn_affine(lr, mean, var, weight, bias, eps)
    return hr, lr

@torch.jit.script
def apply_bn_affine(x:torch.Tensor, mean:torch.Tensor, var:torch.Tensor, weight:torch.Tensor, bias:torch.Tensor, eps: float):
    sqrtvar = torch.sqrt_(var + eps)
    out = x.sub(mean[None, :, None, None]).div_(sqrtvar[None, :, None, None])
    return out.mul_(weight[None, :, None, None]).add_(bias[None, :, None, None])