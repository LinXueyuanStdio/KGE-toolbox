import torch.nn.functional as F


def negative_sample_loss(model,
                         positive_sample, negative_sample, subsampling_weight, mode,
                         single_mode="single"):
    negative_score = model((positive_sample, negative_sample), mode=mode)
    negative_score = F.logsigmoid(-negative_score).mean(dim=1)

    positive_score = model(positive_sample, mode=single_mode)
    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

    positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
    negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
    return (positive_sample_loss + negative_sample_loss) / 2


def train_step(model, optimizer,
               positive_sample, negative_sample, subsampling_weight, mode,
               align_positive_sample, align_negative_sample, align_subsampling_weight, align_mode,
               device="cuda"):
    model.train()
    optimizer.zero_grad()

    positive_sample = positive_sample.to(device)
    negative_sample = negative_sample.to(device)
    subsampling_weight = subsampling_weight.to(device)
    if align_mode is not None:
        align_positive_sample = align_positive_sample.to(device)
        align_negative_sample = align_negative_sample.to(device)
        align_subsampling_weight = align_subsampling_weight.to(device)

    raw_loss = model.loss(model,
                          positive_sample, negative_sample, subsampling_weight,
                          mode, "single")
    if align_mode is not None:
        align_loss = model.loss(model,
                                align_positive_sample, align_negative_sample, align_subsampling_weight,
                                align_mode, "align-single")
    else:
        align_loss = raw_loss

    loss = (raw_loss + align_loss) / 2
    loss.backward()
    optimizer.step()

    return loss.item(), raw_loss.item(), align_loss.item()
