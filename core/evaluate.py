import numpy as np
import torch


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(gt2ds, preds):
    if isinstance(gt2ds, torch.Tensor):
        gt2ds = gt2ds.detach().cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()


    cnt = np.sum(gt2ds)
    if cnt == 0:
        return 0, 0

    joint_error = np.sqrt(np.sum((gt2ds - preds)**2, axis=-1))
    joint_error = np.sum(joint_error) / cnt

    return joint_error, cnt


def root_accuracy(gt2ds, preds):
    if isinstance(gt2ds, torch.Tensor):
        gt2ds = gt2ds.detach().cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    cnt = np.sum(gt2ds)
    if cnt == 0:
        return 0, 0

    joint_error = np.sqrt(np.sum((gt2ds - preds)**2, axis=-1))
    joint_error = np.sum(joint_error) / cnt

    return joint_error, cnt