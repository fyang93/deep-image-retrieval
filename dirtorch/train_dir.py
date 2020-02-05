import sys
import os
import argparse
import json
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from dirtorch.utils.convenient import mkdir, load, save
from dirtorch.utils import common
from dirtorch.utils.common import tonumpy, matmul, pool
from dirtorch.utils.pytorch_loader import get_loader
from dirtorch.test_dir import extract_image_features
import dirtorch.nets as nets
import dirtorch.datasets as datasets

from dirtorch.loss import APLoss, TAPLoss


class AverageMeter(object):
    def __init__(self):
        """Computes and stores the average and current value"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def adjust_learning_rate(optimizer, epoch, lr_base):
    lr = lr_base * (1. - epoch / 300.)
    for grp in optimizer.param_groups:
        grp['lr'] = lr


def train(epoch, train_set, net, trfs, opt, lr, threads=8, batch_size=16, buffer_size=4096):
    adjust_learning_rate(opt, epoch, lr)

    loader = get_loader(train_set, trf_chain=trfs, preprocess=net.preprocess, iscuda=net.iscuda,
                        output=['img', 'label'], batch_size=batch_size, threads=threads, shuffle=True)

    net.eval()
    img_batches = []
    img_feats = torch.zeros((buffer_size, 2048), dtype=torch.float)
    img_labels = torch.zeros((buffer_size), dtype=int)
    batch_in_buffer = buffer_size // batch_size

    losses = AverageMeter()
    with tqdm.tqdm(total=len(loader.sampler) // buffer_size * batch_in_buffer, desc=f'[{epoch}]') as t:
        for i_batch, inputs in enumerate(loader):
            imgs, labels = inputs[:2]
            img_batches.append(imgs)
            imgs = common.variables([imgs], net.iscuda)[0]

            with torch.no_grad():
                desc = net(imgs)

            i = i_batch % batch_in_buffer
            img_feats[i*batch_size:(i+1)*batch_size] = desc.detach().cpu()
            img_labels[i*batch_size:(i+1)*batch_size] = labels
            t.update()

            if i + 1 == batch_in_buffer:
#                 from IPython import embed; embed()
#                 uni_labels, inv_ids, counts = torch.unique(img_labels, return_inverse=True, return_counts=True)
#                 weights = 1 / counts.float()
#                 weights = weights[inv_ids]  # query weights

                img_feats.requires_grad = True
                scores = torch.matmul(img_feats, img_feats.t())
                labels = torch.cat([(img_labels == label).unsqueeze_(0)
                                    for label in img_labels]).int()
                loss = criterion(scores, labels)
                #loss = criterion(scores, labels, weights)
                losses.update(loss.item())
                loss.backward()
                t.set_postfix(loss=f'{losses.val:0.4f}({losses.avg:0.4f})')
                grads = img_feats.grad

                net.train()
                net.apply(freeze_bn)
                opt.zero_grad()
                for i, imgs in enumerate(img_batches):
                    imgs = common.variables([imgs], net.iscuda)[0]
                    grad = common.variables([grads[i*batch_size:(i+1)*batch_size]], net.iscuda)[0]
                    desc = net(imgs)
                    desc.backward(grad)
                opt.step()

                img_batches = []
                img_feats.requires_grad = False
                net.eval()

                if (i_batch + 1) // batch_in_buffer == len(loader.sampler) // buffer_size:
                    break


def test(db, net, trfs, pooling='mean', gemp=3, detailed=False, threads=8, batch_size=16):
    """ Evaluate a trained model (network) on a given dataset.
    The dataset is supposed to contain the evaluation code.
    """
    print("\n>> Evaluation...")
    query_db = db.get_query_db()

    # extract DB feats
    bdescs = []
    qdescs = []

    trfs_list = [trfs] if isinstance(trfs, str) else trfs

    for trfs in trfs_list:
        kw = dict(iscuda=net.iscuda, threads=threads, batch_size=batch_size, same_size='Pad' in trfs or 'Crop' in trfs)
        bdescs.append(extract_image_features(db, trfs, net, desc="DB", **kw))

        # extract query feats
        qdescs.append(bdescs[-1] if db is query_db else extract_image_features(query_db, trfs, net, desc="query", **kw))

    # pool from multiple transforms (scales)
    bdescs = F.normalize(pool(bdescs, pooling, gemp), p=2, dim=1)
    qdescs = F.normalize(pool(qdescs, pooling, gemp), p=2, dim=1)

    bdescs = tonumpy(bdescs)
    qdescs = tonumpy(qdescs)

    scores = matmul(qdescs, bdescs)

    del bdescs
    del qdescs

    res = {}

    try:
        aps = [db.eval_query_AP(q, s) for q, s in enumerate(tqdm.tqdm(scores, desc='AP'))]
        if not isinstance(aps[0], dict):
            aps = [float(e) for e in aps]
            if detailed:
                res['APs'] = aps
            # Queries with no relevants have an AP of -1
            res['mAP'] = float(np.mean([e for e in aps if e >= 0]))
        else:
            modes = aps[0].keys()
            for mode in modes:
                apst = [float(e[mode]) for e in aps]
                if detailed:
                    res['APs'+'-'+mode] = apst
                # Queries with no relevants have an AP of -1
                res['mAP'+'-'+mode] = float(np.mean([e for e in apst if e >= 0]))
    except NotImplementedError:
        print(" AP not implemented!")

    #writer.add_scalar('mAP', res['mAP'], epoch)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--arch', type=str, default='resnet101_rmac', help='backbone architecture')
    parser.add_argument('--resume', type=str, default='', help='model to resume')
    parser.add_argument('--trfs', type=str, default='RandomScale(800,900,can_downscale=True,can_upscale=True), ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), RandomRotation(10), RandomTilting(0.3), RandomErasing(0.3), Pad(800, color=mean), RandomCrop(800)', nargs='+', help='train transforms (can be several)')
    parser.add_argument('--buffer-size', type=int, default=4096, help='buffer size')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--out-dim', type=int, default=2048, help='output dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--pooling', type=str, default='gem', help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')
    parser.add_argument('--threads', type=int, default=8, help='number of thread workders')
    parser.add_argument('--gpu', type=int, default=0, nargs='+', help='GPU ids')
    parser.add_argument('--cache', type=str, required=True, help='path to cache files')

    args = parser.parse_args()
    assert args.buffer_size % args.batch_size == 0
    args.iscuda = common.torch_set_gpu(args.gpu)


    train_set = datasets.create('Landmarks_clean')
    val_set = datasets.create('RParis6K')

    model_options = {'arch': args.arch,
                     'out_dim': args.out_dim,
                     'pooling': args.pooling,
                     'gemp': args.gemp}

    start_epoch = 0
    if os.path.isfile(args.resume):
        checkpoint = common.load_checkpoint(args.resume, args.iscuda)
        net = nets.create_model(pretrained='', **model_options)
        net = common.switch_model_to_cuda(net, args.iscuda, checkpoint)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = int(os.path.splitext(os.path.basename(args.resume))[0].split('_')[-1])
    else:
        net = nets.create_model(pretrained='imagenet', **model_options)
        net = common.switch_model_to_cuda(net, iscuda=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = APLoss(20, -1, 1)

    best_map = 0
    for epoch in range(start_epoch, 300):
        train(epoch, train_set, net, args.trfs, optimizer, lr=args.lr, threads=args.threads, batch_size=args.batch_size, buffer_size=args.buffer_size)
        if (epoch + 1) % 1 == 0:
            res = test(val_set, net, trfs='', pooling='mean', gemp=args.gemp, threads=args.threads, batch_size=1)
            print(' * ' + '\n * '.join(['%s = %g' % p for p in res.items()]))
            cur_map = res.get('mAP') or res.get('mAP-medium')
            if cur_map > best_map:
                best_map = cur_map
                path = os.path.join(args.cache, f'ckpts/ckpt_{epoch}.pt')
                state = {'arch': args.arch,
                         'state_dict': net.state_dict(),
                         'model_options': model_options,
                         'best_mAP': best_map}
                common.save_checkpoint(state, is_best=True, filename=path)
            print()
