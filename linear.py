import json
import os
import random
import shutil
import time
import timm
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as t_datasets

from torch.utils.tensorboard import SummaryWriter
from enum import Enum
from util import misc

DATA_CACHE_DIR = "/data3/datasets/"

class MetricMode(Enum):
    ACC = "accuracy"
    MEAN_CLS_ACC = "mean-class-accuracy"
    MEAN_AP = "mean AP"

# name: {class, root, num_classes, metric}
LINEAR_DATASETS = {
    'in1k': [t_datasets.ImageFolder, f'{DATA_CACHE_DIR}/imagenet', 1000, MetricMode.ACC],
    'in100': [t_datasets.ImageFolder, f'{DATA_CACHE_DIR}/imagenet-100', 100, MetricMode.ACC],
    'cifar10': [t_datasets.CIFAR10, f'{DATA_CACHE_DIR}/cifar', 10, MetricMode.ACC],
    'cifar100': [t_datasets.CIFAR100, f'{DATA_CACHE_DIR}/cifar', 100, MetricMode.ACC],
    
    'aircraft': [t_datasets.FGVCAircraft, f'{DATA_CACHE_DIR}/raw', 100, MetricMode.MEAN_CLS_ACC],
    'dtd': [t_datasets.DTD, f'{DATA_CACHE_DIR}/raw', 47, MetricMode.ACC],   # simplified
    'flowers': [t_datasets.Flowers102, f'{DATA_CACHE_DIR}/raw', 102, MetricMode.MEAN_CLS_ACC],
    'pets': [t_datasets.OxfordIIITPet, f'{DATA_CACHE_DIR}/raw', 37, MetricMode.MEAN_CLS_ACC],
    'caltech101': [t_datasets.Caltech101, f'{DATA_CACHE_DIR}/raw', 102, MetricMode.MEAN_CLS_ACC],   # simplified
    'food101': [t_datasets.Food101, f'{DATA_CACHE_DIR}/raw', 101, MetricMode.ACC],  # simplified
    'sun397': [t_datasets.SUN397, f'{DATA_CACHE_DIR}/raw', 397, MetricMode.ACC],
}
    
def main_print(obj):
    if misc.is_main_process():
        print(obj)
        
        
def get_args_parser():
    parser = argparse.ArgumentParser(description='Linear probe evaluation', add_help=False)

    parser.add_argument('--data', default='', type=str,
                        help='linear probing dataset path')
    parser.add_argument('--output-dir', default='./outputs/', type=str,
                        help='path where to save checkpoints')
    parser.add_argument('--model', default='base', choices=['small', 'base', 'base_p32', 'large'],
                        help='model architecture: (default: base ViT-B/16)')
    parser.add_argument('--workers', default=12, type=int,
                        help='number of data loading workers per GPU (default: 12)')
    parser.add_argument('--epochs', default=90, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='number of samples per-device/per-gpu ')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=0., type=float,
                        help='weight decay (default: 0.)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--eval-freq', default=1, type=int,
                        help='evaluation frequency by epochs (default: 1)')

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')

    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to pretrained checkpoint')
    parser.add_argument('--use_bn', action='store_true',
                        help='use batch norm in the linear classifier')

    parser.add_argument('--dataset', type=str, default='imagenet-1k', 
                        help='name of the dataset to evaluate on')
    parser.add_argument('--drop_last', action='store_true',
                        help='If set, drops the last incomplete batch of data if its size is smaller than batch_size.')
    
    parser.add_argument('--base_lrs', default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
                        type=float, nargs='+')
    return parser


def get_model_and_optimizer(args, num_classes):
    # load pre-trained model
    if os.path.isfile(args.pretrained):
        main_print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location=f"cuda:{args.gpu}")
        state_dict = checkpoint['model']

        prefix = 'visual.'
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and not k.startswith(prefix + 'head'):
                state_dict[k[len('visual.'):]] = state_dict[k]
            del state_dict[k]
    else:
        raise Exception(f"No pre-trained model specified: {args.pretrained}")

    # create model
    patch_size, model_size = 16, args.model
    model_info = args.model.split("_p", 1)
    if len(model_info)==2:
        model_size, patch_size = model_info[0], model_info[1]
    model = timm.create_model(f"vit_{model_size}_patch{patch_size}_224", num_classes=num_classes)
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"head.weight", "head.bias"}

    for name, param in model.named_parameters():
        if name not in ['head.weight', 'head.bias']:
            param.requires_grad = False

    # delete the last fc layer, and instead add a bunch of classifiers
    del model.head
    feat_dim = model.cls_token.shape[-1]
    linear_classifiers, optim_param_groups = add_linear_classifier(
        feat_dim, num_classes, args.base_lrs, args.batch_size, args.use_bn)

    model.cuda(args.gpu)
    if args.distributed:
        linear_classifiers = torch.nn.parallel.DistributedDataParallel(
            linear_classifiers, device_ids=[args.gpu])

    optimizer = torch.optim.SGD(optim_param_groups,
                                lr=0.0,  # fake lr
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    return model, linear_classifiers, optimizer


def get_dataset(dset, root, split, transform):
    needs_offset = [t_datasets.Flowers102]
    target_transform = (lambda y: y - 1) if dset in needs_offset else None

    if dset == t_datasets.ImageFolder:
        return dset(os.path.join(root, split), transform=transform, target_transform=target_transform)
    else:
        try:
            return dset(root, train=(split == 'train'), transform=transform, target_transform=target_transform, download=True)
        except:
            try:
                return dset(root, split=split, transform=transform, target_transform=target_transform, download=True)
            except:
                return dset(root, transform=transform, target_transform=target_transform, download=True)

        
def get_data_loaders(args, dset, data_dir, split=0.8):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img if img.mode == "RGB" else img.convert("RGB")),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: img if img.mode == "RGB" else img.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = args.data if len(args.data) else data_dir
    
    # if (dset in [Caltech101]) and ((not os.path.exists(os.path.join(data_dir, 'train.txt'))) or (not os.path.exists(os.path.join(data_dir, 'test.txt')))):
    #     create_caltech101_splits(data_dir)

    if dset in [t_datasets.Food101]:
        train_dataset = get_dataset(dset, data_dir, 'train', train_transform)
        val_dataset = get_dataset(dset, data_dir, 'test', val_transform)
    elif dset in [t_datasets.OxfordIIITPet]:
        train_dataset = get_dataset(dset, data_dir, 'trainval', train_transform)
        val_dataset = get_dataset(dset, data_dir, 'test', val_transform)
    elif dset in [t_datasets.Caltech101]:
        trainval_dataset = get_dataset(dset, data_dir, 'train', train_transform)
        total_len = len(trainval_dataset)
        train_len = int(split * total_len)
        val_len = total_len - train_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            trainval_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(args.seed)  # for reproducibility
        )
        # Override val transform for val_dataset
        val_dataset.dataset.transform = val_transform
    else:
        train_dataset = get_dataset(dset, data_dir, 'train', train_transform)
        val_dataset = get_dataset(dset, data_dir, 'val', val_transform)
    print(f"train data:{len(train_dataset)}, val data:{len(val_dataset)}")
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=args.drop_last)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, len(train_dataset), train_sampler


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


def add_linear_classifier(feat_dim, num_classes, learning_rates, batch_size, use_bn=False):
    linear_classifier_dict = nn.ModuleDict()
    optim_param_groups = []
    for blr in learning_rates:
        lr = blr * batch_size * misc.get_world_size() / 256

        linear_classifier = nn.Linear(feat_dim, num_classes)
        linear_classifier.weight.data.normal_(mean=0.0, std=0.01)
        linear_classifier.bias.data.zero_()
        if use_bn:
            linear_classifier = nn.Sequential(
                torch.nn.SyncBatchNorm(feat_dim, affine=False, eps=1e-6),
                linear_classifier
            )
        linear_classifier.cuda()

        name = f"{blr:.4f}".replace('.', '_')
        linear_classifier_dict[f"classifier_lr_{name}"] = linear_classifier
        optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    # add to ddp mode
    linear_classifiers = AllClassifiers(linear_classifier_dict)
    return linear_classifiers, optim_param_groups


def train(train_loader, model, linear_classifiers, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.eval()
    linear_classifiers.train(True)

    all_top1 = {k: AverageMeter('Acc@1', ':6.2f') for k in linear_classifiers.module.classifiers_dict.keys()}
    all_top5 = {k: AverageMeter('Acc@5', ':6.2f') for k in linear_classifiers.module.classifiers_dict.keys()}
    all_losses = {k: AverageMeter('Loss', ':.4e') for k in linear_classifiers.module.classifiers_dict.keys()}
    loss_fn = nn.CrossEntropyLoss()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            features = model.forward_features(images)
        # outputs = linear_classifiers(features.mean(dim=1))
        outputs = linear_classifiers(features[:, 0, :])
        cls_losses = {f"loss_{k}": loss_fn(v, target) for k, v in outputs.items()}
        loss = sum(cls_losses.values())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        min_loss = 1e5
        max_acc1 = -1
        max_acc5 = -1
        for k, v in outputs.items():
            acc1, acc5 = accuracy(v, target, topk=(1, 5))
            all_top1[k].update(acc1.item(), images.size(0))
            all_top5[k].update(acc5.item(), images.size(0))
            all_losses[k].update(cls_losses[f"loss_{k}"].item(), images.size(0))
            min_loss = min(min_loss, cls_losses[f"loss_{k}"].item())
            max_acc1 = max(max_acc1, acc1.item())
            max_acc5 = max(max_acc5, acc5.item())

        # logging the best loss/accuracy across all classifiers
        losses.update(min_loss, images.size(0))
        top1.update(max_acc1, images.size(0))
        top5.update(max_acc5, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg}, all_top1, all_top5, all_losses


def validate(val_loader, model, linear_classifiers, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    linear_classifiers.eval()

    all_top1 = {k: AverageMeter('Acc@1', ':6.2f') for k in linear_classifiers.module.classifiers_dict.keys()}
    all_top5 = {k: AverageMeter('Acc@5', ':6.2f') for k in linear_classifiers.module.classifiers_dict.keys()}
    all_losses = {k: AverageMeter('Loss', ':.4e') for k in linear_classifiers.module.classifiers_dict.keys()}

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            features = model.forward_features(images)
            outputs = linear_classifiers(features[:,0,:])

            my_losses = {f"loss_{k}": nn.CrossEntropyLoss()(v, target) for k, v in outputs.items()}
            min_loss = 1e6
            max_acc1 = -1
            max_acc5 = -1
            for k, v in outputs.items():
                acc1, acc5 = accuracy(v, target, topk=(1, 5))
                all_top1[k].update(acc1.item(), images.size(0))
                all_top5[k].update(acc5.item(), images.size(0))
                all_losses[k].update(my_losses[f"loss_{k}"].item(), images.size(0))
                min_loss = min(min_loss, my_losses[f"loss_{k}"].item())
                max_acc1 = max(max_acc1, acc1.item())
                max_acc5 = max(max_acc5, acc5.item())

            # logging the best loss/accuracy across all classifiers
            losses.update(min_loss, images.size(0))
            top1.update(max_acc1, images.size(0))
            top5.update(max_acc5, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        main_print('Monitored (fake) accuracy * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg}, all_top1, all_top5, all_losses


def save_checkpoint(state, is_best, output_dir):
    ckpt_path = f'{output_dir}/linear_checkpoint.pt'
    best_path = f'{output_dir}/linear_best.pt'
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, best_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        main_print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,), metric=MetricMode.ACC, num_classes=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        if metric == MetricMode.ACC:
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        
        elif metric == MetricMode.MEAN_CLS_ACC:
            pred = output.argmax(dim=1)
            cm = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), labels=list(range(output.size(1))))
            cls_acc = cm.diagonal() / cm.sum(axis=1)
            return float(np.nanmean(cls_acc)) * 100

        elif metric == MetricMode.MEAN_AP:
            assert target.ndim == 2 and output.ndim == 2, "For MEAN_AP, target and output should be [B, C]"
            assert num_classes is not None, "num_classes must be specified for MEAN_AP"
            aps = []
            for cls in range(num_classes):
                gt_cls = target[:, cls].cpu().numpy()
                pred_cls = output[:, cls].cpu().numpy()
                ap = voc_eval_cls(gt_cls, pred_cls)  # AP for one class
                aps.append(ap)
            return float(np.mean(aps)) * 100
        else:
            raise NotImplementedError(f"Metric {metric} not supported")


def main(args):
    misc.init_distributed_mode(args)
    cudnn.benchmark = True

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    dset, data_dir, num_classes, metric = LINEAR_DATASETS[args.dataset]
    
    # get model, classifier, and optimizer
    model, linear_classifiers, optimizer = get_model_and_optimizer(args, num_classes)

    # get data loaders
    train_loader, val_loader, num_train_samples, train_sampler = get_data_loaders(args, dset, data_dir)
    
    max_iter = args.epochs * (num_train_samples // (args.batch_size * misc.get_world_size()))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)

    if misc.is_main_process():
        main_print(args)
        
    if misc.is_main_process() and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    args.log_dir = os.path.join(args.output_dir, "tensorboard")
    if misc.is_main_process() and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(args.log_dir)
    else:
        log_writer = None

    max_acc = -1
    name = None

    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write("args:\n")
            f.write(json.dumps(vars(args), indent=4) + "\n")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_stats, train_all_top1, train_all_top5, train_all_losses = train(
            train_loader, model, linear_classifiers, optimizer, scheduler, epoch, args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if (epoch + 1) % args.eval_freq != 0:
            continue
        
        # evaluate on validation set
        val_stats, val_all_top1, val_all_top5, val_all_losses = \
            validate(val_loader, model, linear_classifiers, args)

        all_acc1 = [meter.avg for k, meter in val_all_top1.items()]
        acc1 = max(all_acc1)
        is_best = acc1 > max_acc

        # find which classifier has the best accuracy, and track its name
        for k, meter in val_all_top1.items():
            if meter.avg > max_acc:
                max_acc = meter.avg
                name = k
             
        # log to tensorboard   
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch, }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                for k, v in train_stats.items():
                    log_writer.add_scalar('train/{}'.format(k), v, epoch)
                for k, v in val_stats.items():
                    log_writer.add_scalar('val/{}'.format(k), v, epoch)
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # save checkpoint
        if misc.is_main_process():  # only the first GPU saves checkpoint
            save_checkpoint({
                'args': args,
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'linear_classifiers': linear_classifiers.state_dict(),
                'acc1': acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.output_dir)

        for k in train_all_top1.keys():
            log_stats = {'train_acc1': train_all_top1[k].avg,
                         'train_acc5': train_all_top5[k].avg,
                         'train_loss': train_all_losses[k].avg,
                         'test_acc1': val_all_top1[k].avg,
                         'test_acc5': val_all_top5[k].avg,
                         'test_loss': val_all_losses[k].avg,
                         'epoch': epoch}

            if misc.is_main_process():
                with open(os.path.join(args.output_dir, 'linear_{}.txt'.format(k)), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')

    # copy the log of the best classifier to a file called `linear.txt`
    if max_acc > 0.0:
        acc_msg = f"correct best accuracy:{max_acc:.2f}"
        main_print(acc_msg)
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(acc_msg+"\n")
        if misc.is_main_process():
            shutil.copyfile(
                os.path.join(args.output_dir, 'linear_{}.txt'.format(name)),
                os.path.join(args.output_dir, 'linear.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Linear probe evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
