import argparse
import datetime
import json
import math
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from dataset.util import GaussianBlur, DownSampleAndUpsample
from dataset.data import SupconDataset
from models.losses import MultiPosConLoss, MultiPosConLossMM, ImgTxtConLoss
from models.vision_models import model_dict as v_model_dict
from models.multimodal_models import model_dict as vt_model_dict

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.quality_driven_weight import quality_driven_module

def get_args_parser():
    parser = argparse.ArgumentParser('GenView pre-training', add_help=False)
    parser.add_argument('--epochs', default=15, type=int)

    # Model parameters
    parser.add_argument('--model', default='base', type=str,
                        help='Name of model to train')
    parser.add_argument('--tokenizer', default='CLIP', type=str, choices=['CLIP'],
                        help='tokenization choice (only CLIP here)')

    # add self-supervised learning parameters
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-scale', default=1.0, type=float,
                        help='loss scale for SimCLR objective')
    parser.add_argument('--ssl-temp', default=0.1, type=float,
                        help='softmax temperature for SimCLR objective')
    parser.add_argument('--ssl-temp-cos', action='store_true',
                        help='gradually increase the ssl temperature')
    parser.add_argument('--ssl-temp-min', default=0.05, type=float,
                        help='minimum temperature of the cosine cycle')
    parser.add_argument('--ssl-temp-max', default=0.1, type=float,
                        help='maximum temperature of the cosine cycle')
    parser.add_argument('--vl-projection', default='linear', type=str,
                        choices=['linear', 'mlp'], help='projection head type')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (absolute lr)')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 for AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.98,
                        help='beta2 for AdamW optimizer')
    parser.add_argument('--blr', type=float, default=2.0e-4,
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0.,
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--clip_grad', type=float, default=None,
                    help='Max norm for gradient clipping. If None, no clipping is applied.')

    parser.add_argument('--warmup_epochs', type=float, default=1.0,
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--early_stop_epoch', type=int, default=None, 
                        help='Epoch to stop training early')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--save_freq', default=5, type=int,
                        help='the frequency to save the model')
    parser.add_argument('--save_epoch', nargs='+',
                        help='')
    parser.add_argument('--n_keep', default=3, type=int,
                        help='number of checkpoints to keep')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Dataset parameters
    parser.add_argument('--csv_path', default='./data/MAE/MAE_train.csv',
                        help='csv file path')
    parser.add_argument('--syn_csv_path', default=None,
                        help='csv file path for synthetic data')
    parser.add_argument('--folder_list', nargs='+',
                        help='A list of items')
    parser.add_argument('--syn_ratio', type=float, default=1.0)
    parser.add_argument('--syn_idx_list', nargs='+', default=[])
    parser.add_argument('--folder_suffix_list', nargs='+')
    parser.add_argument('--real_images_path_suffix', nargs='+')
    parser.add_argument('--sample_mode', type=str, choices=['default', 'fixed+random', 'random'],
                        help=help='Specifies the sampling mode for selecting image folders. Options are: '
                         '`default` for using all folders, '
                         '`fixed+random` for using the first folder and a random selection from the others, '
                         '`random` for randomly selecting one folder.')
    parser.add_argument('--n_img', type=int, default=1,
                        help='number of images per caption sample, default: 1')
    parser.add_argument('--num_crop', type=int, default=1,
                        help='number of crops per images')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per GPU.')
    parser.add_argument('--weak_aug', action='store_true',
                        help='use weak augmentation for each image')
    # 'path1' and 'path2' are useful when the merged dataset has duplicate image names and prompts, such as when using different batches of the same web-crawled dataset
    parser.add_argument('--path1', default='/data3/datasets/CC3M/cc3m-wds/train_data',
                        help='Path to the first directory containing images. Used when image paths start with "path1/".')
    parser.add_argument('--path2', default='/data1/datasets/CC3M/raw',
                        help='Path to the second directory containing images. Used when image paths start with "path2/".')

    # downsample aug parameter
    parser.add_argument('--downsample', action='store_true', default=True,
                        help='randomly downsample images')
    parser.add_argument('--downsample_prob', default=0.05, type=float,
                        help='prob for applying this augmentation')
    parser.add_argument('--down_res', default=[64, 128], nargs='+', type=int,
                        help='a list of downsample resolutions')
    parser.add_argument('--down_prob', default=None, nargs='+',
                        help='a list of downsample probabilities (corresponds to each resolution)')
    parser.add_argument('--launcher', choices=["none", "slurm", "pytorch"], default="none", 
                        help="job launcher. Options: none (standalone), slurm (via srun), pytorch (via torchrun).")
    
    # additional parameter
    parser.add_argument('--gamma_ii', type=float, default=2.0,
                        help='controls sensitivity for scaling image-to-image loss with quality-driven learning.')
    parser.add_argument('--gamma_it', type=float, default=2.0,
                        help='controls sensitivity for scaling image-to-text loss with quality-driven learning.')
    parser.add_argument('--epoch_switch', type=int, default=0)
    parser.add_argument('--early_loss_coefs', nargs='+', default=[1,0,0,0],
                        help="coefficients of original image loss, QD image loss, original text loss, QD text loss during early stages")
    parser.add_argument('--later_loss_coefs', nargs='+', default=[1,0,0,0],
                        help="coefficients of original image loss, QD image loss, original text loss, QD text loss during later stages")
    parser.add_argument('--standard_array_path', type=str, default='data/pca_results/convnext_base_w-laion2b-s13b-b82k-augreg/eigenvecters/pca_vectors.npy',
                    help='Path to store PCA feature values used in quality-driven learning.')
    
    return parser


def main_print(obj):
    if misc.is_main_process():
        print(obj)

    
def main(args):
    misc.init_distributed_mode(args)
    # init_distributed_mode(args)

    # ======= adapt args =======
    if args.save_epoch is not None:
        args.save_epoch = [int(i) for i in args.save_epoch]
    print("args.save_epoch", args.save_epoch)
    
    if args.down_res:
        args.down_res = [int(x) for x in args.down_res]
    if args.down_prob:
        args.down_prob = [float(x) for x in args.down_prob]
    
    args.syn_idx_list = [int(x) for x in args.syn_idx_list]
    
    args.early_loss_coefs = [float(x) for x in args.early_loss_coefs]
    args.later_loss_coefs = [float(x) for x in args.later_loss_coefs]
    
    args.align_img_txt = bool(args.early_loss_coefs[2]) or bool(args.early_loss_coefs[3]) or bool(args.later_loss_coefs[2]) or bool(args.later_loss_coefs[3])
    assert args.align_img_txt == (bool(args.early_loss_coefs[2]) or bool(args.early_loss_coefs[3]))
    assert args.align_img_txt == (bool(args.later_loss_coefs[2]) or bool(args.later_loss_coefs[3]))
        
    args.align_img_img = bool(args.early_loss_coefs[0]) or bool(args.early_loss_coefs[1]) or bool(args.later_loss_coefs[0]) or bool(args.later_loss_coefs[1])
    assert args.align_img_img == (bool(args.early_loss_coefs[0]) or bool(args.early_loss_coefs[1]))
    assert args.align_img_img == (bool(args.later_loss_coefs[0]) or bool(args.later_loss_coefs[1]))
        
    assert args.align_img_txt or args.align_img_img
    # ======= adapt args =======

    main_print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    main_print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # specify data loading
    if args.weak_aug:
        main_print('using weak augmentation')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    else:
        main_print('using strong augmentation')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    if args.downsample:
        main_print('add downsample augmentation')
        train_transform = transforms.Compose([
            transforms.RandomApply([DownSampleAndUpsample(down_res=args.down_res, p=args.down_prob)],
                                   p=args.downsample_prob),
            train_transform])

    train_dataset = SupconDataset(
        input_filename=args.csv_path,
        syn_input_filename=args.syn_csv_path,
        root_list=args.folder_list,
        syn_idx_list=args.syn_idx_list,
        root_suffix_list=args.folder_suffix_list,
        real_images_path_suffix=args.real_images_path_suffix,
        transforms=train_transform,
        num_views=args.n_img,
        num_crop=args.num_crop,
        tokenizer=args.tokenizer if args.align_img_txt else None,
        syn_ratio=args.syn_ratio,
        path1=args.path1,
        path2=args.path2,
        sample_mode=args.sample_mode
    )
    main_print(len(train_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=args.pin_mem, sampler=train_sampler, drop_last=True)

    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    n_views = len(args.folder_list)
    if args.sample_mode=='fixed+random':
        n_views = 2
    elif args.sample_mode=='random':
        n_views = 1

    if not args.align_img_txt: # only img-img
        model = v_model_dict[args.model](ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim)
        criterion = MultiPosConLoss(temperature=args.ssl_temp)
    if not args.align_img_img: # only img-txt
        model = vt_model_dict[args.model](ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim,
                                         vl_projection=args.vl_projection, image_mlp_enable=False)
        criterion = ImgTxtConLoss(temperature=args.ssl_temp)
    else:   # both
        model = vt_model_dict[args.model](ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim,
                                          vl_projection=args.vl_projection, image_mlp_enable=True)
        criterion = MultiPosConLossMM(temperature=args.ssl_temp)
    main_print(model)
    main_print(criterion)
    model = model.to(device)
    criterion = criterion.to(device)
    model_without_ddp = model

    if args.lr is None:  # only base_lr is specified
        eff_batch_size = args.batch_size * misc.get_world_size()
        args.lr = args.blr * eff_batch_size * args.n_img / 256
        args.lr = args.lr * args.num_crop / 2  # previous line assumes num_crop=2

    main_print("lr: %.3e" % args.lr)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=False)
        model_without_ddp = model.module

    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))
    loss_scaler = NativeScaler()

    # resume model if needed
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write("args:\n")
            f.write(json.dumps(vars(args), indent=4) + "\n")

    qd_module = None
    if args.early_loss_coefs[1] or args.early_loss_coefs[3] or args.later_loss_coefs[1] or args.later_loss_coefs[3]:
        qd_module = quality_driven_module(opt.standard_array_path)
        
    main_print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    args.cur_coefs = args.early_loss_coefs
    for epoch in range(args.start_epoch, args.epochs):            
        if args.early_stop_epoch is not None and epoch >= args.early_stop_epoch:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
        if epoch == args.epoch_switch:
            args.cur_coefs = args.later_loss_coefs
        main_print(f"epoch:{epoch}, cur_coefs:{args.cur_coefs}")
        
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, n_views, loss_scaler, criterion,
            log_writer=log_writer,
            qd_module=qd_module,
            args=args
        )
                
        if args.output_dir and (
            epoch % args.save_freq == 0
            or epoch + 1 == args.epochs
            or (args.save_epoch is not None and epoch in args.save_epoch)
        ):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, n_keep=args.n_keep)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        misc.write_log(log_stats, log_writer, args.output_dir)

        # always save the last model
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }
        checkpoint_path = os.path.join(args.output_dir, 'epoch_last.pth')
        misc.save_on_master(to_save, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    main_print('Training time {}'.format(total_time_str))


def distributed_exit():
    """
    Blocks until all processes reach this point, then safely exits.
    """
    if not dist.is_initialized():
        print("Distributed not initialized. Exiting...")
        sys.exit(0)

    dist.barrier()  # Wait for all ranks to reach this point
    if dist.get_rank() == 0:
        print("All processes reached barrier. Exiting now...")
    sys.exit(0)

    
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, n_views: int, loss_scaler, loss_fn=None,
                    log_writer=None,
                    qd_module=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    optimizer.zero_grad()

    # this is a pseudo label to index samples for loss function
    label_input = torch.arange(args.batch_size).to(device) + \
        args.batch_size * misc.get_rank()
    text_input = None

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header, args.output_dir)):
        loader_len = len(data_loader)
        misc.adjust_learning_rate(optimizer, data_iter_step / loader_len + epoch, args)
        
        img_input = data[0].to(device, non_blocking=True)   # bsz, n_views*3, 224, 224
        text_input = data[1].to(device, non_blocking=True) if (bool(args.cur_coefs[2]) or bool(args.cur_coefs[3])) else None
        
        QD_img_img = (args.cur_coefs[1]!=0)
        QD_img_txt = (args.cur_coefs[3]!=0)
        pure_real_img = data[2].to(device, non_blocking=True) if QD_img_txt else None
        
        with torch.cuda.amp.autocast():
            outputs = model(img_input, label_input, text_input)
        
        weight_dict = {"loss_weight":None, "loss_weight_img_txt":None}
        if QD_img_img or QD_img_txt:
            img_views = torch.chunk(img_input, chunks=n_views, dim=1) 
            kwargs = {
                'QD_img_img': QD_img_img,
                'QD_img_txt': QD_img_txt,
                'views_list': img_views,
                'labels':outputs['labels'] if 'labels' in outputs else outputs['image_labels'],
                'text': text_input,
                'image': pure_real_img,
                'text_labels':outputs.get('text_labels'),
                'gamma':{'gamma_ii':args.gamma_ii, 'gamma_it':args.gamma_it}
            }
            weight_dict = qd_module.compute_pairwise_weights(**kwargs)
        
        loss_dict = loss_fn(outputs, args.cur_coefs, **weight_dict)
        loss = loss_dict['loss']

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters())

        if grad_norm is not None and not math.isfinite(grad_norm):
            warning_msg = f"[Warning] Grad norm is {grad_norm}, skipping optimizer step at epoch {epoch}, iteration {data_iter_step}"
            print(warning_msg)
            misc.write_log({"epoch": epoch, "iteration":data_iter_step, "warning": warning_msg}, log_writer, args.output_dir)
            grad_norm = 0.0
            loss_value = 0.0
        
        metric_logger.update(grad_norm=grad_norm)
        optimizer.zero_grad()
        
        # clamp logit scale for image-text contrast
        if args.align_img_txt:
            misc.get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = misc.get_model(model).logit_scale.exp().item()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / loader_len + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if args.align_img_txt:
                log_writer.add_scalar('logit', logit_scale, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    main_print(("Averaged stats:", metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
