import builtins
import datetime
import glob
import os
import time
import json
from collections import defaultdict, deque
from pathlib import Path
import math

import torch
import torch.distributed as dist
# import math
inf = math.inf


def adjust_ssl_temperature(epoch, args):
    """adjust the ssl temperature"""
    t = args.ssl_temp_max - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * \
        (args.ssl_temp_max - args.ssl_temp_min)
    return t


def adjust_weight_decay(optimizer, epoch, args):
    """chnage the weight decay with half-cycle cosine"""
    wd = args.weight_decay_end - (args.weight_decay_end - args.weight_decay) * \
        (1. + math.cos(math. pi * epoch / args.epochs)) / 2
    for i, param_group in enumerate(optimizer.param_groups):
        if param_group['weight_decay'] > 0:
            param_group['weight_decay'] = wd
    return wd


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) /
                           (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def add_weight_decay(model, weight_decay):
    """add weight decay, and skip biases and norm layers"""
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": weight_decay},
                    {"params": p_non_wd, "weight_decay": 0}]
    return optim_params


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, log_dir=None):
        i = 0
        try:
            iterable_len = len(iterable)
        except:
            iterable_len = iterable.num_batches
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(iterable_len))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        
        if log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file_path = os.path.join(log_dir, 'detailed_log.txt')
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == iterable_len - 1:
                eta_seconds = iter_time.global_avg * (iterable_len - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if is_main_process():
                    if torch.cuda.is_available():
                        msg = log_msg.format(
                            i, iterable_len, eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB)
                    else:
                        msg = log_msg.format(
                            i, iterable_len, eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time))
                    print(msg)
                    if log_dir:
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(msg + '\n')
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        if is_main_process():
            final_msg = '{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / iterable_len)
            print(final_msg)

            # Write final message to log file
            if log_dir:
                with open(log_file_path, 'a') as log_file:
                    log_file.write(final_msg + '\n')


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # 添加 Slurm 环境变量适配 
    if 'SLURM_PROCID' in os.environ: 
        import subprocess    
        proc_id = int(os.environ['SLURM_PROCID']) 
        ntasks = int(os.environ['SLURM_NTASKS']) 
        local_id = int(os.environ['SLURM_LOCALID'])
        os.environ['RANK'] = str(proc_id)
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['LOCAL_RANK'] = str(local_id)
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend,
                                         init_method=args.dist_url,
                                         world_size=args.world_size,
                                         rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None,
                 create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if not update_grad:
            return None
        self._scaler.unscale_(optimizer)
        
        if clip_grad is not None:
            assert parameters is not None
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            norm = get_grad_norm_(parameters)

        if norm is None or not math.isfinite(norm):
            print(f"[Warning] Detected invalid grad norm: {norm}, skipping optimizer update")
            self._scaler.update()
            return norm 
        
        self._scaler.step(optimizer)
        self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters
            ]),
            norm_type
        )
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, n_keep=10):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir,
                              tag="checkpoint-%s" % epoch_name,
                              client_state=client_state)

    # keep n_keep checkpoints
    if is_main_process():
        model_list = glob.glob(os.path.join(args.output_dir, 'checkpoint-*.pth'))
        if len(model_list) > n_keep:
            epochs = [os.path.basename(m).split('-')[-1].split('.')[0] for m in model_list]
            epochs = [int(e) for e in epochs]
            epochs.sort(reverse=True)
            for e in range(n_keep, len(epochs)):
                if ((args.save_epoch is not None) and (e in args.save_epoch)):
                    continue
                name = os.path.join(args.output_dir, 'checkpoint-%s.pth' % str(epochs[e]))
                cmd = 'rm %s' % name
                os.system(cmd)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if ('optimizer' in checkpoint and 'epoch' in checkpoint
                and not (hasattr(args, 'eval') and args.eval)):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def write_log(msg, log_writer, log_dir):
    if log_dir and is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(msg) + "\n")
            

def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_all, tensor, async_op=False)  # performance opt

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

