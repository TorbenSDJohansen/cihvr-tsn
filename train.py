#!/usr/bin/env python3
"""
@author: sa-tsdj
"""

import time
import os
import logging

from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch

from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch import nn

import torchvision.utils

# timm imports
from timm.data import (
    create_dataset,
    resolve_data_config,
    AugMixDataset,
    )
from timm.models import (
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
    convert_splitbn_model,
    model_parameters,
    )
from timm.utils import (
    CheckpointSaver,
    dispatch_clip_grad,
    ApexScaler, NativeScaler,
    distribute_bn, reduce_tensor,
    setup_default_logging,
    AverageMeter, accuracy,
    ModelEmaV2,
    random_seed,
    update_summary, get_outdir,
    )
from timm.loss import (
    LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy,
    )
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

# timmsn imports
from timmsn.loss import (
    SequenceCrossEntropy,
    LabelSmoothingSequenceCrossEntropy,
    SoftTargetSequenceCrossEntropy,
    )
from timmsn.utils import sequence_accuracy, initial_log
from timmsn.models import create_model
from timmsn.data import create_loader, Mixup, FastCollateMixup
from timmsn.data.parsers import setup_sqnet_parser, setup_bdatasets_parser

# other imports
from argparser import parse_args

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True # pylint: disable=C0103
except ImportError:
    has_apex = False # pylint: disable=C0103

has_native_amp = False # pylint: disable=C0103
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True # pylint: disable=C0103
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True # pylint: disable=C0103
except ImportError:
    has_wandb = False # pylint: disable=C0103

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train') # pylint: disable=C0103

def main(): # pylint: disable=R0914, R0912, R0915, C0116
    setup_default_logging()
    args, args_text = parse_args()

    if len(args.dataset) == 1:
        args.dataset = args.dataset[0]

    parser_train = parser_eval = clean_pred = None
    if isinstance(args.dataset, str) and args.dataset.startswith('bdatasets.'):
        assert args.sequence_model, 'bdatasets are for sequences'
        parser_train, parser_eval, clean_pred = setup_bdatasets_parser(
            args=args, purpose='train', split=args.data_split or 'train',
            )
    elif args.formatter is not None:
        assert args.sequence_model
        parser_train, parser_eval, clean_pred = setup_sqnet_parser(
            args=args, purpose='train', split=args.data_split or 'train',
            )

    if args.sequence_model:
        assert args.num_classes is not None
    else:
        assert args.num_classes is None or len(args.num_classes) == 1
        args.num_classes = None if args.num_classes is None else args.num_classes[0]

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' # pylint: disable=W1201
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    # See if implicit --resume is possible.
    if args.resume == '' and args.output != '' and args.experiment != '':
        potential_resume = os.path.join(args.output, args.experiment, 'last.pth.tar')
        if os.path.isfile(potential_resume):
            args.resume = potential_resume
            if args.local_rank == 0:
                print(f'--resume not specified, but continuing from experiment where checkpoint exists. Using "{args.resume}" to resume from.')

    # If resuming, no sense in first loading pretrained weights or initial
    # checkpoint and then overwriting with resume checkpoint.
    if args.resume:
        args.pretrained = False
        args.initial_checkpoint = False
    elif args.initial_checkpoint: # And no sense 2x load here either
        args.pretrained = False

    model = create_model(
        args.model,
        sqnet_version=args.sqnet_version,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        drop_modules=args.drop_modules)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.local_rank == 0:
        _logger.info( # pylint: disable=W1202
            f'Model {safe_model_name(args.model)} created, param count: {sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last) # pylint: disable=E1101

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != 'native':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {} (0-{})'.format(num_epochs, num_epochs- 1)) # pylint: disable=W1202

    # create the train and eval datasets
    dataset_train = create_dataset(
        parser_train or args.dataset,
        root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size, repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        parser_eval or args.dataset,
        root=args.data_dir, split=args.val_split, is_training=False,
        batch_size=args.batch_size)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        resize_method=args.resize_method,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        resize_method=args.resize_method,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # setup loss function
    if args.sequence_model:
        if mixup_active:
            train_loss_fn = SoftTargetSequenceCrossEntropy().cuda()
        elif args.smoothing:
            train_loss_fn = LabelSmoothingSequenceCrossEntropy(args.smoothing, args.sequence_model).cuda()
        else:
            train_loss_fn = SequenceCrossEntropy().cuda()
    elif args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()

    if args.sequence_model:
        validate_loss_fn = SequenceCrossEntropy().cuda()
    else:
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup accuracy function
    if args.sequence_model:
        validate_accuracy_fn = sequence_accuracy
    else:
        validate_accuracy_fn = accuracy

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f: # pylint: disable=C0103
            f.write(args_text)

    if args.log_wandb and args.local_rank == 0:
        if has_wandb:
            wandb.init(
                project=args.wandb_project_prefix + os.path.split(args.output)[-1],
                name=args.experiment,
                dir=output_dir,
                resume='auto',
                config=args,
                )
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    if args.initial_log:
        assert clean_pred is not None
        if args.rank == 0:
            initial_log(
                args, output_dir, loader_train, loader_eval, clean_pred,
                log_wandb=args.log_wandb and has_wandb,
                )

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            start_epoch_time = time.time()
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
            train_metrics['epoch_time'] = time.time() - start_epoch_time

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader=loader_eval, loss_fn=validate_loss_fn, args=args, amp_autocast=amp_autocast,
                                    accuracy_fn=validate_accuracy_fn)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)',
                    accuracy_fn=validate_accuracy_fn)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch)) # pylint: disable=W1202


def train_one_epoch( # pylint: disable=R0913, R0914, R0912, R0915, C0116
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader): # pylint: disable=W0622
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last) # pylint: disable=E1101

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl) # pylint: disable=C0103

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, last_idx,
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([
        ('loss', losses_m.avg),
        # ('total_batch_time', batch_time_m.sum), # Not particularly interesting, captured in epoch time
        ])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', accuracy_fn=accuracy): # pylint: disable=R0913, R0914, C0116
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    acc_metric_1_m = AverageMeter()
    acc_metric_2_m = AverageMeter()

    if args.sequence_model:
        acc_metric_1_name, acc_metric_2_name = 'SeqAcc', 'TokenAcc'
    else:
        acc_metric_1_name, acc_metric_2_name = 'Acc@1', 'Acc@5'

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader): # pylint: disable=W0622
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last) # pylint: disable=E1101

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)) and not args.sequence_model:
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                assert not args.sequence_model, 'TTA not implemented for sequence models'
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)

            if args.sequence_model:
                acc_metric_1, acc_metric_2 = accuracy_fn(output, target)
            else:
                acc_metric_1, acc_metric_2 = accuracy_fn(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc_metric_1 = reduce_tensor(acc_metric_1, args.world_size)
                acc_metric_2 = reduce_tensor(acc_metric_2, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            acc_metric_1_m.update(acc_metric_1.item(), target.size(0))
            acc_metric_2_m.update(acc_metric_2.item(), target.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    '{acc_metric_1_name}: {acc_metric_1_m.val:>7.4f} ({acc_metric_1_m.avg:>7.4f})  '
                    '{acc_metric_2_name}: {acc_metric_2_m.val:>7.4f} ({acc_metric_2_m.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, acc_metric_1_name=acc_metric_1_name,
                        acc_metric_2_name=acc_metric_2_name,
                        acc_metric_1_m=acc_metric_1_m, acc_metric_2_m=acc_metric_2_m))

    metrics = OrderedDict([
        ('loss', losses_m.avg),
        (acc_metric_1_name, acc_metric_1_m.avg),
        (acc_metric_2_name, acc_metric_2_m.avg),
        ])

    return metrics


def fake_validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', accuracy_fn=accuracy):
    if args.sequence_model:
        acc_metric_1_name, acc_metric_2_name = 'SeqAcc', 'TokenAcc'
    else:
        acc_metric_1_name, acc_metric_2_name = 'Acc@1', 'Acc@5'
    metrics = OrderedDict([
        ('loss', -1),
        (acc_metric_1_name, -1),
        (acc_metric_2_name, -1),
        ])

    return metrics


if __name__ == '__main__':
    main()
