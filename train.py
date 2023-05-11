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
    )
from timm.utils import (
    NativeScaler,
    distribute_bn,
    setup_default_logging,
    ModelEmaV2,
    random_seed,
    get_outdir,
    )
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

# timmsn imports
from timmsn.loss import (
    SequenceCrossEntropy,
    LabelSmoothingSequenceCrossEntropy, # TODO consider using v2
    SoftTargetSequenceCrossEntropy,
    CTCLoss,
    Seq2SeqCrossEntropy,
    LabelSmoothingSeq2SeqCrossEntropy,
    )
from timmsn.utils import (
    sequence_accuracy,
    CTCSeqAcc,
    Seq2SeqAccuracy,
    initial_log,
    update_summary,
    wandb_init,
    CheckpointSaver,
    )
from timmsn.models import create_model_v2
from timmsn.data import create_loader, Mixup, FastCollateMixup
from timmsn.data.parsers import (
    setup_sqnet_parser,
    setup_bdatasets_parser,
    setup_tarball_parser,
    )
from timmsn.data.formatters.constants import PAD_IDX

# other imports
from argparser import parse_args
from engine import train_one_epoch, validate

has_amp = False # pylint: disable=C0103
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_amp = True # pylint: disable=C0103
except AttributeError:
    pass

try:
    # want to do import to set has_wandb even if not used directly
    import wandb # pylint: disable=W0611
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
        parser_train, parser_eval, clean_pred, num_classes = setup_bdatasets_parser(
            purpose='train',
            dataset=args.dataset,
            data_dir=args.data_dir,
            evalset=args.evalset,
            formatter_name=args.formatter,
            num_classes=args.num_classes,
            split=args.data_split,
            verbose=args.local_rank,
            formatter_kwargs=args.formatter_kwargs,
            )
    elif args.formatter is not None:
        setup_fn = setup_tarball_parser if args.read_from_tar else setup_sqnet_parser

        parser_train, parser_eval, clean_pred, num_classes = setup_fn(
            purpose='train',
            dataset=args.dataset,
            data_dir=args.data_dir,
            dataset_structure=args.dataset_structure,
            evalset=args.evalset,
            formatter_name=args.formatter,
            num_classes=args.num_classes,
            split=args.data_split,
            verbose=args.local_rank,
            formatter_kwargs=args.formatter_kwargs,
            dataset_cells=args.dataset_cells,
            dataset_cells_eval=args.dataset_cells_eval,
            labels_subdir=args.labels_subdir,
            )

    args.num_classes = num_classes
    assert args.num_classes is not None

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

    model = create_model_v2(
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
        drop_modules=args.drop_modules,
        image_size=args.input_size,
        classifier_name=args.classifier,
        tl_from_input_size=args.tl_from_input_size,
        )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.local_rank == 0:
        _logger.info('Model {} created with classifier {}, param count: {}'.format( # pylint: disable=W1202
            safe_model_name(args.model),
            args.classifier,
            sum([m.numel() for m in model.parameters()])),
            )

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
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress # do nothing
    loss_scaler = None
    if args.amp and has_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    elif args.amp:
        if args.local_rank == 0:
            _logger.warning("AMP is not available, using float32. Upgrade to PyTorch>=1.6")
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
        _logger.info('Scheduled epochs: {} ({}-{})'.format(num_epochs - start_epoch, start_epoch, num_epochs - 1)) # pylint: disable=W1202

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
        use_multi_epochs_loader=args.use_multi_epochs_loader,
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

    if mixup_active and args.ctc:
        raise ValueError('Mixup not implemented for ctc models')
    if args.ctc and args.seq2seq:
        raise ValueError('Not possible to enable CTC and seq2seq at once')
    if args.ctc and args.smoothing:
        _logger.warning('WARNING: CTC and smoothing enabled, but smoothing not used with CTC')

    # setup loss function
    if mixup_active and args.seq2seq:
        raise NotImplementedError
        # Recall challenges related to need to keep unmixed targets, and in
        # particular how this is difficult with pre-fetching

    if mixup_active:
        train_loss_fn = SoftTargetSequenceCrossEntropy().cuda()
    elif args.ctc:
        # FIXME does not work with --amp
        # -> RuntimeError: "ctc_loss_cuda" not implemented for 'Half'
        # https://discuss.pytorch.org/t/ctc-loss-ctc-loss-not-support-float16/148800
        # Note that train loop works, but the `loss = loss_fn(output, target)`
        # part of validate breaks, as it is not placed within the context of
        # the `with amp_autocast():` block due to the conditional TTA in
        # between.
        train_loss_fn = CTCLoss(
            blank_label=model.blank,
            width=model.width,
            num_classes=args.num_classes,
            ).cuda()
    elif args.seq2seq and args.smoothing:
        train_loss_fn = LabelSmoothingSeq2SeqCrossEntropy(
            pad_idx=PAD_IDX, smoothing=args.smoothing,
            ).cuda()
    elif args.seq2seq and not args.smoothing:
        train_loss_fn = Seq2SeqCrossEntropy(pad_idx=PAD_IDX).cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingSequenceCrossEntropy(args.smoothing, True).cuda()
    else:
        train_loss_fn = SequenceCrossEntropy().cuda()

    # setup validate metrics
    if args.ctc:
        validate_loss_fn = train_loss_fn
        validate_accuracy_fn = CTCSeqAcc(model.blank)
    elif args.seq2seq:
        validate_loss_fn = Seq2SeqCrossEntropy(pad_idx=PAD_IDX).cuda()
        validate_accuracy_fn = Seq2SeqAccuracy(PAD_IDX)
    else:
        validate_loss_fn = SequenceCrossEntropy().cuda()
        validate_accuracy_fn = sequence_accuracy

    # setup checkpoint saver and eval metric tracking
    eval_metric = 'eval_' + args.eval_metric
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
        decreasing = eval_metric == 'loss'
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f: # pylint: disable=C0103, W1514
            f.write(args_text)

    if args.log_wandb and args.local_rank == 0:
        if has_wandb:
            wandb_init(
                output_dir=output_dir,
                project=args.wandb_project_prefix + os.path.split(args.output)[-1],
                name=args.experiment,
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
                log_wandb=args.log_wandb and has_wandb, mixup_fn=mixup_fn,
                )

    try:
        for epoch in range(start_epoch, num_epochs):
            start_epoch_time = time.time()

            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                logger=_logger, lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader=loader_eval, loss_fn=validate_loss_fn, args=args, amp_autocast=amp_autocast,
                                    accuracy_fn=validate_accuracy_fn, logger=_logger)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)',
                    accuracy_fn=validate_accuracy_fn, logger=_logger)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            epoch_time = time.time() - start_epoch_time

            metrics = OrderedDict()
            metrics.update(train_metrics)
            metrics.update(eval_metrics)
            metrics.update({'epoch_time': epoch_time})

            if output_dir is not None:
                update_summary(
                    epoch, metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = -1 / (epoch + 1) if args.skip_validate else eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch)) # pylint: disable=W1202


if __name__ == '__main__':
    main()
