# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import time
import os
import argparse
import logging

from typing import Iterable, Callable

from collections import OrderedDict
from contextlib import suppress

import torch

import torchvision.utils

from timm.models import model_parameters
from timm.utils import (
    AverageMeter,
    dispatch_clip_grad,
    reduce_tensor,
    )

from timmsn.utils import create_mask
from timmsn.data.formatters.constants import PAD_IDX


def train_one_epoch( # pylint: disable=R0913, R0914, R0912, R0915, C0116
        epoch: int,
        model: torch.nn.Module,
        loader: Iterable,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        args: argparse.Namespace,
        logger: logging.Logger,
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_ema=None,
        mixup_fn=None,
        ):
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
            if args.seq2seq:
                # TODO perhaps add kwargs and pass on that is empty if not seq2seq
                # Note how the first token of each sequence (BoS) is not incldued.

                target_input = target[:, :-1] # FIXME when using mixup we need
                # to keep the unmixed targets to use seq2seq modelling, which
                # is not possible at current when using prefetching, and also
                # requires saving the unmixed ones even when not using
                # prefetching
                target_mask, target_padding_mask = create_mask(target_input, PAD_IDX, 'cuda')
                output = model(input, target_input, target_mask, target_padding_mask)
            else:
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
                logger.info(
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
        ('train_loss', losses_m.avg),
        ('train_time', batch_time_m.sum),
        ('train_data_time', data_time_m.sum),
        ('train_model_time', batch_time_m.sum - data_time_m.sum),
        ])


def validate(
        model: torch.nn.Module,
        loader: Iterable,
        loss_fn: torch.nn.Module,
        args: argparse.Namespace,
        accuracy_fn: Callable,
        logger: logging.Logger,
        amp_autocast=suppress,
        log_suffix='',
        ): # pylint: disable=R0913, R0914, C0116
    acc_metric_1_name, acc_metric_2_name = 'SeqAcc', 'TokenAcc'

    if args.skip_validate:
        metrics = OrderedDict([
            ('eval_loss', None),
            ('eval_' + acc_metric_1_name, None),
            ('eval_' + acc_metric_2_name, None),
            ])

        return metrics

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    acc_metric_1_m = AverageMeter()
    acc_metric_2_m = AverageMeter()

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
                if args.seq2seq:
                    # TODO perhaps add kwargs and pass one that is empty if not seq2seq
                    # Note how the first token of each sequence (BoS) is not incldued.
                    target_input = target[:, :-1]
                    target_mask, target_padding_mask = create_mask(target_input, PAD_IDX, 'cuda')
                    output = model(input, target_input, target_mask, target_padding_mask)
                else:
                    output = model(input)

            # TODO what to do with below for, e.g., DeIT?
            # if isinstance(output, (tuple, list)) and not args.sequence_model:
            #     output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                # TODO keep below to remind later impl.?
                # assert not args.sequence_model, 'TTA not implemented for sequence models'
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)

            if args.seq2seq:
                # If Seq2Seq model skip initial BoS token for each sequence
                acc_metric_1, acc_metric_2 = accuracy_fn(output, target[:, 1:])
            else:
                acc_metric_1, acc_metric_2 = accuracy_fn(output, target)

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
                logger.info(
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
        ('eval_loss', losses_m.avg),
        ('eval_' + acc_metric_1_name, acc_metric_1_m.avg),
        ('eval_' + acc_metric_2_name, acc_metric_2_m.avg),
        ])

    return metrics
