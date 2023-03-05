# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import os
import logging

from functools import partial
from contextlib import suppress

import numpy as np
import pandas as pd

import torch
import torch.nn.parallel

from timm.models import (
    apply_test_time_pool,
    load_checkpoint,
    )
from timm.data import (
    create_dataset,
    resolve_data_config,
    )
from timm.utils import (
    setup_default_logging,
    set_jit_legacy,
    )

# timmsn imports
from timmsn.utils import (
    predict_sequence_v2,
    multiple_coverage_acc_plot,
    multiple_certainty_acc_plot,
    MontageMaker,
    )
from timmsn.models import create_model
from timmsn.data import create_loader
from timmsn.data.parsers import setup_sqnet_parser, setup_bdatasets_parser

# other imports
from argparser import parse_args

has_cv2 = False # pylint: disable=C0103
try:
    import cv2
    has_cv2 = True # pylint: disable=C0103
except ImportError:
    pass

has_apex = False # pylint: disable=C0103
try:
    from apex import amp
    has_apex = True # pylint: disable=C0103
except ImportError:
    pass

has_native_amp = False # pylint: disable=C0103
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True # pylint: disable=C0103
except AttributeError:
    pass


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate') # pylint: disable=C0103

def validate(args): # pylint: disable=C0116, R0914, R0912, R0915
    args.pretrained = args.pretrained or not args.checkpoint

    output_dir = args.output
    assert not os.path.isdir(output_dir), f'cannot write to already existing dir {output_dir}'

    if args.checkpoint == '':
        print('WARNING: Evaluating without any checkpoint to load model from.')
    else:
        # If checkpoint, no sense in first loading pretrained weights and then
        # overriding with checkpoint.
        args.pretrained = False

    if len(args.dataset) == 1:
        args.dataset = args.dataset[0]

    parser_eval = clean_pred = None
    if isinstance(args.dataset, str) and args.dataset.startswith('bdatasets.'):
        assert args.sequence_model, 'bdatasets are for sequences'
        parser_eval, _, clean_pred = setup_bdatasets_parser(
            args=args, purpose='test', split=args.data_split or 'test',
            )
    elif args.formatter is not None:
        assert args.sequence_model
        parser_eval, _, clean_pred = setup_sqnet_parser(
            args=args, purpose='test', split=args.data_split or 'test',
            )

    if args.sequence_model:
        assert args.num_classes is not None
    else:
        assert args.num_classes is None or len(args.num_classes) == 1
        args.num_classes = None if args.num_classes is None else args.num_classes[0]

    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        sqnet_version=args.sqnet_version,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint: #
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count)) # pylint: disable=W1201

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if not args.no_test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()

    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last) # pylint: disable=E1101

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    dataset = create_dataset(
        root=args.data_dir, name=parser_eval or args.dataset, split=args.val_split,
        load_bytes=args.tf_preprocessing, class_map=args.class_map)

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        resize_method=args.resize_method,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    # TODO add initial-log here. Need to rewrite function to work when only one
    # loader available.

    model.eval()

    preds, seq_prob, files, labels, digit_probs = predict_sequence_v2(
        model=model,
        loader=loader,
        amp_autocast=amp_autocast,
        no_prefetcher=args.no_prefetcher,
        channels_last=args.channels_last,
        retrieve_labels=True,
        )

    print('Cleaning predictions and labels.')
    clean_pred_accept_all = partial(clean_pred, assert_consistency=False)
    preds_clean = np.array(list(map(clean_pred_accept_all, preds)))
    labels_clean = np.array(list(map(clean_pred, labels.astype(int))))

    pred_df = pd.DataFrame({
        'filename_full': files,
        'label': labels_clean,
        'pred': preds_clean,
        'prob': seq_prob,
        })

    acc = 100 * (pred_df['label'] == pred_df['pred']).mean()
    print(f'Accuracy: {acc}%.')

    if args.return_individual_probs:
        for i, probs in enumerate(digit_probs.T):
            pred_df[f'prob_{i}'] = probs

    os.makedirs(output_dir, exist_ok=False)
    print(f'Writing output to "{output_dir}".')

    pred_df.to_csv(os.path.join(output_dir, 'preds.csv'), index=False)

    if args.plots is not None and ('cov-acc' in args.plots or 'cer-acc' in args.plots):
        eval_dfs = {'full': pred_df,}
        if args.eval_plots_omit_most_occ > 0:
            vals, counts = np.unique(labels_clean, return_counts=True)
            sort_occ = pd.DataFrame({'vals': vals, 'counts': counts}).sort_values(
                'counts', ascending=False,
                )['vals'].values
            eval_dfs = {
                **eval_dfs,
                **{f'excluding {tuple(sort_occ[0:i])}': pred_df[~pred_df['label'].isin(sort_occ[:i])]
                   for i in range(1, 1 + args.eval_plots_omit_most_occ)}
                }
            eval_dfs = {k: v for k, v in eval_dfs.items() if len(v) > 0}

    if args.plots is not None and 'cov-acc' in args.plots:
        multiple_coverage_acc_plot(
            eval_dfs=eval_dfs,
            fn_cov_acc_plot=os.path.join(output_dir, 'cov_acc.png'),
            )

    if args.plots is not None and 'cer-acc' in args.plots:
        multiple_certainty_acc_plot(
            eval_dfs=eval_dfs,
            fn_cer_acc_plot=os.path.join(output_dir, 'cer_acc.png'),
            )

    if args.plots is not None and 'montage' in args.plots:
        if has_cv2:
            montage_maker = MontageMaker(
                files=files,
                im_shape=tuple(args.input_size[-2:][::-1]),
                preds=preds_clean,
                seq_prob=seq_prob,
                targets=labels_clean,
                )
            montage = montage_maker.build_correct_incorrect_montage(10)
            cv2.imwrite(os.path.join(output_dir, 'montage.png'), montage) # pylint: disable=E1101
        else:
            _logger.warning('You have requested to create cv2 montage but package not found.')

def main():
    setup_default_logging()
    args, _ = parse_args()

    validate(args)


if __name__ == '__main__':
    main()
