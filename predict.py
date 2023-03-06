# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import os
import logging
import shutil

from contextlib import suppress
from functools import partial

import torch

import numpy as np
import pandas as pd

from timm.models import apply_test_time_pool
from timm.data import resolve_data_config, create_dataset
from timm.utils import setup_default_logging

# timmsn imports
from timmsn.utils import (
    predict_sequence_v2,
    MontageMaker,
    )
from timmsn.models import create_model
from timmsn.data import create_loader
from timmsn.data.parsers import (
    setup_sqnet_parser,
    setup_bdatasets_parser,
    setup_sqnet_parser_predict_folders,
    )

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
_logger = logging.getLogger('inference') # pylint: disable=C0103

def main():
    setup_default_logging()
    args, _ = parse_args()

    args.pretrained = args.pretrained or not args.checkpoint

    output_dir = args.output
    assert not os.path.isdir(output_dir), f'cannot write to already existing dir {output_dir}'

    if args.checkpoint == '':
        print('WARNING: Predicting without any checkpoint to load model from.')
    else:
        # If checkpoint, no sense in first loading pretrained weights and then
        # overwriting with resume checkpoint.
        args.pretrained = False

    if len(args.dataset) == 1:
        args.dataset = args.dataset[0]

    parser_predict = clean_pred = unpacked_folders = None
    if args.predict_folders is not None:
        parser_predict, clean_pred, unpacked_folders = setup_sqnet_parser_predict_folders(args)
    elif isinstance(args.dataset, str) and args.dataset.startswith('bdatasets.'):
        assert args.sequence_model, 'bdatasets are for sequences'
        parser_predict, _, clean_pred = setup_bdatasets_parser(
            args=args, purpose='predict', split=args.data_split or 'predict',
            )
    elif args.formatter is not None:
        assert args.sequence_model
        parser_predict, _, clean_pred = setup_sqnet_parser(
            args=args, purpose='predict', split=args.data_split or 'predict',
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

    # create model
    model = create_model(
        args.model,
        sqnet_version=args.sqnet_version,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        )

    _logger.info('Model %s created, param count: %d' % # pylint: disable=W1201
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

    model = model.cuda()

    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last) # pylint: disable=E1101

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    dataset = create_dataset(
        root=args.data_dir, name=parser_predict or args.dataset, split=args.val_split,
        load_bytes=args.tf_preprocessing, class_map=args.class_map)

    loader = create_loader(
        dataset,
        input_size=config['input_size'],
        batch_size=args.batch_size,
        resize_method=args.resize_method,
        use_prefetcher=args.prefetcher,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])

    # TODO add initial-log here. Need to rewrite function to work when only one
    # loader available and when labels are not available (i.e., omit montage).

    model.eval()

    preds, seq_prob, files, _, digit_probs = predict_sequence_v2(
        model=model,
        loader=loader,
        amp_autocast=amp_autocast,
        no_prefetcher=args.no_prefetcher,
        channels_last=args.channels_last,
        retrieve_labels=False,
        )

    if unpacked_folders is not None and not args.keep_unpacked:
        for unpacked_folder in unpacked_folders:
            print(f'Deleting earlier unpacked tarball "{unpacked_folder}".')
            shutil.rmtree(unpacked_folder)

    print('Cleaning predictions and labels.')
    clean_pred_accept_all = partial(clean_pred, assert_consistency=False)
    preds_clean = np.array(list(map(clean_pred_accept_all, preds)))

    pred_df = pd.DataFrame({
        'filename_full': files,
        'pred': preds_clean,
        'prob': seq_prob,
        })

    if args.return_individual_probs:
        for i, probs in enumerate(digit_probs.T):
            pred_df[f'prob_{i}'] = probs

    os.makedirs(output_dir)
    print(f'Writing output to "{output_dir}".')

    pred_df.to_csv(os.path.join(output_dir, 'preds.csv'), index=False)

    if args.plots is not None and 'montage' in args.plots:
        if has_cv2:
            montage_maker = MontageMaker(
                files=files,
                im_shape=tuple(args.input_size[-2:][::-1]),
                preds=preds_clean,
                seq_prob=seq_prob,
                targets=None,
                )
            montage = montage_maker.build_simple_montage((2, 10))
            cv2.imwrite(os.path.join(output_dir, 'montage.png'), montage) # pylint: disable=E1101
        else:
            _logger.warning('You have requested to create cv2 montage but package not found.')


if __name__ == '__main__':
    main()
