# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import os
import logging
import shutil

from functools import partial
from contextlib import suppress

import torch

import numpy as np
import pandas as pd

from timm.models import apply_test_time_pool, safe_model_name, load_checkpoint
from timm.data import resolve_data_config, create_dataset
from timm.utils import setup_default_logging

# timmsn imports
from timmsn.utils import (
    predict_sequence_v2,
    MontageMaker,
    PredictSequenceAR,
    PredictSequenceCTC,
    DecoderFactory,
    )
from timmsn.models import create_model_v2
from timmsn.data import create_loader
from timmsn.data.parsers import (
    setup_sqnet_parser,
    setup_bdatasets_parser,
    setup_sqnet_parser_predict_folders,
    )
from timmsn.data.formatters.constants import PAD_IDX, BOS_IDX, EOS_IDX

# other imports
from argparser import parse_args

has_cv2 = False # pylint: disable=C0103
try:
    import cv2
    has_cv2 = True # pylint: disable=C0103
except ImportError:
    pass

has_amp = False # pylint: disable=C0103
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_amp = True # pylint: disable=C0103
except AttributeError:
    pass


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference') # pylint: disable=C0103

def main(): # pylint: disable=R0914, R0912, R0915
    setup_default_logging()
    args, _ = parse_args()

    args.pretrained = args.pretrained or not args.checkpoint

    output_dir = args.output
    assert not os.path.isdir(output_dir), f'cannot write to already existing dir {output_dir}'

    if args.checkpoint == '':
        _logger.warning('Predicting without any checkpoint to load model from.')
    else:
        # If checkpoint, no sense in first loading pretrained weights and then
        # overwriting with resume checkpoint.
        args.pretrained = False

    if len(args.dataset) == 1:
        args.dataset = args.dataset[0]

    parser_predict = clean_pred = unpacked_folders = None
    if args.predict_folders is not None:
        parser_predict, clean_pred, unpacked_folders, num_classes = setup_sqnet_parser_predict_folders(
            formatter_name=args.formatter,
            predict_folders=args.predict_folders,
            predict_on_subfolders=args.predict_on_subfolders,
            num_classes=args.num_classes,
            verbose=args.local_rank,
            formatter_kwargs=args.formatter_kwargs,
            )
    elif isinstance(args.dataset, str) and args.dataset.startswith('bdatasets.'):
        parser_predict, _, clean_pred, num_classes = setup_bdatasets_parser(
            purpose='predict',
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
        parser_predict, _, clean_pred, num_classes = setup_sqnet_parser(
            purpose='predict',
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

    # setup automatic mixed-precision (AMP) op casting
    amp_autocast = suppress # do nothing
    if args.amp and has_amp:
        amp_autocast = torch.cuda.amp.autocast
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Predicting in mixed precision.')
    elif args.amp:
        if args.local_rank == 0:
            _logger.warning("AMP is not available, using float32. Upgrade to PyTorch>=1.6")
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Predicting in float32.')

    # create model
    model = create_model_v2(
        feature_extractor_name=args.model,
        sqnet_version=args.sqnet_version,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript,
        image_size=args.input_size,
        classifier_name=args.classifier,
        )

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    _logger.info('Model {} created with classifier {}, param count: {}'.format( # pylint: disable=W1202
        safe_model_name(args.model),
        args.classifier,
        sum([m.numel() for m in model.parameters()])),
        )

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if not args.no_test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    model = model.cuda()

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last) # pylint: disable=E1101

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    dataset = create_dataset(
        root=args.data_dir, name=parser_predict or args.dataset, split=args.val_split,
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

    model.eval()

    if args.seq2seq:
        decoder_type = args.decoder.split('-')[0] if args.decoder else 'greedy'

        beam_size = 3

        if decoder_type == 'beam' and '-' in args.decoder:
            beam_size = int(args.decoder.split('-')[1])

        # Note that when `decoder_type` is, e.g., greedy, `beam_size` is
        # properly discarded/not used in `DecoderFactory`
        decoder = DecoderFactory(
            decoder_type=decoder_type,
            max_len=len(args.num_classes),
            start_symbol=BOS_IDX,
            end_symbol=EOS_IDX,
            beam_size=beam_size,
            pad_sequences=True,
            pad_idx=PAD_IDX,
            )

        predict_fn = PredictSequenceAR(
            decoder=decoder,
            decoder_is_elementwise=decoder_type != 'greedy',
            )
    elif args.ctc:
        predict_fn = PredictSequenceCTC( # TODO change CTC formatter to use the global IDXs etc., right now messy
            blank=model.blank,
            max_len=model.width,
            pad_idx=max(args.num_classes) - 1,
            )
    else:
        predict_fn = predict_sequence_v2

    preds, seq_prob, files, _, digit_probs = predict_fn(
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
