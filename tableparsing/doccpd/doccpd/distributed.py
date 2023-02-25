import os
import torch
import torch.multiprocessing as mp
from PIL import Image
from pathlib import Path
from queue import Empty
import numpy as np


_IDX = 0


def ready_files(filedir, queue, event):
    image_list = list(Path(filedir).rglob(f'*.jpg'))

    while len(image_list) > 0:
        if queue.full():
            time.sleep(0.05)
            continue
        else:
            image_path = image_list.pop()
            image = Image.open(image_path).convert('RGB')
            queue.put((image, image_path.name))

    event.set()
    queue.join()


def find_keypoints(detector, in_queue, out_queue, file_event, keypoint_event, n_keypoints, device):

    while not (file_event.is_set() and in_queue.empty()):
        try:
            image, name = in_queue.get(block=True, timeout=0.1)
            image = np.array(image).astype(int)
        except Empty:
            continue

        with torch.no_grad():
            keypoints = detector.find_keypoints(image, n_keypoints)

        in_queue.task_done()
        out_queue.put((keypoints, image, name))

    keypoint_event.set()
    out_queue.join()


def point_drift(cropped_template, overlay, in_queue, keypoint_event, iterations, outdir, info_freq):
    from doccpd.pointdrift import PointDrift

    global _IDX

    while not (keypoint_event.is_set() and in_queue.empty()):
        try:
            keypoints, image, name = in_queue.get(block=True, timeout=0.1)
        except Empty:
            continue

        pdrift = PointDrift(cropped_template)
        pdrift.fit(keypoints, iterations=iterations, show_fit=False)

        transformed = pdrift.apply_transform(image)
        overlay.write_cells(transformed, name, outdir)

        _IDX += 1

        if _IDX % info_freq == 0:
            print(f'Processed {_IDX} documents...')

        in_queue.task_done()


def run_distributed(keypoint_detectors, 
                    template, 
                    overlay, 
                    pdrift_iters, 
                    n_keypoints,
                    filedir,
                    outdir,
                    info_freq,
                    n_cpus):
    print(f'Working using {len(keypoint_detectors)} keypoint CPUs and {n_cpus} PointDrift CPUs...')

    file_q = mp.JoinableQueue()
    keypoint_q = mp.JoinableQueue()
    file_event = mp.Event()
    keypoint_event = mp.Event()

    file_process = mp.Process(target=ready_files, args=(filedir, file_q, file_event))

    keypoint_processes = []

    for detector in keypoint_detectors:
        keypoint_processes.append(
                mp.Process(target=find_keypoints, args=(
                    detector, file_q, keypoint_q, file_event, keypoint_event, n_keypoints, detector._device)))

    pdrift_processes = [mp.Process(target=point_drift, args=(
        template, overlay, keypoint_q, keypoint_event, pdrift_iters, outdir, info_freq))
        for i in range(n_cpus)]

    file_process.start()

    for kp_process in keypoint_processes:
        kp_process.start()

    for pd_process in pdrift_processes:
        pd_process.start()

    for pd_process in pdrift_processes:
        pd_process.join()

    for kp_process in keypoint_processes:
        kp_process.join()

    file_process.join()
    keypoint_q.close()
    file_q.close()

    print('Finished run...')


def create_detectors(ver_ckpt, hor_ckpt, crop_info, n_cpus):
    from doccpd.keypoints import make_2D_detector

    device = torch.device('cpu')

    detectors = []
    for cpu in range(n_cpus):
        #device = torch.device(f'cuda:{gpu}')
        detectors.append(
                make_2D_detector(ver_ckpt, hor_ckpt, device=device, crop_info=crop_info))
    return detectors


def test_run():
    from doccpd.keypoints import make_2D_detector
    from doccpd.pointdrift import PointDrift
    from doccpd.template import Template, Overlay, crop_template

    crop_info = {
    'top': 0.2,
    'bot': 0.02,
    'left': 0.5,
    'right':0.1
    }

    template = Template.from_xml('../tests/fixtures/1916DKCensus_Template_rightside_13240230.xml', 10_000)
    cropped_template = crop_template(template, crop_info)

    overlay = Overlay.from_xml('../tests/fixtures/1916DKCensus_Overlay_income_wealth_tax__Occ_empl_13240230.xml')

    ckpt = 'ckpts/CP_epoch13.pth'

    detectors = create_detectors(ckpt, ckpt, crop_info, 2)
    print('Created detectors')

    run_distributed(detectors, cropped_template, overlay, 30, 'imgdir', 4)


if __name__ == '__main__':
    mp.set_start_method("spawn")

    N_DETECTORS = 2
    N_POINT_DRIFTERS = 2

    test_run()
    





















