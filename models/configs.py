from easydict import EasyDict as edict

from constants import SRC_DIR


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

dave2_parameters = edict(
        batch_size=64,
        num_workers=0,
        image_size=(120, 320),
        normalization=(0.5, 0.5),
        model_name='dave2',
        checkpoint=f'{SRC_DIR}/models/dave2/dave2.tar'
    )

transfer_parameters = edict(
        batch_size=1,
        num_workers=0,
        image_size=(120, 320),
        normalization=(0.5, 0.5),
        model_name='transfer',
        checkpoint=f'{SRC_DIR}/models/transfer/epoch_119.tar'
    )


DriverNet_parameters = edict(
        batch_size=1,
        num_workers=0,
        image_size=(66, 200),
        model_name='DriverNet',
        checkpoint=f'{SRC_DIR}/models/DriverNet/DriverNet.pth'
    )