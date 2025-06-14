import sys
import os
import os.path as osp
from collections import namedtuple

_cur_dir = osp.dirname(osp.abspath(__file__))
os.environ["ATK_ASSET_PATH"] = str(osp.join(_cur_dir, '../../../asset/'))

from .model import *
from .data import *
from .attack import *
from .defense import *
from .evaluator import *


REGISTER_TABLE = namedtuple(
    'REGISTER_TABLE', 
    'models datasets datamodules attacks defenses evaluators hooks'
)

register_table = REGISTER_TABLE(
    models=models,
    datasets=datasets,
    datamodules=datamodules,
    attacks=attacks,
    defenses=defenses,
    evaluators=evaluators,
    hooks=hooks,
)
