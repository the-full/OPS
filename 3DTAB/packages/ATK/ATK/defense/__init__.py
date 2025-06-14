from .DropPoint import SORDefense, SRSDefense
from .DUP_Net import DUPNet
from .IF_Defense import ONetMesh, ONetOpt, ConvONetOpt
from .basic_defense import BasicDefense


defenses = {
    'SOR': SORDefense,
    'SRS': SRSDefense,
    'DUPNet': DUPNet,
    'ONet-Remesh': ONetMesh,
    'ONet-Optimize': ONetOpt,
    'ConvONet-Optimize': ConvONetOpt,
    'none': BasicDefense,
}
