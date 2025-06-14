from .FGSM import FGSM
from .I_FGSM import IterateFGSM
from .MI_FGSM import MIFGSM 
from .NI_FGSM import NIFGSM
from .VMI_FGSM import VMIFGSM
from .VNI_FGSM import VNIFGSM
from .PGN import PGN
from .IR import IR
from .AI_FGTM import AIFGTM
from .GRA import GRA
from .SIM import SIM
from .OPS import OPS
from .GNP import GNP
from .IE_FGSM import IEFGSM
from .MIE_FGSM import MIEFGSM
from .EMI_FGSM import EMIFGSM
from .SMI_FGRM import SMIFGRM
from .VA_I_FGSM import VAIFGSM
from .MIG import MIG
from .GI_MI_FGSM import GIMIFGSM
from .I_FGSSM import IFGSSM
from .PC_I_FGSM import PCIFGSM
from .RAP import RAP
from .DTA import DTA
from .NCS import NCS
from .HiTADV import HiTADV
from .PFAttack import PFAttack
from .GeoA3 import GeoA3
from .kNN import KNN
from .AOF import AOF
from .AdvPC import AdvPC
from .SI_ADV import SIFGSM
from .attack_template import BasicAttack


attacks = {
    'FGSM': FGSM,
    'I-FGSM': IterateFGSM,
    'MI-FGSM': MIFGSM,
    'NI-FGSM': NIFGSM,
    'VMI-FGSM': VMIFGSM,
    'VNI-FGSM': VNIFGSM,
    'IR': IR,
    'AI-FGTM': AIFGTM,
    'GRA': GRA,
    'PGN': PGN,
    'SIM': SIM,
    'OPS': OPS,
    'GNP': GNP,
    'IE-FGSM': IEFGSM,
    'MIE-FGSM': MIEFGSM,
    'EMI-FGSM': EMIFGSM,
    'SMI-FGRM': SMIFGRM,
    'VA-I-FGSM': VAIFGSM,
    'MIG': MIG,
    'GI-MI-FGSM': GIMIFGSM,
    'I-FGS^2M': IFGSSM,
    'PC-I-FGSM': PCIFGSM,
    'RAP': RAP,
    'DTA': DTA,
    'NCS': NCS,
    'HiT-ADV': HiTADV,
    'PF-Attack': PFAttack,
    'GeoA3': GeoA3,
    'kNN': KNN,
    'AOF': AOF,
    'AdvPC': AdvPC,
    'SI-FGSM': SIFGSM,
    'none': BasicAttack,
}
