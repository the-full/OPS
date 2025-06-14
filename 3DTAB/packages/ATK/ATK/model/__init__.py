from .pointnet import PointNet
from .pointnet2 import PointNetPPMSG
from .pointnet2 import PointNetPPSSG
from .pointconv import PointConvDensitySSG
from .dgcnn import DGCNN
from .pointcnn import Classifier 
from .curvenet import CurveNet
from .pt_menghao import PointTransformerCls as PT1
from .pt_hengshuang import PointTransformerCls as PT2
from .point_pn import Point_PN
from .vn_pointnet import VNPointNet
from .vn_dgcnn import VNDGCNN
from .vn_transformer import InvariantClassifier
from .point_cat import PointCAT
from .rscnn_ssn import RSCNN_SSN
from .pointmlp import pointMLP, pointMLPElite
from .point_transformer_v1 import PointTransformerCls26, PointTransformerCls38, PointTransformerCls50
from .point_transformer_v2 import PointTransformerV2Cls
from .simpleview import MVModel
from .repsurf_ssg_umb import Model as RepSurf
from .repsurf_ssg_umb_2x import Model as RepSurf2
from .dgcnn_voxel import DGCNNVoxel


models = {
    'PointNet': PointNet,
    'PointNet++_MSG': PointNetPPMSG,
    'PointNet++_SSG': PointNetPPSSG,
    'PointConv': PointConvDensitySSG,
    'DGCNN': DGCNN,
    'PointCNN': Classifier,
    'CurveNet': CurveNet,
    'PT_Menghao': PT1,
    'PT_Hengshuang': PT2,
    'Point-PN': Point_PN,
    'VN-PointNet': VNPointNet,
    'VN-DGCNN': VNDGCNN,
    'VN-Transformer': InvariantClassifier,
    'PointCAT': PointCAT,
    'RSCNN': RSCNN_SSN,
    'PointMLP': pointMLP,
    'PointMLPElite': pointMLPElite,
    'PointTransformer-v1-Cls26': PointTransformerCls26,
    'PointTransformer-v1-Cls38': PointTransformerCls38,
    'PointTransformer-v1-Cls50': PointTransformerCls50,
    'PointTransformer-v2': PointTransformerV2Cls,
    'SimpleView': MVModel, # TODO: training result weirdo, need some time...
    'RepSurf': RepSurf,
    'RepSurf2': RepSurf2,
    'DGCNN-V': DGCNNVoxel,
}
