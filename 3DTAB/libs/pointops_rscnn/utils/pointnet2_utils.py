import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from .linalg_utils import pdist2, PDist2Order
from collections import namedtuple
from . import pytorch_utils as pt_utils
from . import my_point_utils as point_utils
from typing import List, Tuple

#from _ext import pointnet2
try:
    import _ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *



class RandomDropout(nn.Module):

    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(
            X, theta, self.train, self.inplace
        )


class FurthestPointSampling(Function):

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        """B, N, _ = xyz.size()

        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
        pointnet2.furthest_point_sampling_wrapper(
            B, N, npoint, xyz, temp, output
        )
        return output"""
        out = _ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(out)
        return out

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()

        #output = torch.cuda.FloatTensor(B, C, npoint)

        #pointnet2.gather_points_wrapper(
        #    B, C, N, npoint, features, idx, output
        #)

        ctx.for_backwards = (idx, C, N)

        #return output
        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        """idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(
            B, C, N, npoint, grad_out_data, idx, grad_features.data
        )"""
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor,
                known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        """
        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(
            ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        """output = torch.cuda.FloatTensor(B, c, n)

        pointnet2.three_interpolate_wrapper(
            B, c, m, n, features, idx, weight, output
        )

        return output"""
        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        #grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())

        #grad_out_data = grad_out.data.contiguous()
        #pointnet2.three_interpolate_grad_wrapper(
        #    B, c, n, m, grad_out_data, idx, weight, grad_features.data
        #)
        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None #torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of points to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of points to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        #output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        #pointnet2.group_points_wrapper(
        #    B, C, N, nfeatures, nsample, features, idx, output
        #)

        ctx.for_backwards = (idx, N)
        return _ext.group_points(features, idx)
        #return output

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        #B, C, npoint, nsample = grad_out.size()
        #grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())

        #grad_out_data = grad_out.data.contiguous()
        #pointnet2.group_points_grad_wrapper(
        #   B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data
        #)
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None # torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply


class BallQuery(Function):

    @staticmethod
    def forward(
            ctx, radius: float, nsample: int, xyz: torch.Tensor,
            new_xyz: torch.Tensor, fps_idx: torch.IntTensor
    ) -> torch.Tensor:
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        """B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_wrapper(
            B, N, npoint, radius, nsample, new_xyz, xyz, fps_idx, idx
        )
        """
        idx = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(idx)
        # return torch.cat([fps_idx.unsqueeze(2), idx], dim = 2)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of points to gather in the ball
    """

    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            features: torch.Tensor = None,
            fps_idx: torch.IntTensor = None
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz, fps_idx)
        idx = torch.cat([fps_idx.unsqueeze(2), idx], dim = 2)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(
            xyz_trans, idx
        )  # (B, 3, npoint, nsample)
        # xyz_flipped = xyz.permute(0,2,1).contiguous()
        # new_xyz_flipped = new_xyz.permute(0,2,1).contiguous()
        # idx = point_utils.query_ball_point(self.radius, self.nsample, xyz_flipped, new_xyz_flipped)
        # grouped_xyz = point_utils.index_points(xyz_flipped, idx) # (B, 3, npoint, nsample)
        raw_grouped_xyz = grouped_xyz
        grouped_xyz = grouped_xyz - new_xyz.transpose(1,2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            # grouped_features = point_utils.index_points(features, idx)
            if self.use_xyz:
                new_features = torch.cat([raw_grouped_xyz, grouped_xyz, grouped_features],
                                         dim=1)  # (B, C + 3 + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = torch.cat([raw_grouped_xyz, grouped_xyz], dim = 1)

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz: bool = True):
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            features: torch.Tensor = None
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
