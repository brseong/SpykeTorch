from typing import Annotated, Any
import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
from jaxtyping import Float, Int, UInt8
from .utils import to_pair


# padding
# pad = (padLeft, padRight, padTop, padBottom)
def pad(input: torch.Tensor, pad: tuple[int, int, int, int], value: int = 0):
    r"""Applies 2D padding on the input tensor.

    Args:
        input (Tensor): The input tensor.
        pad (tuple): A tuple of 4 integers in the form of (padLeft, padRight, padTop, padBottom)
        value (int or float): The value of padding. Default: 0

    Returns:
        Tensor: Padded tensor.
    """
    return fn.pad(input, pad, value=value)


# pooling
def pooling(
    input: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: None | int | tuple[int, int] = None,
    padding: int | tuple[int, int] = 0,
):
    r"""Performs a 2D max-pooling over an input signal (spike-wave or potentials) composed of several input
    planes.

    Args:
        input (Tensor): The input tensor.
        kernel_size (int or tuple): Size of the pooling window.
        stride (int or tuple, optional): Stride of the pooling window. Default: None
        padding (int or tuple, optional): Size of the padding. Default: 0

    Returns:
        Tensor: The result of the max-pooling operation.
    """
    return fn.max_pool2d(input, kernel_size, stride, padding)


def fire(
    potentials: Float[torch.Tensor, "T C H W"], threshold: float | None = None
) -> tuple[Int[torch.Tensor, "T C H W"], Float[torch.Tensor, "T C H W"]]:
    r"""Computes the spike-wave tensor from tensor of potentials. If :attr:`threshold` is :attr:`None`, all the neurons
    emit one spike (if the potential is greater than zero) in the last time step.

    Args:
        potentials (torch.Tensor): The tensor of input potentials. Shape: (T, C, H, W)
        threshold (float): Firing threshold. Default: None

    Returns:
        spikes_and_thresholded (tuple[torch.Tensor, torch.Tensor]): The tensor of spike-wave and the tensor of thresholded potentials.
        each of shape (T, C, H, W).
    """
    thresholded = potentials.detach().clone()
    if threshold is None:
        thresholded[:-1] = 0
    else:
        # change the potentials to 0 if they are less than the threshold.
        fn.threshold(thresholded, threshold, 0, inplace=True)
    return thresholded.sign(), thresholded


def fire_(potentials: torch.Tensor, threshold: bool | None = None):
    r"""The inplace version of :func:`~fire`"""
    if threshold is None:
        potentials[:-1] = 0
    else:
        fn.threshold(potentials, threshold, 0, inplace=True)
    potentials.sign_()


def threshold(potentials, threshold=None):
    r"""Applies a threshold on potentials by which all of the values lower or equal to the threshold becomes zero.
    If :attr:`threshold` is :attr:`None`, only the potentials corresponding to the final time step will survive.

    Args:
        potentials (Tensor): The tensor of input potentials.
        threshold (float): The threshold value. Default: None

    Returns:
        Tensor: Thresholded potentials.
    """
    outputs = potentials.clone().detach()
    if threshold is None:
        outputs[:-1] = 0
    else:
        fn.threshold_(outputs, threshold, 0)
    return outputs


def threshold_(potentials, threshold=None):
    r"""The inplace version of :func:`~threshold`"""
    if threshold is None:
        potentials[:-1] = 0
    else:
        fn.threshold_(potentials, threshold, 0)


# in each position, the most fitted feature will survive (first earliest spike then maximum potential)
# it is assumed that the threshold function is applied on the input potentials
def pointwise_inhibition(
    thresholded_potentials: Float[torch.Tensor, "T C H W"],
) -> Float[torch.Tensor, "T C H W"]:
    r"""Performs point-wise inhibition between feature maps. After inhibition, at most one neuron is allowed to fire at each
    position, which is the neuron with the earliest spike time. If the spike times are the same, the neuron with the maximum
    potential will be chosen. As a result, the potential of all of the inhibited neurons will be reset to zero.

    Args:
        thresholded_potentials (Tensor): The tensor of thresholded input potentials.

    Returns:
        Tensor: Inhibited potentials.
    """
    # maximum of each position in each time step
    # brings the maximum value and its channel index
    max_values, max_indices = torch.max(
        thresholded_potentials, dim=1, keepdim=True
    )  # (T, 1, H, W), (T, 1, H, W)
    print("max_values", max_values.shape)
    print("max_indices", max_indices.shape)
    # compute signs for detection of the earliest spike
    clamp_pot = max_values.sign()  # T, 1, H, W
    print("clamp_pot", clamp_pot.shape)
    # maximum of clamped values is the indices of the earliest spikes
    clamp_pot_max_1 = (
        clamp_pot.size(0) - clamp_pot.sum(dim=0, keepdim=True)
    ).long()  # (1, 1, H, W) Tensor range from 0 to T
    print("clamp_pot_max_1", clamp_pot_max_1.shape)
    clamp_pot_max_1.clamp_(0, clamp_pot.size(0) - 1)  # clamp to valid range
    clamp_pot_max_0 = clamp_pot[-1:, :, :, :]  # (1, 1, H, W), the last time step

    # finding winners (maximum potentials between early spikes)
    # Since max_indices stores the index of the channel with the maximum value, which is clamped.
    # for each position, the number of the time steps which have >= 0 potential becomes the winner channel index.
    # This means that the channel with the maximum value in the earliest time step will be the winner.
    winners = max_indices.gather(
        0, clamp_pot_max_1
    )  # (1, 1, H, W) as torch.gather output is the same shape as index
    print("winners", winners.shape)
    # generating inhibition coefficient
    coef = torch.zeros_like(thresholded_potentials[0]).unsqueeze_(0)
    coef.scatter_(1, winners, clamp_pot_max_0)
    # applying inhibition to potentials (broadcasting multiplication)
    return torch.mul(thresholded_potentials, coef)


# inhibiting particular features, preventing them to be winners
# inhibited_features is a list of features numbers to be inhibited
def feature_inhibition_(potentials, inhibited_features):
    r"""The inplace version of :func:`~feature_inhibition`"""
    if len(inhibited_features) != 0:
        potentials[:, inhibited_features, :, :] = 0


def feature_inhibition(potentials, inhibited_features):
    r"""Inhibits specified features (reset the corresponding neurons' potentials to zero).

    Args:
        potentials (Tensor): The tensor of input potentials.
        inhibited_features (List): The list of features to be inhibited.

    Returns:
        Tensor: Inhibited potentials.
    """
    potentials_copy = potentials.clone().detach()
    if len(inhibited_features) != 0:
        feature_inhibition_(potentials_copy, inhibited_features)
    return potentials_copy


# returns list of winners
# inhibition_radius is to increase the chance of diversity among features (if needed)
def get_k_winners(
    potentials: torch.Tensor,
    kwta: int = 1,
    inhibition_radius: int = 0,
    spikes: torch.Tensor | None = None,
) -> list[tuple[int, int, int]]:
    r"""Finds at most :attr:`kwta` winners first based on the earliest spike time, then based on the maximum potential.
    It returns a list of winners, each in a tuple of form (feature, row, column).

    .. note::

        Winners are selected sequentially. Each winner inhibits surrounding neruons in a specific radius in all of the
        other feature maps. Note that only one winner can be selected from each feature map.

    Args:
        potentials (Tensor): The tensor of input potentials.
        kwta (int, optional): The number of winners. Default: 1
        inhibition_radius (int, optional): The radius of lateral inhibition. Default: 0
        spikes (Tensor, optional): Spike-wave corresponding to the input potentials. Default: None

        maximum (Tensor): The tensor of maximum values.

    Returns:
        List: List of winners.
    """
    if spikes is None:
        spikes = potentials.sign()
    # finding earliest potentials for each position in each feature
    maximum = (spikes.size(0) - spikes.sum(dim=0, keepdim=True)).long()
    maximum.clamp_(0, spikes.size(0) - 1)
    values = potentials.gather(dim=0, index=maximum)  # gathering values
    # propagating the earliest potential through the whole timesteps
    truncated_pot = spikes * values
    # summation with a high enough value (maximum of potential summation over timesteps) at spike positions
    v = truncated_pot.max() * potentials.size(0)
    truncated_pot.addcmul_(spikes, v)
    # summation over all timesteps
    total = truncated_pot.sum(
        dim=0, keepdim=True
    )  #:`total` is the sum of potentials over all timesteps

    total.squeeze_(0)
    global_pooling_size = tuple(total.size())
    winners = []
    for _ in range(kwta):
        max_val, max_idx = total.view(-1).max(0)
        if max_val.item() != 0:
            # finding the 3d position of the maximum value
            max_idx_unraveled = np.unravel_index(max_idx.item(), global_pooling_size)
            # adding to the winners list
            winners.append(max_idx_unraveled)
            # preventing the same feature to be the next winner
            total[max_idx_unraveled[0], :, :] = 0
            # columnar inhibition (increasing the chance of leanring diverse features)
            if inhibition_radius != 0:
                rowMin, rowMax = (
                    max(0, max_idx_unraveled[-2] - inhibition_radius),
                    min(total.size(-2), max_idx_unraveled[-2] + inhibition_radius + 1),
                )
                colMin, colMax = (
                    max(0, max_idx_unraveled[-1] - inhibition_radius),
                    min(total.size(-1), max_idx_unraveled[-1] + inhibition_radius + 1),
                )
                total[:, rowMin:rowMax, colMin:colMax] = 0
        else:
            break
    return winners


# decrease lateral intencities by factors given in the inhibition_kernel
def intensity_lateral_inhibition(intencities, inhibition_kernel):
    r"""Applies lateral inhibition on intensities. For each location, this inhibition decreases the intensity of the
    surrounding cells that has lower intensities by a specific factor. This factor is relative to the distance of the
    neighbors and are put in the :attr:`inhibition_kernel`.

    Args:
        intencities (Tensor): The tensor of input intensities.
        inhibition_kernel (Tensor): The tensor of inhibition factors.

    Returns:
        Tensor: Inhibited intensities.
    """
    intencities.squeeze_(0)
    intencities.unsqueeze_(1)

    inh_win_size = inhibition_kernel.size(-1)
    rad = inh_win_size // 2
    # repeat each value
    values = intencities.reshape(intencities.size(0), intencities.size(1), -1, 1)
    values = values.repeat(1, 1, 1, inh_win_size)
    values = values.reshape(
        intencities.size(0),
        intencities.size(1),
        -1,
        intencities.size(-1) * inh_win_size,
    )
    values = values.repeat(1, 1, 1, inh_win_size)
    values = values.reshape(
        intencities.size(0),
        intencities.size(1),
        -1,
        intencities.size(-1) * inh_win_size,
    )
    # extend patches
    padded = fn.pad(intencities, (rad, rad, rad, rad))
    # column-wise
    patches = padded.unfold(-1, inh_win_size, 1)
    patches = patches.reshape(
        patches.size(0),
        patches.size(1),
        patches.size(2),
        -1,
        patches.size(3) * patches.size(4),
    )
    patches.squeeze_(-2)
    # row-wise
    patches = patches.unfold(-2, inh_win_size, 1).transpose(-1, -2)
    patches = patches.reshape(patches.size(0), patches.size(1), 1, -1, patches.size(-1))
    patches.squeeze_(-3)
    # compare each element by its neighbors
    coef = values - patches
    coef.clamp_(min=0).sign_()  # "ones" are neighbors greater than center
    # convolution with full stride to get accumulative inhibiiton factor
    factors = fn.conv2d(coef, inhibition_kernel, stride=inh_win_size)
    result = intencities + intencities * factors

    intencities.squeeze_(1)
    intencities.unsqueeze_(0)
    result.squeeze_(1)
    result.unsqueeze_(0)
    return result


# performs local normalization
# on each region (of size radius*2 + 1) the mean value is computed and
# intensities will be divided by the mean value
# x is a 4D tensor
def local_normalization(
    input: Annotated[torch.Tensor, "1 features height width"],
    normalization_radius: int,
    eps: float = 1e-12,
) -> Annotated[torch.Tensor, "1 features height width"]:
    r"""Applies local normalization. on each region (of size radius*2 + 1) the mean value is computed and the
    intensities will be divided by the mean value. The input is a 4D tensor.

    Args:
        input (Tensor): The input tensor of shape (1, features, height, width).
        normalization_radius (int): The radius of normalization window.

    Returns:
        Tensor: Locally normalized tensor.
    """
    # computing local mean by 2d convolution
    kernel = torch.ones(
        1,
        1,
        normalization_radius * 2 + 1,
        normalization_radius * 2 + 1,
        device=input.device,
    ).float() / ((normalization_radius * 2 + 1) ** 2)
    # rearrange 4D tensor so input channels will be considered as minibatches
    y = input.squeeze(0)  # removes minibatch dim which was 1
    y.unsqueeze_(
        1
    )  # adds a dimension after channels so previous channels are now minibatches
    means = fn.conv2d(y, kernel, padding=normalization_radius) + eps  # computes means
    y = y / means  # normalization
    # swap minibatch with channels
    y.squeeze_(1)
    y.unsqueeze_(0)
    return y
