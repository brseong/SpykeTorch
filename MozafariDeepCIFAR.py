#################################################################################
# Reimplementation of the 10-Class Digit Recognition Experiment Performed in:   #
# https://arxiv.org/abs/1804.00227                                              #
#                                                                               #
# Reference:                                                                    #
# Mozafari, Milad, et al.,                                                      #
# "Combining STDP and Reward-Modulated STDP in                                  #
# Deep Convolutional Spiking Neural Networks for Digit Recognition."            #
# arXiv preprint arXiv:1804.00227 (2018).                                       #
#                                                                               #
# Original implementation (in C++/CUDA) is available upon request.              #
#################################################################################

import os
from typing import Literal, cast

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torchvision  # type: ignore
import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Image as PILImage
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import utils
from torchvision import transforms  # type: ignore
from jaxtyping import UInt8, Float, Int

use_cuda = True


class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()  # type: ignore

        self.conv1 = snn.Convolution(18, 90, 7, 0.8, 0.05)
        self.conv1_t = 15
        self.k1 = 5
        self.r1 = 3

        self.conv2 = snn.Convolution(90, 250, 5, 0.8, 0.05)
        self.conv2_t = 10
        self.k2 = 8
        self.r2 = 1

        self.conv3 = snn.Convolution(250, 200, 5, 0.8, 0.05)

        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        self.stdp3 = snn.STDP(self.conv3, (0.004, -0.003), False, 0.2, 0.8)
        self.anti_stdp3 = snn.STDP(self.conv3, (-0.004, 0.0005), False, 0.2, 0.8)
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.decision_map = list[int]()
        for i in range(10):
            self.decision_map.extend([i] * 20)

        self.ctx = dict[str, torch.Tensor | list[tuple[int, int, int]] | None](
            input_spikes=None, potentials=None, output_spikes=None, winners=None
        )
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

    def forward(self, input: UInt8[torch.Tensor, "15 18 32 32"], max_layer: int):  # noqa: F722
        input = sf.pad(input.float(), (2, 2, 2, 2), 0)
        if self.training:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = (
                        torch.tensor(
                            self.stdp1.learning_rate[0][0].item(),
                            device=self.stdp1.learning_rate[0][0].device,
                        )
                        * 2
                    )
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self.ctx["input_spikes"] = input
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1, 1, 1, 1))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t)
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 500:
                    self.spk_cnt2 = 0
                    ap = (
                        torch.tensor(
                            self.stdp2.learning_rate[0][0].item(),
                            device=self.stdp2.learning_rate[0][0].device,
                        )
                        * 2
                    )
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp2.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
                self.ctx["input_spikes"] = spk_in
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 3, 3), (2, 2, 2, 2))
            pot = self.conv3(spk_in)
            spk, _ = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            self.ctx["input_spikes"] = spk_in
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t)
            if max_layer == 1:
                return spk, pot
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1, 1, 1, 1)))
            spk, pot = sf.fire(pot, self.conv2_t)
            if max_layer == 2:
                return spk, pot
            pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2, 2, 2, 2)))
            spk, _ = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output

    def stdp(self, layer_idx: Literal[1, 2]):
        if layer_idx == 1:
            self.stdp1(
                self.ctx["input_spikes"],
                self.ctx["potentials"],
                self.ctx["output_spikes"],
                self.ctx["winners"],
            )
        if layer_idx == 2:
            self.stdp2(
                self.ctx["input_spikes"],
                self.ctx["potentials"],
                self.ctx["output_spikes"],
                self.ctx["winners"],
            )

    def update_learning_rates(
        self, stdp_ap: float, stdp_an: float, anti_stdp_ap: float, anti_stdp_an: float
    ):
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp3(
            self.ctx["input_spikes"],
            self.ctx["potentials"],
            self.ctx["output_spikes"],
            self.ctx["winners"],
        )

    def punish(self):
        self.anti_stdp3(
            self.ctx["input_spikes"],
            self.ctx["potentials"],
            self.ctx["output_spikes"],
            self.ctx["winners"],
        )


def train_unsupervise(
    network: CIFAR10Net,
    data: UInt8[torch.Tensor, "batch 15 18 32 32"],  # noqa: F722
    layer_idx: int,
):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        if layer_idx in [1, 2]:
            network.stdp(cast(Literal[1, 2], layer_idx))
        else:
            raise ValueError("Invalid layer index")


def train_rl(
    network: CIFAR10Net,
    data: UInt8[torch.Tensor, "batch 15 18 32 32"],  # noqa: F722
    target: Int[torch.Tensor, "batch"],  # noqa: F821
):
    network.train()
    perf = np.array([0, 0, 0])  # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0] += 1
                network.reward()
            else:
                perf[1] += 1
                network.punish()
        else:
            perf[2] += 1
    return perf / len(data)


def test(
    network: CIFAR10Net,
    data: UInt8[torch.Tensor, "15 18 32 32"],  # noqa: F722
    target: Int[torch.Tensor, "batch"],  # noqa: F821
):
    network.eval()
    perf = np.array([0, 0, 0])  # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0] += 1
            else:
                perf[1] += 1
        else:
            perf[2] += 1
    return perf / len(data)


class S1C1Transform:
    def __init__(self, filter: utils.Filter, timesteps: int = 15):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0

    def __call__(self, _image: PILImage) -> Float[torch.Tensor, "15 18 32 32"]:
        """Transforms an image to a temporal image

        Args:
            _image (PIL.Image.Image): Image of size (3, 32, 32).

        Returns:
            torch.Tensor: Tensor of size (timesteps, 3*# of DoG kernels, 32, 32).
        """
        if self.cnt % 1000 == 0:
            print(self.cnt)
        self.cnt += 1
        image = self.to_tensor(_image) * 255  # 0-255 ranged, 3x32x32
        image.unsqueeze_(1)  # 3x1x32x32
        image = self.filter(image)  # 3x6x32x32
        image = image.reshape(1, 18, 32, 32)  # 1x18x32x32
        image = sf.local_normalization(image, 8)  # 1x18x32x32
        temporal_image = self.temporal_transform(image)  # 15x18x32x32
        return temporal_image.sign().byte()


kernels = [
    utils.DoGKernel(3, 3 / 9, 6 / 9),
    utils.DoGKernel(3, 6 / 9, 3 / 9),
    utils.DoGKernel(7, 7 / 9, 14 / 9),
    utils.DoGKernel(7, 14 / 9, 7 / 9),
    utils.DoGKernel(13, 13 / 9, 26 / 9),
    utils.DoGKernel(13, 26 / 9, 13 / 9),
]
filter = utils.Filter(kernels, padding=6, thresholds=50)
s1c1 = S1C1Transform(filter)

datatype = tuple[UInt8[torch.Tensor, "15 18 32 32"], int]
data_root = "data"
CIFAR10_train = utils.CacheDataset(
    torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=s1c1
    )
)
CIFAR10_test = utils.CacheDataset(
    torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=s1c1
    )
)
CIFAR10_loader = DataLoader[datatype](CIFAR10_train, batch_size=1000, shuffle=False)
CIFAR10_testLoader = DataLoader[datatype](
    CIFAR10_test, batch_size=len(CIFAR10_test), shuffle=False
)

mozafari = CIFAR10Net()
if use_cuda:
    mozafari.cuda()

# Training The First Layer
print("Training the first layer")
if os.path.isfile("saved_l1_cifar10.net"):
    mozafari.load_state_dict(torch.load("saved_l1_cifar10.net"))  # type: ignore
else:
    for epoch in range(2):
        print("Epoch", epoch)
        iter = 0
        for data, targets in CIFAR10_loader:
            print("Iteration", iter)
            train_unsupervise(mozafari, data, 1)
            print("Done!")
            iter += 1
    torch.save(mozafari.state_dict(), "saved_l1_cifar10.net")  # type: ignore
# Training The Second Layer
print("Training the second layer")
if os.path.isfile("saved_l2_cifar10.net"):
    mozafari.load_state_dict(torch.load("saved_l2_cifar10.net"))  # type: ignore
else:
    for epoch in range(4):
        print("Epoch", epoch)
        iter = 0
        for data, targets in CIFAR10_loader:
            print("Iteration", iter)
            train_unsupervise(mozafari, data, 2)
            print("Done!")
            iter += 1
    torch.save(mozafari.state_dict(), "saved_l2_cifar10.net")  # type: ignore

# initial adaptive learning rates
apr = mozafari.stdp3.learning_rate[0][0].item()
anr = mozafari.stdp3.learning_rate[0][1].item()
app = mozafari.anti_stdp3.learning_rate[0][1].item()
anp = mozafari.anti_stdp3.learning_rate[0][0].item()

adaptive_min = 0
adaptive_int = 1
apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

# perf
best_train = np.array([0.0, 0.0, 0.0, 0.0])  # correct, wrong, silence, epoch
best_test = np.array([0.0, 0.0, 0.0, 0.0])  # correct, wrong, silence, epoch

# Training The Third Layer
print("Training the third layer")
for epoch in range(680):
    print("Epoch #:", epoch)
    perf_train = np.array([0.0, 0.0, 0.0])
    for data, targets in CIFAR10_loader:
        perf_train_batch = train_rl(mozafari, data, targets)
        print(perf_train_batch)
        # update adaptive learning rates
        apr_adapt = apr * (perf_train_batch[1] * adaptive_int + adaptive_min)
        anr_adapt = anr * (perf_train_batch[1] * adaptive_int + adaptive_min)
        app_adapt = app * (perf_train_batch[0] * adaptive_int + adaptive_min)
        anp_adapt = anp * (perf_train_batch[0] * adaptive_int + adaptive_min)
        mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
        perf_train += perf_train_batch
    perf_train /= len(CIFAR10_loader)
    if best_train[0] <= perf_train[0]:
        best_train = np.append(perf_train, epoch)
    print("Current Train:", perf_train)
    print("   Best Train:", best_train)

    for data, targets in CIFAR10_testLoader:
        perf_test = test(mozafari, data, targets)
        if best_test[0] <= perf_test[0]:
            best_test = np.append(perf_test, epoch)
            torch.save(mozafari.state_dict(), "saved_cifar10.net")  # type: ignore
        print(" Current Test:", perf_test)
        print("    Best Test:", best_test)
