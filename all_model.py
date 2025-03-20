import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from motion_magnification_learning_based_master.magnet import Manipulator as MagManipulator
from motion_magnification_learning_based_master.magnet import Encoder_No_texture as MagEncoder_No_texture


def gen_state_dict(weights_path):
    st = torch.load(weights_path)
    st_ks = list(st.keys())
    st_vs = list(st.values())
    state_dict = {}
    for st_k, st_v in zip(st_ks, st_vs):
        state_dict[st_k.replace('module.', '')] = st_v
    return state_dict

class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)

class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            raise NotImplementedError
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

class eca_layer_2d_v2(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel):
        super(eca_layer_2d_v2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs(math.log(channel,2)+1)/2)
        k_size = t if t%2 else (t+1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)

        y_avg = self.conv(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_max = self.conv(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y_avg+y_max)

        return x * y.expand_as(x)

class SKD_TSTSAN(nn.Module):
    def __init__(self, out_channels=5, amp_factor=5):
        super(SKD_TSTSAN, self).__init__()
        self.Aug_Encoder_L = MagEncoder_No_texture(dim_in=16)
        self.Aug_Encoder_S = MagEncoder_No_texture(dim_in=1)
        self.Aug_Encoder_T = MagEncoder_No_texture(dim_in=2)
        self.Aug_Manipulator_L = MagManipulator()
        self.Aug_Manipulator_S = MagManipulator()
        self.Aug_Manipulator_T = MagManipulator()

        self.conv1_L = nn.Conv2d(32, out_channels=64, kernel_size=5, stride=1)
        self.conv1_S = nn.Conv2d(32, out_channels=64, kernel_size=5, stride=1)
        self.conv1_T = nn.Conv2d(32, out_channels=64, kernel_size=5, stride=1)

        self.relu = nn.ReLU()
        self.bn1_L = nn.BatchNorm2d(64)
        self.bn1_S = nn.BatchNorm2d(64)
        self.bn1_T = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)

        self.AC1_conv1_L = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.AC1_conv1_S = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.AC1_conv1_T = TemporalShift(nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1), n_segment=2,
                                     n_div=8)
        self.AC1_bn1_L = nn.BatchNorm2d(128)
        self.AC1_bn1_S = nn.BatchNorm2d(128)
        self.AC1_bn1_T = nn.BatchNorm2d(128)

        self.AC1_conv2_L = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.AC1_conv2_S = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.AC1_conv2_T = TemporalShift(nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1), n_segment=2,
                                     n_div=8)
        self.AC1_bn2_L = nn.BatchNorm2d(128)
        self.AC1_bn2_S = nn.BatchNorm2d(128)
        self.AC1_bn2_T = nn.BatchNorm2d(128)
        self.AC1_pool = nn.AdaptiveAvgPool2d(1)
        self.AC1_fc = nn.Linear(in_features=384, out_features=out_channels)

        self.conv2_L = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_S = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_T = TemporalShift(nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1), n_segment=2,
                                     n_div=8)
        self.bn2_L = nn.BatchNorm2d(64)
        self.bn2_S = nn.BatchNorm2d(64)
        self.bn2_T = nn.BatchNorm2d(64)

        self.conv3_L = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_S = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_T = TemporalShift(nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1), n_segment=2,
                                     n_div=8)
        self.bn3_L = nn.BatchNorm2d(64)
        self.bn3_S = nn.BatchNorm2d(64)
        self.bn3_T = nn.BatchNorm2d(64)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.AC2_conv1_L = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.AC2_conv1_S = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.AC2_conv1_T = TemporalShift(nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1), n_segment=2,
                                     n_div=8)
        self.AC2_bn1_L = nn.BatchNorm2d(128)
        self.AC2_bn1_S = nn.BatchNorm2d(128)
        self.AC2_bn1_T = nn.BatchNorm2d(128)

        self.AC2_conv2_L = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.AC2_conv2_S = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.AC2_conv2_T = TemporalShift(nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1), n_segment=2,
                                     n_div=8)
        self.AC2_bn2_L = nn.BatchNorm2d(128)
        self.AC2_bn2_S = nn.BatchNorm2d(128)
        self.AC2_bn2_T = nn.BatchNorm2d(128)
        self.AC2_pool = nn.AdaptiveAvgPool2d(1)
        self.AC2_fc = nn.Linear(in_features=384, out_features=out_channels)

        self.all_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv4_L = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_S = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_T = TemporalShift(nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1), n_segment=2,
                                     n_div=8)
        self.bn4_L = nn.BatchNorm2d(128)
        self.bn4_S = nn.BatchNorm2d(128)
        self.bn4_T = nn.BatchNorm2d(128)

        self.conv5_L = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_S = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_T = TemporalShift(nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1), n_segment=2,
                                     n_div=8)
        self.bn5_L = nn.BatchNorm2d(128)
        self.bn5_S = nn.BatchNorm2d(128)
        self.bn5_T = nn.BatchNorm2d(128)

        self.fc2 = nn.Linear(in_features=384, out_features=out_channels)

        self.ECA1 = eca_layer_2d_v2(64)
        self.ECA2 = eca_layer_2d_v2(64)
        self.ECA3 = eca_layer_2d_v2(64)
        self.ECA4 = eca_layer_2d_v2(128)
        self.ECA5 = eca_layer_2d_v2(128)

        self.AC1_ECA1 = eca_layer_2d_v2(128)
        self.AC1_ECA2 = eca_layer_2d_v2(128)
        self.AC2_ECA1 = eca_layer_2d_v2(128)
        self.AC2_ECA2 = eca_layer_2d_v2(128)

        self.amp_factor = amp_factor

        self.consensus = ConsensusModule("avg")

        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        x1 = input[:, 2:18, :, :]
        x1_onset = input[:, 18:34, :, :]
        x2 = input[:, 0, :, :].unsqueeze(dim=1)
        x2_onset = input[:, 1, :, :].unsqueeze(dim=1)
        x3 = input[:, 34:, :, :]

        bsz = x1.shape[0]

        x3 = torch.reshape(x3, (bsz * 2, 2, 48, 48))

        x3_onset = torch.zeros(bsz * 2, 2, 48, 48).cuda()

        motion_x1_onset = self.Aug_Encoder_L(x1_onset)
        motion_x1 = self.Aug_Encoder_L(x1)
        x1 = self.Aug_Manipulator_L(motion_x1_onset, motion_x1, self.amp_factor)
        motion_x2_onset = self.Aug_Encoder_S(x2_onset)
        motion_x2 = self.Aug_Encoder_S(x2)
        x2 = self.Aug_Manipulator_S(motion_x2_onset, motion_x2, self.amp_factor)
        motion_x3_onset = self.Aug_Encoder_T(x3_onset)
        motion_x3 = self.Aug_Encoder_T(x3)
        x3 = self.Aug_Manipulator_T(motion_x3_onset, motion_x3, self.amp_factor)

        x1 = self.conv1_L(x1)
        x1 = self.bn1_L(x1)
        x1 = self.relu(x1)
        x1 = self.ECA1(x1)
        x1 = self.maxpool(x1)

        x2 = self.conv1_S(x2)
        x2 = self.bn1_S(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x3 = self.conv1_T(x3)
        x3 = self.bn1_T(x3)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3)

        AC1_x1 = self.AC1_conv1_L(x1)
        AC1_x1 = self.AC1_bn1_L(AC1_x1)
        AC1_x1 = self.relu(AC1_x1)
        AC1_x1 = self.AC1_ECA1(AC1_x1)
        AC1_x1 = self.AC1_conv2_L(AC1_x1)
        AC1_x1 = self.AC1_bn2_L(AC1_x1)
        AC1_x1 = self.relu(AC1_x1)
        AC1_x1 = self.AC1_ECA2(AC1_x1)
        AC1_x1 = self.AC1_pool(AC1_x1)
        AC1_x1_all = AC1_x1.view(AC1_x1.size(0), -1)

        AC1_x2 = self.AC1_conv1_S(x2)
        AC1_x2 = self.AC1_bn1_S(AC1_x2)
        AC1_x2 = self.relu(AC1_x2)
        AC1_x2 = self.AC1_conv2_S(AC1_x2)
        AC1_x2 = self.AC1_bn2_S(AC1_x2)
        AC1_x2 = self.relu(AC1_x2)
        AC1_x2 = self.AC1_pool(AC1_x2)
        AC1_x2_all = AC1_x2.view(AC1_x2.size(0), -1)

        AC1_x3 = self.AC1_conv1_T(x3)
        AC1_x3 = self.AC1_bn1_T(AC1_x3)
        AC1_x3 = self.relu(AC1_x3)
        AC1_x3 = self.AC1_conv2_T(AC1_x3)
        AC1_x3 = self.AC1_bn2_T(AC1_x3)
        AC1_x3 = self.relu(AC1_x3)
        AC1_x3 = self.AC1_pool(AC1_x3)
        AC1_x3_all = AC1_x3.view(AC1_x3.size(0), -1)

        AC1_x3_all = AC1_x3_all.view((-1, 2) + AC1_x3_all.size()[1:])
        AC1_x3_all = self.consensus(AC1_x3_all)
        AC1_x3_all = AC1_x3_all.squeeze(1)
        AC1_feature = torch.cat((AC1_x1_all, AC1_x2_all, AC1_x3_all), 1)
        AC1_x_all = self.dropout(AC1_feature)
        AC1_x_all = self.AC1_fc(AC1_x_all)


        x1 = self.conv2_L(x1)
        x1 = self.bn2_L(x1)
        x1 = self.relu(x1)
        x1 = self.ECA2(x1)
        x1 = self.conv3_L(x1)
        x1 = self.bn3_L(x1)
        x1 = self.relu(x1)
        x1 = self.ECA3(x1)
        x1 = self.avgpool(x1)

        x2 = self.conv2_S(x2)
        x2 = self.bn2_S(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_S(x2)
        x2 = self.bn3_S(x2)
        x2 = self.relu(x2)
        x2 = self.avgpool(x2)

        x3 = self.conv2_T(x3)
        x3 = self.bn2_T(x3)
        x3 = self.relu(x3)
        x3 = self.conv3_T(x3)
        x3 = self.bn3_T(x3)
        x3 = self.relu(x3)
        x3 = self.avgpool(x3)

        AC2_x1 = self.AC2_conv1_L(x1)
        AC2_x1 = self.AC2_bn1_L(AC2_x1)
        AC2_x1 = self.relu(AC2_x1)
        AC2_x1 = self.AC2_ECA1(AC2_x1)
        AC2_x1 = self.AC2_conv2_L(AC2_x1)
        AC2_x1 = self.AC2_bn2_L(AC2_x1)
        AC2_x1 = self.relu(AC2_x1)
        AC2_x1 = self.AC2_ECA2(AC2_x1)
        AC2_x1 = self.AC2_pool(AC2_x1)
        AC2_x1_all = AC2_x1.view(AC2_x1.size(0), -1)

        AC2_x2 = self.AC2_conv1_S(x2)
        AC2_x2 = self.AC2_bn1_S(AC2_x2)
        AC2_x2 = self.relu(AC2_x2)
        AC2_x2 = self.AC2_conv2_S(AC2_x2)
        AC2_x2 = self.AC2_bn2_S(AC2_x2)
        AC2_x2 = self.relu(AC2_x2)
        AC2_x2 = self.AC2_pool(AC2_x2)
        AC2_x2_all = AC2_x2.view(AC2_x2.size(0), -1)

        AC2_x3 = self.AC2_conv1_T(x3)
        AC2_x3 = self.AC2_bn1_T(AC2_x3)
        AC2_x3 = self.relu(AC2_x3)
        AC2_x3 = self.AC2_conv2_T(AC2_x3)
        AC2_x3 = self.AC2_bn2_T(AC2_x3)
        AC2_x3 = self.relu(AC2_x3)
        AC2_x3 = self.AC2_pool(AC2_x3)
        AC2_x3_all = AC2_x3.view(AC2_x3.size(0), -1)

        AC2_x3_all = AC2_x3_all.view((-1, 2) + AC2_x3_all.size()[1:])
        AC2_x3_all = self.consensus(AC2_x3_all)
        AC2_x3_all = AC2_x3_all.squeeze(1)
        AC2_feature = torch.cat((AC2_x1_all, AC2_x2_all, AC2_x3_all), 1)
        AC2_x_all = self.dropout(AC2_feature)
        AC2_x_all = self.AC2_fc(AC2_x_all)


        x1 = self.conv4_L(x1)
        x1 = self.bn4_L(x1)
        x1 = self.relu(x1)
        x1 = self.ECA4(x1)
        x1 = self.conv5_L(x1)
        x1 = self.bn5_L(x1)
        x1 = self.relu(x1)
        x1 = self.ECA5(x1)
        x1 = self.all_avgpool(x1)
        x1_all = x1.view(x1.size(0), -1)

        x2 = self.conv4_S(x2)
        x2 = self.bn4_S(x2)
        x2 = self.relu(x2)
        x2 = self.conv5_S(x2)
        x2 = self.bn5_S(x2)
        x2 = self.relu(x2)
        x2 = self.all_avgpool(x2)
        x2_all = x2.view(x2.size(0), -1)

        x3 = self.conv4_T(x3)
        x3 = self.bn4_T(x3)
        x3 = self.relu(x3)
        x3 = self.conv5_T(x3)
        x3 = self.bn5_T(x3)
        x3 = self.relu(x3)
        x3 = self.all_avgpool(x3)
        x3_all = x3.view(x3.size(0), -1)

        x3_all = x3_all.view((-1, 2) + x3_all.size()[1:])

        x3_all = self.consensus(x3_all)
        x3_all = x3_all.squeeze(1)

        final_feature = torch.cat((x1_all, x2_all, x3_all), 1)
        x_all = self.dropout(final_feature)
        x_all = self.fc2(x_all)
        return x_all, AC1_x_all, AC2_x_all, final_feature, AC1_feature, AC2_feature


def get_model(model_name, class_num, alpha):
    if model_name == "SKD_TSTSAN":
        return SKD_TSTSAN(class_num, alpha)

