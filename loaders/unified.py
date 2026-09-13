import importlib
import torch
from torch import nn
from torch.nn import functional as F
from ..core import ROOT, models, sha256


class SharpnessPreActBlock(nn.Module):
    """Pre-activation BasicBlock used by the ICML 2023 checkpoint release."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class SharpnessPreActResNet18(nn.Module):
    """CIFAR PreActResNet18, width 64, matching tml-epfl/sharpness-vs-generalization."""

    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.bn = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, 10)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(SharpnessPreActBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4).flatten(1)
        return self.linear(out)

class Classifier(nn.Module):
    """Input: unnormalized float RGB NCHW in [0,1]; output: ten raw logits."""
    def __init__(self, network, metadata, preprocessing='native', common_mean=None, common_std=None):
        super().__init__()
        self.network, self.metadata, self.preprocessing = network, metadata, preprocessing
        if preprocessing == 'native':
            mean, std = metadata['normalization_mean'], metadata['normalization_std']
        elif preprocessing == 'common':
            if common_mean is None or common_std is None:
                raise ValueError('common preprocessing requires explicit common_mean and common_std')
            mean, std = common_mean, common_std
        elif preprocessing == 'none':
            mean, std = [0, 0, 0], [1, 1, 1]
        else:
            raise ValueError(preprocessing)
        if len(mean) != 3 or len(std) != 3 or any(s <= 0 for s in std):
            raise ValueError('Invalid normalization')
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32) or not images.is_floating_point():
            raise ValueError('Expected floating tensor [N,3,32,32]')
        return self.network((images - self.mean) / self.std)

def build_network(row):
    spec = row['loader_spec']
    if spec['adapter'] == 'modelzoos_resnet18':
        from torchvision.models.resnet import ResNet, BasicBlock
        net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        return net
    if spec['adapter'] == 'sharpness_preact_resnet18':
        return SharpnessPreActResNet18()
    module = importlib.import_module('model_zoo.vendor.' + spec['module'])
    return getattr(module, spec['factory'])(pretrained=False)

def read_state(path, row=None):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if row is not None and row.get('loader_spec', {}).get('state_key'):
        state = state[row['loader_spec']['state_key']]
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise TypeError('Expected a tensor state_dict')
    return state

def load_model(model_id, *, device='cpu', preprocessing='native', common_mean=None, common_std=None, check_hash=True):
    rows = {r['model_id']: r for r in models()}
    if model_id not in rows:
        raise KeyError(model_id)
    row = rows[model_id]
    if row.get('status') in ('duplicate', 'failed'):
        raise ValueError(f'Model is excluded: {model_id}')
    path = ROOT / row['local_checkpoint_path']
    if check_hash and (not row.get('sha256') or sha256(path) != row['sha256']):
        raise ValueError('Checkpoint integrity check failed')
    net = build_network(row)
    net.load_state_dict(read_state(path, row), strict=True)
    return Classifier(net, row, preprocessing, common_mean, common_std).to(device).eval()

def pair_relation(first, second):
    by_id = {r['model_id']: r for r in models()}
    a = by_id[first] if isinstance(first, str) else first
    b = by_id[second] if isinstance(second, str) else second
    same_arch = a['topology_id'] == b['topology_id']
    same_lineage = a['lineage_id'] == b['lineage_id']
    return dict(same_lineage=same_lineage, same_architecture=same_arch,
                same_family=a['architecture_family'] == b['architecture_family'],
                different_architecture=not same_arch, hard_negative=same_arch and not same_lineage)
