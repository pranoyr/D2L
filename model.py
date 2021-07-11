import torch
from torch import nn
from models.residual_net import ResidualNet
from models.resnet import ResNet101


def generate_model(model_name, num_classes):
	if model_name == 'ResidualNet-18':
		model = ResidualNet("ImageNet", 18, num_classes, "CBAM")
	if model_name == "Resnet-101":
		model = ResNet101(num_classes)
	return model