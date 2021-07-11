import torch
from torch import nn
from models.residual_net import ResidualNet


def generate_model(model_name):
	if model_name == 'ResidualNet-18':
		model = ResidualNet("ImageNet", 18, 1, "CBAM")
	return model