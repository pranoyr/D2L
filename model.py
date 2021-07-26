import torch
from torch import nn
from models.residual_net import ResidualNet
from models.resnet import ResNet101
from models.resnet_full import resnet101, resnet152, resnet18, resnet34, resnet50
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn

def generate_model(model_name):
	if model_name == 'ResidualNet-18':
		model = ResidualNet("ImageNet", 18, "CBAM")

	#Resnet
	if model_name == "Resnet-18":
		model = resnet18(pretrained=True)

	if model_name == "Resnet-34":
		model = resnet34(pretrained=True)

	if model_name == "Resnet-50":
		model = resnet50(pretrained=True)
		
	if model_name == "Resnet-152":
		model = resnet152(pretrained=True)

	if model_name == "Resnet-101":
		model = ResNet101(pretrained=True)
		
	# VGG
	if model_name == "Vgg11":
		model = vgg11(pretrained=True)

	if model_name == "Vgg11_bn":
		model = vgg11_bn(pretrained=True)

	if model_name == "Vgg13":
		model = vgg13(pretrained=True)

	if model_name == "Vgg13_bn":
		model = vgg13_bn(pretrained=True)

	if model_name == "Vgg16":
		model = vgg16(pretrained=True)

	if model_name == "Vgg16_bn":
		model = vgg16_bn(pretrained=True)

	if model_name == "Vgg19":
		model = vgg19(pretrained=True)

	if model_name == "Vgg19_bn":
		model = vgg19_bn(pretrained=True)

	return model