from torchvision.models import  alexnet, resnet18, vgg11, vgg11_bn, vgg16, vgg16_bn, resnet34, resnet50
from cnn import SimpleCNN, softCrossEntropy, WeightedMultiLabelBinaryClassification, WeightedMeanSquareError, WeightedNLLLoss
import sys

def select_model(args):
    model_type = args.model_type
    kwargs = {"num_classes" : args.num_classes}

    if model_type == 'alexnet':
        model = alexnet(pretrained=False, progress=True, **kwargs)
    elif model_type == 'resnet18':
        model = resnet18(pretrained=False, progress=True, **kwargs)
    elif model_type == 'resnet34':
        model = resnet34(pretrained=False, progress=True, **kwargs)
    elif model_type == 'resnet50':
        model = resnet50(pretrained=False, progress=True, **kwargs)
    elif model_type == 'vgg11':
        model = vgg11(pretrained=False, progress=True, **kwargs)
    elif model_type == 'vgg11_bn':
        model = vgg11_bn(pretrained=False, progress=True, **kwargs)
    elif model_type == 'vgg16':
        model = vgg16(pretrained=False, progress=True, **kwargs)
    elif model_type == 'vgg16_bn':
        model = vgg16_bn(pretrained=False, progress=True, **kwargs)
    else:

        model = SimpleCNN(num_classes= args.num_classes, num_cov_layer=args.num_cov_layer, channels= args.channels,
                          kernel_size= args.kernel_size, stride= args.stride,
                          num_mlp_layer=args.num_mlp_layer, mlp_hids= args.mlp_hids,
                          adp_output_size=args.adp_output_size, dropout= args.dropout)

        #model = SimpleCNN(num_classes=5)

    if args.verbose:
        print(model)
    return model


def select_criterion(args):
    loss_type = args.loss_type
    if loss_type == 'nll':
        criterion = WeightedNLLLoss()
    elif loss_type == 'sce':
        criterion = softCrossEntropy()
    elif loss_type == 'bce':
        criterion = WeightedMultiLabelBinaryClassification()
    elif loss_type == 'mse':
        criterion = WeightedMeanSquareError()
    else:
        print("Do not support loss type : {}".format(loss_type))
        sys.exit()
    if args.verbose:
        print("Loss type: {}".format(loss_type))
    return criterion