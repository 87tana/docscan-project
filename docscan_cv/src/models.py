import torch.nn as nn
from torchvision import models
from transformers import AutoModelForImageClassification, AutoImageProcessor

def _set_trainable(model, freeze_backbone, head_module):
    if not freeze_backbone:
        return
    for p in model.parameters():
        p.requires_grad = False
    for p in head_module.parameters():
        p.requires_grad = True

class HFClassifierWrapper(nn.Module):
    """Makes a Hugging Face model behave like a plain torchvision model:
    forward(x) -> logits, instead of forward(x) -> an object with a .logits attribute."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        return self.hf_model(pixel_values=x).logits


def build_model(config):
    name = config["model_name"]
    num_classes = config["num_classes"]
    freeze_backbone = config.get("freeze_backbone", True)

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        _set_trainable(model, freeze_backbone, model.fc)
        image_size, mean, std = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        _set_trainable(model, freeze_backbone, model.classifier[1])
        image_size, mean, std = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        _set_trainable(model, freeze_backbone, model.classifier[3])
        image_size, mean, std = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]



    elif name == "vit_base16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        _set_trainable(model, freeze_backbone, model.heads.head)
        image_size, mean, std = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif name == "dit_base":
        checkpoint = config["hf_checkpoint"]
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        hf_model = AutoModelForImageClassification.from_pretrained(
            checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        model = HFClassifierWrapper(hf_model)
        _set_trainable(hf_model, freeze_backbone, hf_model.classifier)
        image_size = processor.size.get("height", 224)
        mean, std = processor.image_mean, processor.image_std

    else:
        raise ValueError(f"Unknown model_name: {name}")

    return model, image_size, mean, std
