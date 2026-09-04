from torchvision import transforms


def build_transform(image_size, mean, std, train=False):
    ops = [transforms.Resize((image_size, image_size))]

    if train:
        ops += [
            transforms.RandomRotation(3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(ops)
