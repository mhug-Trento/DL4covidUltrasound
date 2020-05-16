import torchvision.transforms as transforms
from PIL import Image
import cv2


class NumpyToPIL:

    def __call__(self, frame):
        return Image.fromarray(frame).convert('RGB')


def get_transforms(args, split):
    '''
    Gets a sample set of transformations to obtain training images
    :return: callable transforming numpy frames to torch tensors
    '''

    transform_list = {
        'train':
            [  # Sobel(),
                NumpyToPIL(),
                transforms.RandomResizedCrop(
                    size=(args.img_size, args.img_size),
                    scale=(0.9, 1.0),
                    ratio=(9 / 10, 10 / 9)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(23)], p=0.8),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.125)], p=0.8),
                transforms.ToTensor()
            ],
        'test': [
            NumpyToPIL(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor()
        ]
    }

    return transforms.Compose(transform_list[split])