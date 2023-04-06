import os

import cv2
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


def get_test_loader(ir_root, vis_root, batchsize=1, testsize=320,
                    shuffle=False, num_workers=8, pin_memory=True):
    # dataset = TestDataset(ir_root=ir_root, vis_root=vis_root, testsize=testsize)
    dataset = TestFusionDataset(ir_root=ir_root, vis_root=vis_root, testsize=testsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.resize((1024 // 2, 768 // 2), Image.BILINEAR)
        return img.convert('RGB')


class TestFusionDataset(data.Dataset):
    def __init__(self, ir_root, vis_root, testsize):
        self.testsize = testsize
        # get filenames
        self.irimages = [os.path.join(ir_root, f) for f in os.listdir(ir_root)
                         if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.visimages = [os.path.join(vis_root, f) for f in os.listdir(vis_root)
                          if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

        # sorted files
        self.irimages = sorted(self.irimages)
        self.visimages = sorted(self.visimages)

        # transforms

        self.img_transform = transforms.Compose([transforms.ToTensor()])
        self.toPIL = transforms.ToPILImage()
        self.size = len(self.visimages)
        if len(self.visimages) != len(self.irimages):
            raise ValueError('ir and vis img num is different.')

    def __getitem__(self, index):
        # read imgs
        irimage = self.gray_loader(self.irimages[index])
        visimage_rgb = rgb_loader(self.visimages[index])
        visimage_bri, visimage_clr = self.bri_clr_loader(self.visimages[index])

        visimage_bri = self.toPIL(visimage_bri)
        visimage_clr = self.toPIL(visimage_clr)

        irimage = self.img_transform(irimage)
        visimage_rgb = self.img_transform(visimage_rgb)
        visimage_bri = self.img_transform(visimage_bri)
        visimage_clr = self.img_transform(visimage_clr)

        return irimage, visimage_rgb, visimage_bri, visimage_clr, self.irimages[index]

    def bri_clr_loader(self, path):
        img1 = cv2.imread(path)
        img1 = cv2.resize(img1, (1024 // 2, 768 // 2),
                          interpolation=cv2.INTER_LINEAR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        color = img1[:, :, 0:2]
        brightness = img1[:, :, 2]
        return brightness, color

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.resize((1024 // 2, 768 // 2), Image.BILINEAR)
            return img.convert('L')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.resize((1024 // 2, 768 // 2), Image.BILINEAR)
            return img.convert('L')

    def __len__(self):
        return self.size
