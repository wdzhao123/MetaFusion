import logging
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

from models.metafusion_net import FusionNet as FusionNetwork
from utils.dataloader import get_test_loader

device = 'cuda:0'


def test(test_loader, model, checkpoint, save_path):
    val_save_path = save_path
    os.makedirs(val_save_path, exist_ok=True)
    model.eval()
    model.load_state_dict(torch.load(checkpoint), strict=True)
    tqdm.write('load from{}'.format(checkpoint))
    with torch.no_grad():

        for i, (irimage, visimage_rgb, visimage_bri, visimage_clr, image_name) in enumerate(tqdm(test_loader), start=1):
            ir_image = irimage.to(device)
            visimage_rgb = visimage_rgb.to(device)
            visimage_bri = visimage_bri.to(device)

            _, res_weight = model(torch.cat([ir_image, visimage_rgb], dim=1))
            fus_img = res_weight[:, 0, :, :] * ir_image + res_weight[:, 1, :, :] * visimage_bri

            # HSV2RGB
            bri = fus_img.detach().cpu().numpy() * 255
            bri = bri.reshape([fus_img.size()[2], fus_img.size()[3]])
            bri = np.where(bri < 0, 0, bri)
            bri = np.where(bri > 255, 255, bri)
            im1 = Image.fromarray(bri.astype(np.uint8))

            clr = visimage_clr.numpy().squeeze().transpose(1, 2, 0) * 255
            clr = np.concatenate((clr, bri.reshape(fus_img.size()[2], fus_img.size()[3], 1)), axis=2)

            clr[:, :, 2] = im1
            clr = cv2.cvtColor(clr.astype(np.uint8), cv2.COLOR_HSV2RGB)

            if 'TNO' in image_name[0]:
                cv2.imwrite(
                    os.path.join(val_save_path, os.path.split(image_name[0])[1]),
                    clr)
            else:
                cv2.imwrite(
                    os.path.join(val_save_path, os.path.split(image_name[0])[1][:-4] + '.jpg'),
                    clr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./weight/model_weight.pth', help='fusion network weight')
    parser.add_argument('--blocks', type=int, default=3, help='blocks number')
    parser.add_argument('--test_ir_root', type=str, default='', required=True, help='the test ir images root')
    parser.add_argument('--test_vis_root', type=str, default='', required=True, help='the test vis images root')
    parser.add_argument('--save_path', type=str, default='./res/', help='the fusion results will be saved here')

    opt = parser.parse_args()

    # build the model
    fusion_net = FusionNetwork(block_num=opt.blocks, feature_out=False).to(device)
    print(fusion_net)

    # load data
    tqdm.write('load data...')

    test_loader = get_test_loader(
        ir_root=opt.test_ir_root,
        vis_root=opt.test_vis_root,
        batchsize=1,
        shuffle=False
    )

    test(
        test_loader=test_loader,
        model=fusion_net,
        checkpoint=opt.checkpoint,
        save_path=opt.save_path
    )
