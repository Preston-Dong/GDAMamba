import numpy as np
import torch
import torch.utils.data as data
import scipy.io as sio


def generate_label_image_from_real_labels(sample_coords, real_labels):
    """
    根据选定的训练样本坐标和真实标签生成标签图像。
    使用真实标签来确定哪些像素是有效样本，哪些是变化或不变化。

    参数:
    sample_coords (torch.Tensor): 包含样本坐标的张量，形状为 [num_samples, 2]，其中每行是 (H, W) 坐标。
    real_labels (torch.Tensor): 真实标签图像，形状为 (H, W)，其中每个像素的值为 0、1 或 -1。

    返回:
    torch.Tensor: 生成的标签图像，大小为 (H, W)，其中未标注的区域为 -1，标注区域为 0 或 1。
    """
    # 获取图像的高度和宽度，通过真实标签的形状获取
    H, W = real_labels.shape

    # 初始化标签图像，所有像素值为 -1，表示不参与的像素
    labels_image = torch.full((H, W), -1, dtype=real_labels.dtype)

    # 遍历选定的样本坐标
    for idx, (h, w) in enumerate(sample_coords[:, 1:]):
        # 只在真实标签图像上有有效标签的坐标位置设置标签
        if real_labels[h, w] != -1:  # 如果真实标签不为 -1（即该位置有效）
            labels_image[h, w] = real_labels[h, w]  # 赋值为真实标签（0 或 1）

    return labels_image


class HSICD_data(data.Dataset):
    def __init__(self, data_sample, cfg):

        self.phase = cfg['phase']
        self.img1 = data_sample['img1_pad']
        self.img2 = data_sample['img2_pad']
        self.patch_coordinates = data_sample['patch_coordinates']
        self.gt = data_sample['img_gt']

        if self.phase == 'train':
            self.data_indices = data_sample['train_sample_center']
        elif self.phase == 'test':
            self.data_indices = data_sample['test_sample_center']

    def create_sample_mask(self, data_sample, indices, is_train=True):
        sample_mask = torch.zeros_like(data_sample['img_gt'], dtype=torch.uint8)
        for index in indices:
            img_index = self.patch_coordinates[index[0]]
            sample_mask[img_index[0]:img_index[1], img_index[2]:img_index[3]] = 1

        if is_train:
            train_sample_mask = sample_mask
            train_sample_mask = np.array(train_sample_mask)
            sio.savemat('train_sample_mask.mat', {'train_sample_mask': train_sample_mask})
        else:
            test_sample_mask = sample_mask
            test_sample_mask = np.array(test_sample_mask)
            sio.savemat('test_sample_mask.mat', {'test_sample_mask': test_sample_mask})

        return sample_mask

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        index = self.data_indices[idx]
        img_index = self.patch_coordinates[index[0]]
        img1 = self.img1[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        img2 = self.img2[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        label = self.gt[index[1], index[2]]

        return img1, img2, label, index
