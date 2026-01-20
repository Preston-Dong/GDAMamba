from scipy.io import loadmat
import numpy as np

def get_Farmland_dataset():
    data_set_before = loadmat(r'./datasets/Yancheng/farm06.mat')['imgh']
    data_set_after = loadmat(r'./datasets/Yancheng/farm07.mat')['imghl']
    ground_truth = loadmat(r'./datasets/Yancheng/label.mat')['label']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt


def get_GF5B_dataset():
    data_set_before = loadmat(r'./datasets/GF5B/GF5B_BI.mat')['img1']
    data_set_after = loadmat(r'./datasets/GF5B/GF5B_BI.mat')['img2']
    ground_truth = loadmat(r'./datasets/GF5B/GF5B_BI.mat')['binary_label']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')-1
    img1 = (img1 - img1.min()) / (img1.max()-img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    # img1[img1 < 0] = 0
    # img2[img2 < 0] = 0

    return img1, img2, gt


def get_River_dataset():
    data_set_before = loadmat(r'./datasets/river/river_before.mat')['river_before']
    data_set_after = loadmat(r'./datasets/river/river_after.mat')['river_after']
    ground_truth = loadmat(r'./datasets/river/groundtruth.mat')['lakelabel_v1']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')/255.0

    return img1, img2, gt


def get_USA_dataset():
    """USA equals Herminston"""
    data_mat = loadmat(r'./datasets/USA/USA_Change_Dataset.mat')
    data_set_before = data_mat['T1']
    data_set_after = data_mat['T2']
    ground_truth = data_mat['Binary']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt


def get_Herminston_dataset():
    data_set_before = loadmat(r'./datasets/hermiston/hermiston2004.mat')['HypeRvieW']
    data_set_after = loadmat(r'./datasets/hermiston/hermiston2007.mat')['HypeRvieW']
    ground_truth = loadmat(r'./datasets/hermiston/rdChangesHermiston_5classes.mat')['gt5clasesHermiston']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')
    gt[gt > 1.0] = 1.0

    return img1, img2, gt


def get_BayArea_dataset():
    """attention: 0 is unlabeled"""
    data_set_before = loadmat(r'./datasets/bayArea/mat/Bay_Area_2013.mat')['HypeRvieW']
    data_set_after = loadmat(r'./datasets/bayArea/mat/Bay_Area_2015.mat')['HypeRvieW']
    ground_truth = loadmat(r'./datasets/bayArea/mat/bayArea_gtChanges2.mat.mat')['HypeRvieW']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')-1
    gt = np.where(gt == 0, 1, np.where(gt == 1, 0, gt))

    return img1, img2, gt


def get_SantaBarbara_dataset():
    """attention: 0 is unlabeled"""
    data_set_before = loadmat(r'./datasets/santaBarbara/mat/barbara_2013.mat')['HypeRvieW']
    data_set_after = loadmat(r'./datasets/santaBarbara/mat/barbara_2014.mat')['HypeRvieW']
    ground_truth = loadmat(r'./datasets/santaBarbara/mat/barbara_gtChanges.mat')['HypeRvieW']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')-1
    gt = np.where(gt == 0, 1, np.where(gt == 1, 0, gt))

    return img1, img2, gt


def get_dataset(current_dataset):
    if current_dataset == 'Farmland':
        return get_Farmland_dataset()
    elif current_dataset == 'GF5B':
        return get_GF5B_dataset()
    elif current_dataset == 'River':
        return get_River_dataset()
    elif current_dataset == 'USA':
        return get_USA_dataset()
    elif current_dataset == 'Herminston':
        return get_Herminston_dataset()
    elif current_dataset == 'BayArea':
        return get_BayArea_dataset()
    elif current_dataset == 'SantaBarbara':
        return get_SantaBarbara_dataset()

