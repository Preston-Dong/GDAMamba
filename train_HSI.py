import os
import torch.nn as nn
import argparse
import scipy.io as io

# import configs_SSTFormer.configs_USA as cfg
import torch.optim as optim
from data.HSICD_data import HSICD_data, generate_label_image_from_real_labels
from data.get_train_test_set import get_train_test_set as get_set
from tools.train import train as fun_train
from tools.test import test as fun_test
import imageio
from tools.show import *
from tools.assessment import *
from model.model_cfg import get_model
import random


def seed_torch(seed=1):
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------defined paramaters------------------ #
# # Setting Params
parser = argparse.ArgumentParser(description='Training for change detection')
parser.add_argument('-d', '--dataset', dest='dataset', choices=['farmland', 'river', 'USA'],
                    default='USA', help="Name of dataset.")
parser.add_argument('-m', '--model_name', type=str, default='GDAMamba', help='model used')
parser.add_argument('--seed', type=int, default=1, help='number of seed')

args = parser.parse_args()


def main():
    if args.dataset == 'farmland':
        import configs.GDAMamba.configs_farmland as cfg
    elif args.dataset == 'river':
        import configs.GDAMamba.configs_river as cfg
    elif args.dataset == 'USA':
        import configs.GDAMamba.configs_USA as cfg

    current_dataset = cfg.current_dataset
    current_model = cfg.current_model
    model_name = current_dataset + current_model
    print('model: {}'.format(model_name))

    cfg_data = cfg.data
    cfg_train = cfg.train['train_model']
    cfg_optim = cfg.train['optimizer']
    cfg_test = cfg.test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_torch(seed=args.seed)

    # Store
    save_folder = cfg_test['save_folder']
    model_name = cfg.current_model.replace('_', '')

    save_path = os.path.join(save_folder, model_name + '_Samples_%s' % str(cfg.train_set_num))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mat_save_path = os.path.join(save_path, 'Mat')
    if not os.path.exists(mat_save_path):
        os.makedirs(mat_save_path)

    weight_save_path = os.path.join(save_path, 'Weights')
    if not os.path.exists(weight_save_path):
        os.makedirs(weight_save_path)

    fig_save_path = os.path.join(save_path, 'Fig')
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)

    data_sets = get_set(cfg_data)
    img_gt = data_sets['img_gt']
    train_data_label = generate_label_image_from_real_labels(sample_coords=data_sets['train_sample_center'],
                                                             real_labels=img_gt)
    test_data_label = generate_label_image_from_real_labels(sample_coords=data_sets['test_sample_center'],
                                                            real_labels=img_gt)
    data_sets['train_data_label'] = train_data_label
    data_sets['test_data_label'] = test_data_label

    data_sets['img1'] = data_sets['img1'].unsqueeze(0)
    data_sets['img2'] = data_sets['img2'].unsqueeze(0)

    model = get_model(cfg, device)
    loss_fun = nn.CrossEntropyLoss(reduction='none')
    if cfg_optim['typename'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg_optim['lr'], momentum=cfg_optim['momentum'],
                              weight_decay=cfg_optim['weight_decay'])
    elif cfg_optim['typename'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg_optim['lr'],
                               weight_decay=cfg_optim['weight_decay'])
    elif cfg_optim['typename'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=cfg_optim['lr'],
                                weight_decay=cfg_optim['weight_decay'])

    fun_train(data_sets, model, loss_fun, optimizer, device, cfg_train, model_save_path=weight_save_path)

    # test
    pred_train_label, pred_train_acc = fun_test(data_sets, model, device, cfg_test,
                                                model_save_path=weight_save_path, test_type='train_acc')
    pred_test_label, pred_test_acc = fun_test(data_sets, model, device, cfg_test,
                                              model_save_path=weight_save_path, test_type='test_acc')
    pred_label, pred_acc = fun_test(data_sets, model, device, cfg_test,
                                    model_save_path=weight_save_path, test_type='all_acc')

    print('pred_train_acc {:.2f}%, pred_test_acc {:.2f}%, pred_all_acc {:.2f}%'.format(pred_train_acc,
                                                                                       pred_test_acc,
                                                                                       pred_acc))
    predict_img = pred_label.squeeze(0).cpu()

    FP = (predict_img == 1) & (img_gt == 0)
    FN = (predict_img == 0) & (img_gt == 1)
    change = np.zeros(predict_img.shape + (3,), dtype=np.uint8) + 128
    change[FP] = [255, 0, 0]
    change[FN] = [0, 0, 255]
    TP = (predict_img == 1) & (img_gt == 1)
    TN = (predict_img == 0) & (img_gt == 0)

    change[TP] = [255, 255, 255]
    change[TN] = [0, 0, 0]
    conf_mat, oa, kappa_co, P, R, F1, acc = accuracy_assessment(img_gt, predict_img)
    assessment_result = [round(oa, 4) * 100, round(kappa_co, 4), round(F1, 4) * 100, round(P, 4) * 100,
                         round(R, 4) * 100, model_name]
    print('assessment_result', assessment_result)

    # 保存
    save_name = cfg_test['save_name']

    io.savemat(mat_save_path + '/' + save_name + ".mat",
               {"predict_img": np.array(predict_img.cpu()), "oa": assessment_result})
    imageio.imwrite(fig_save_path + '/' + save_name + '_predict_img.png', change)

    # 示例字典
    data = {'confuse_matrix': conf_mat, 'OA': round(oa, 4) * 100, 'Kappa': round(kappa_co, 4),
            'F1': round(F1, 4) * 100, 'P': round(P, 4) * 100, 'R': round(R, 4) * 100, 'ACC': round(acc, 4) * 100}

    # 将字典保存为txt文件
    with open(save_path + '/' + save_name + ".txt", 'a') as file:
        for key, value in data.items():
            file.write(f"{key}:\n{value}\n")
    print('save predict_img successful!')


if __name__ == '__main__':
    main()
