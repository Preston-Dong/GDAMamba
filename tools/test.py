import torch
from torch.utils.data import DataLoader
import os


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix前缀 'module.' '''
    print('remove prefix \'{}\''.format(prefix))

    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)['model']
    else:
        device = torch.cuda.current_device()  # gpu
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))['model']

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def test(data_sets, model, device, cfg, model_save_path, test_type='train_acc'):
    num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']

    model = load_model(model, os.path.join(model_save_path, cfg['save_name'] + '_Final.pth'), device)
    model.eval()
    model = model.to(device)

    if gpu_num > 1 and cfg['gpu_train']:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # Data load
    img1 = data_sets['img1']
    img2 = data_sets['img2']
    if test_type == 'train_acc':
        target = data_sets['train_data_label']
    elif test_type == 'test_acc':
        target = data_sets['test_data_label']
    elif test_type == 'all_acc':
        target = data_sets['img_gt']
    img1 = img1.to(device)
    img2 = img2.to(device)
    target = target.to(device)
    # 创建掩码，只有有标签的像素为 1，其他无标签的像素为 0
    mask = (target != -1).float()  # 只有值不为 -1 的部分是有效标签
    # labels 是形状 (H, W)，需要把它调整成 (N, H, W) 进行匹配
    targets = torch.where(target == -1, torch.tensor(0.0).to(device), target).unsqueeze(0)  # 形状变为 (1, H, W)

    with torch.no_grad():
        prediction = model(img1, img2)
    # 计算预测的类别
    predict_label = torch.argmax(prediction, dim=1)  # 预测类别，形状为 (N, H, W)
    # 计算准确率，只有有效标签区域才参与比较
    correct = (predict_label == targets).float() * mask  # 计算预测是否正确并应用掩码
    test_acc = correct.sum() / mask.sum() * 100  # 计算有效标签区域的准确率
    print('OA {:.2f}%'.format(test_acc))

    return predict_label, test_acc
