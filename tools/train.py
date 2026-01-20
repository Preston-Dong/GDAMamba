import torch
import time
import datetime
import math
import os

from torch.utils.data import DataLoader


def adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, step_index):
    if epoch < 1:
        lr = 0.0001 * lr_init
    elif epoch <= step_index[0]:
        lr = lr_init
    elif epoch <= step_index[1]:
        lr = lr_init * lr_gamma
    elif epoch > step_index[1]:
        lr = lr_init * lr_gamma ** 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train(data_sets, model, loss_fun, optimizer, device, cfg, model_save_path):
    torch.autograd.set_detect_anomaly(True)
    num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']

    save_name = cfg['save_name']

    lr_init = cfg['lr']
    lr_gamma = cfg['lr_gamma']
    lr_step = cfg['lr_step']
    lr_adjust = cfg['lr_adjust']

    epoch_size = cfg['epoch']

    # gpu_num
    if gpu_num > 1 and cfg['gpu_train']:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    '''# Load the model and start training'''
    model.train()

    if cfg['reuse_model']:
        print('load model...')
        checkpoint = torch.load(cfg['reuse_file'], map_location=device)
        start_epoch = checkpoint['epoch']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        start_epoch = 0

    batch_num = 1
    train_loss_save = []
    train_acc_save = []

    print('start training...')

    for epoch in range(start_epoch + 1, epoch_size + 1):

        epoch_time0 = time.time()
        epoch_loss = 0

        if lr_adjust:
            lr = adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, lr_step)
        else:
            lr = lr_init

        img1 = data_sets['img1']
        img2 = data_sets['img2']

        target = data_sets['train_data_label']
        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)
        prediction = model(img1, img2)

        # 创建掩码，只有有标签的像素为 1，其他无标签的像素为 0
        mask = (target != -1).float()  # 只有值不为 -1 的部分是有效标签

        # 由于预测结果是 (N, C, H, W)，可以直接计算交叉熵损失
        # labels 是形状 (H, W)，需要把它调整成 (N, H, W) 进行匹配
        targets = torch.where(target == -1, torch.tensor(0.0).to(device), target).unsqueeze(0)  # 形状变为 (1, H, W)
        loss = loss_fun(prediction, targets.long())
        # 使用掩码过滤无标签区域，确保无标签区域不参与损失计算
        masked_loss = loss.squeeze(0) * mask

        # 计算有效标签的平均损失
        mean_loss = masked_loss.sum() / mask.sum()

        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()
        epoch_loss += mean_loss.item()
        # 计算预测的类别
        predict_label = torch.argmax(prediction, dim=1)  # 预测类别，形状为 (N, H, W)
        # 计算准确率，只有有效标签区域才参与比较
        correct = (predict_label == targets).float() * mask  # 计算预测是否正确并应用掩码
        train_acc = correct.sum() / mask.sum() * 100  # 计算有效标签区域的准确率

        epoch_time = time.time() - epoch_time0
        epoch_eta = int(epoch_time * (epoch_size - epoch))

        print('Epoch: {}/{} || lr: {} || loss: {} || Train acc: {:.2f}% || '
              'Epoch time: {:.4f}s || Epoch ETA: {}'
              .format(epoch, epoch_size, lr, epoch_loss / batch_num, train_acc,
                      epoch_time, str(datetime.timedelta(seconds=epoch_eta))
                      )
              )

        train_loss_save.append(epoch_loss / batch_num)
        train_acc_save.append(train_acc)

    # Store the final model
    save_model = dict(
        model=model.state_dict(),
        epoch=epoch_size
    )
    torch.save(save_model, os.path.join(model_save_path, save_name + '_Final.pth'))
