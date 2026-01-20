current_dataset = 'Farmland'
current_model = '_GDAMamba'

current = current_dataset + current_model

patch_size = 7
lr = 5e-3
epoch_number = 100
num_iters = 5

phase = ['train', 'test', 'no_gt']
train_set_num = 0.01

data = dict(
    current_dataset=current_dataset,
    train_set_num=train_set_num,
    patch_size=patch_size,

    train_data=dict(
        phase=phase[0]
    ),
    test_data=dict(
        phase=phase[1]
    ),
)

# 2. model
model = dict(
    bands=155,
    num_classes=2,
)

# 3. train
train = dict(
    optimizer=dict(
        typename='Adam',
        lr=lr,
        momentum=0.9,
        weight_decay=5e-3
    ),
    train_model=dict(
        gpu_train=True,
        gpu_num=1,
        workers_num=12,
        epoch=epoch_number,
        lr=lr,
        lr_adjust=True,
        lr_gamma=0.1,
        lr_step=[35, 70],
        save_folder='./weights/' + current_dataset + '/',
        save_name=current,
        reuse_model=False,
        reuse_file='./weights/' + current + '_Final.pth',
        lambdas1=0.1,
        lambdas2=0.2,
    ),
)

# 4. test
test = dict(
    batch_size=1000,
    gpu_train=True,
    gpu_num=0,
    workers_num=8,
    model_weights='./weights/' + current_dataset + '/' + current + '_Final.pth',
    save_name=current,
    save_folder='./result' + '/' + current_dataset
)



