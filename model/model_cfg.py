
def get_model(cfg, device, **kwargs):
    from model.GDAMamba import GDAMamba as fun_model
    model = fun_model(input_dim=cfg.model['bands'], num_classes=cfg.model['num_classes'],
                      hidden_dim=32, blocks=3).to(device)
    return model
