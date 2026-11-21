import timm


def build_network(model_name, num_classes, img_size=224, pretrained=True):
    kwargs = {}
    if timm.get_pretrained_cfg(model_name).fixed_input_size:
        kwargs["img_size"] = img_size
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)


def split_parameters(network):
    """Returns (backbone params, head params)."""
    head = list(network.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in network.parameters() if id(p) not in head_ids]
    return backbone, head
