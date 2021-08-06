from .network import ProxylessNAS, ProxylessConfig
from .utils import tf_indices_to_pytorch_spec


def create_proxylessnas_model(indices,
                              widths=None,
                              num_classes=1000,
                              drop_rate=0.0,
                              bn_momentum=0.1,
                              bn_eps=1e-3,
                              **kwargs):
    if widths is None:
        widths = [16, 32, 40, 80, 96, 192, 320]
    config = ProxylessConfig(
        stem_width=32,
        final_width=1280,
        width_mult=1.0,
        num_labels=num_classes,
        dropout_rate=drop_rate,
        stages=[
            {'depth_range': [1, 1], 'exp_ratio_range': [1], 'kernel_size_range': [3], 'width': widths[0], 'downsample': False},
            {'depth_range': [1, 4], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[1], 'downsample': True},
            {'depth_range': [1, 4], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[2], 'downsample': True},
            {'depth_range': [1, 4], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[3], 'downsample': True},
            {'depth_range': [1, 4], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[4], 'downsample': False},
            {'depth_range': [1, 4], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[5], 'downsample': True},
            {'depth_range': [1, 1], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[6], 'downsample': False}
        ]
    )
    model = ProxylessNAS(config, reset_parameters=False)
    model.reset_parameters(bn_momentum=bn_momentum, bn_eps=bn_eps, track_running_stats=True)
    model.activate(tf_indices_to_pytorch_spec(indices, model.searchspace()))
    model.prune()
    return model
