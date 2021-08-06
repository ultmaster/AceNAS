from .network import NDS, NdsConfig


def darts_indices_to_spec(indices, searchspace):
    def _convert(key, value):
        if value.isdigit():
            result[key] = searchspace[key][int(value)]
        else:
            assert value in searchspace[key]
            result[key] = value

    indices = indices.split(':')
    result = {}
    for i in range(0, 16, 4):
        _convert(f'normal_{i // 4}_x_op', indices[i])
        result[f'normal_{i // 4}_x_input'] = int(indices[i + 1])
        _convert(f'normal_{i // 4}_y_op', indices[i + 2])
        result[f'normal_{i // 4}_y_input'] = int(indices[i + 3])
    for i in range(16, 32, 4):
        _convert(f'reduce_{(i - 16) // 4}_x_op', indices[i])
        result[f'reduce_{(i - 16) // 4}_x_input'] = int(indices[i + 1])
        _convert(f'reduce_{(i - 16) // 4}_y_op', indices[i + 2])
        result[f'reduce_{(i - 16) // 4}_y_input'] = int(indices[i + 3])
    if 'depth' in searchspace and 'width' in searchspace:
        result['depth'] = searchspace['depth'][0]
        result['width'] = searchspace['width'][0]
    return result


def darts_spec_to_indices(spec, searchspace):
    indices = []
    for cell in ['normal', 'reduce']:
        for node_id in range(0, 4):
            for subnode in ['x', 'y']:
                op_key = f'{cell}_{node_id}_{subnode}_op'
                indices.append(searchspace[op_key].index(spec[op_key]))
                indices.append(spec[f'{cell}_{node_id}_{subnode}_input'])
    return ':'.join(map(str, indices))


def create_darts_model(indices,
                       init_channels=36,
                       num_layers=20,
                       model_type='cifar'):
    config = NdsConfig(
        init_channels=[init_channels],
        num_layers=[num_layers],
        model_type=model_type,
        n_nodes=4,
        concat_all=True,
        op_candidates=[
            'max_pool_3x3',
            'avg_pool_3x3',
            'skip_connect',
            'sep_conv_3x3',
            'sep_conv_5x5',
            'dil_conv_3x3',
            'dil_conv_5x5',
            'none'
        ],
        use_aux=True
    )

    model = NDS(config)
    if indices is None:
        return model
    model.activate(darts_indices_to_spec(indices, model.searchspace()))
    model.prune()
    return model
