from .component import (
    encoder_dict, 
    decoder_dict
)
from .conv_onet import ConvolutionalOccupancyNetwork


def get_conv_onet(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']

    # for pointcloud_crop
    try:
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except KeyError:
        pass
    # local positional encoding
    if 'local_coord' in cfg['model'].keys():
        encoder_kwargs['local_coord'] = cfg['model']['local_coord']
        decoder_kwargs['local_coord'] = cfg['model']['local_coord']
    if 'pos_encoding' in cfg['model']:
        encoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    assert cfg['data']['input_type'] != 'pointcloud_crop'

    decoder = decoder_dict[decoder](
        dim=dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )

    assert encoder != 'idx'
    if encoder is not None:
        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    model = ConvolutionalOccupancyNetwork(
        decoder, encoder, device=device
    )

    return model


def update_reso(reso, depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        reso (int): defined resolution
        depth (int): U-Net number of layers
    '''
    base = 2**(int(depth) - 1)
    if ~(reso / base).is_integer(): # when this is not integer, U-Net dimension error
        for i in range(base):
            if ((reso + i) / base).is_integer():
                reso = reso + i
                break    
    return reso
