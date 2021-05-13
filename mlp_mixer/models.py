from mlp_mixer import MLPMixer


def mlp_mixer_s16(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=16, num_layers=8, hidden_dim=512,
                  tokens_hidden_dim=256, channels_hidden_dim=2048)
    return MLPMixer(num_classes, image_size, channels, **params)

def mlp_mixer_s32(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=32, num_layers=8, hidden_dim=512,
                  tokens_hidden_dim=256, channels_hidden_dim=2048)
    return MLPMixer(num_classes, image_size, channels, **params)

def mlp_mixer_b16(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=16, num_layers=12, hidden_dim=768,
                  tokens_hidden_dim=384, channels_hidden_dim=3072)
    return MLPMixer(num_classes, image_size, channels, **params)

def mlp_mixer_b32(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=32, num_layers=12, hidden_dim=768,
                  tokens_hidden_dim=384, channels_hidden_dim=3072)
    return MLPMixer(num_classes, image_size, channels, **params)

def mlp_mixer_l16(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=16, num_layers=24, hidden_dim=1024,
                  tokens_hidden_dim=512, channels_hidden_dim=4096)
    return MLPMixer(num_classes, image_size, channels, **params)

def mlp_mixer_l32(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=32, num_layers=24, hidden_dim=1024,
                  tokens_hidden_dim=512, channels_hidden_dim=4096)
    return MLPMixer(num_classes, image_size, channels, **params)

def mlp_mixer_h14(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=14, num_layers=32, hidden_dim=1280,
                  tokens_hidden_dim=640, channels_hidden_dim=5120)
    return MLPMixer(num_classes, image_size, channels, **params)
