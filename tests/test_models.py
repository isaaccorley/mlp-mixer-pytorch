import pytest

import torch
import mlp_mixer


DEVICE = "cpu"
PARAMS = dict(num_classes=10, image_size=224, channels=3)
x = torch.randn(1, PARAMS["channels"], PARAMS["image_size"], PARAMS["image_size"])
x = x.to(torch.float32)
x = x.to(DEVICE)


def test_mlp_mixer_s16():
    model = mlp_mixer.mlp_mixer_s16(**PARAMS)
    model = model.to(DEVICE)
    y_hat = model(x)
    assert y_hat.shape == (1, PARAMS["num_classes"])
    assert y_hat.dtype == torch.float32

def test_mlp_mixer_s32():
    model = mlp_mixer.mlp_mixer_s32(**PARAMS)
    model = model.to(DEVICE)
    y_hat = model(x)
    assert y_hat.shape == (1, PARAMS["num_classes"])
    assert y_hat.dtype == torch.float32

def test_mlp_mixer_b16():
    model = mlp_mixer.mlp_mixer_b16(**PARAMS)
    model = model.to(DEVICE)
    y_hat = model(x)
    assert y_hat.shape == (1, PARAMS["num_classes"])
    assert y_hat.dtype == torch.float32

def test_mlp_mixer_b32():
    model = mlp_mixer.mlp_mixer_b32(**PARAMS)
    model = model.to(DEVICE)
    y_hat = model(x)
    assert y_hat.shape == (1, PARAMS["num_classes"])
    assert y_hat.dtype == torch.float32

def test_mlp_mixer_l16():
    model = mlp_mixer.mlp_mixer_l16(**PARAMS)
    model = model.to(DEVICE)
    y_hat = model(x)
    assert y_hat.shape == (1, PARAMS["num_classes"])
    assert y_hat.dtype == torch.float32

def test_mlp_mixer_l32():
    model = mlp_mixer.mlp_mixer_l32(**PARAMS)
    model = model.to(DEVICE)
    y_hat = model(x)
    assert y_hat.shape == (1, PARAMS["num_classes"])
    assert y_hat.dtype == torch.float32

def test_mlp_mixer_h14():
    model = mlp_mixer.mlp_mixer_h14(**PARAMS)
    model = model.to(DEVICE)
    y_hat = model(x)
    assert y_hat.shape == (1, PARAMS["num_classes"])
    assert y_hat.dtype == torch.float32
