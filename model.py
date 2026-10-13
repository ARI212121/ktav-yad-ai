# -*- coding: utf-8 -*-
"""רשת CRNN (רשת קונבולוציה + BiLSTM + CTC) לזיהוי שורות כתב יד.
קלט: תמונת גווני אפור בגובה 48 פיקסלים, רוחב משתנה.
פלט: רצף הסתברויות תווים לאורך ציר הרוחב (רוחב/4 צעדי זמן).
"""
import os
import torch
import torch.nn as nn

IMG_H = 48
DOWNSAMPLE_W = 4  # פקטור הקטנת הרוחב ברשת


class CRNN(nn.Module):
    def __init__(self, num_classes, nh=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                       # 48 -> 24, W -> W/2
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                       # 24 -> 12, W -> W/4
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),                     # 12 -> 6
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),                     # 6 -> 3
            nn.Conv2d(256, 256, (3, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 3 -> 1
        )
        self.rnn = nn.LSTM(256, nh, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(nh * 2, num_classes)

    def forward(self, x):                # x: (B, 1, 48, W)
        f = self.cnn(x)                  # (B, 256, 1, W/4)
        f = f.squeeze(2).permute(0, 2, 1)  # (B, T, 256)
        r, _ = self.rnn(f)
        return self.fc(r)                # (B, T, num_classes)


def save_checkpoint(path, model, charset, step=0, meta=None, training_state=None):
    checkpoint = {"model": model.state_dict(), "charset": charset,
                  "step": step, "meta": meta or {}}
    if training_state is not None:
        checkpoint["training_state"] = training_state
    temporary = path + ".tmp"
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_checkpoint(path, num_classes, charset):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    old_charset = ckpt.get("charset")
    old_state = ckpt.get("model")
    if not isinstance(old_charset, str) or not isinstance(old_state, dict):
        raise ValueError(f"invalid checkpoint in {path}")
    if old_charset != charset and not charset.startswith(old_charset):
        raise ValueError(f"charset mismatch in {path}")
    model = CRNN(num_classes)
    if old_charset == charset:
        model.load_state_dict(old_state)
    else:
        expanded = model.state_dict()
        for name, tensor in old_state.items():
            if name == "fc.weight":
                if (tensor.shape[1:] != expanded[name].shape[1:]
                        or tensor.shape[0] != len(old_charset) + 1):
                    raise ValueError(f"incompatible classifier in {path}")
                expanded[name][:tensor.shape[0]].copy_(tensor)
            elif name == "fc.bias":
                if tensor.shape[0] != len(old_charset) + 1:
                    raise ValueError(f"incompatible classifier in {path}")
                expanded[name][:tensor.shape[0]].copy_(tensor)
            elif name in expanded and tensor.shape == expanded[name].shape:
                expanded[name].copy_(tensor)
            else:
                raise ValueError(f"incompatible model weights in {path}: {name}")
        added_rows = len(old_charset) + 1
        expanded["fc.weight"][added_rows:].zero_()
        expanded["fc.bias"][added_rows:].fill_(-5.0)
        model.load_state_dict(expanded)
        ckpt["charset_upgraded_from"] = old_charset
        ckpt["charset"] = charset
    return model, ckpt


if __name__ == "__main__":
    from charset import NUM_CLASSES
    m = CRNN(NUM_CLASSES)
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 1, IMG_H, 320)
    y = m(x)
    print(f"params: {n_params/1e6:.1f}M, in {tuple(x.shape)} -> out {tuple(y.shape)}")
