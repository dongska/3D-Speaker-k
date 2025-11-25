import torch
import torchaudio
from speakerlab.utils.builder import build
from speakerlab.utils.config import build_config

# ===== 修改为你自己的路径 =====
exp_dir = "/root/private_data/g813_u1/dsk/3D-Speaker/egs/voxceleb/sv-cam++/exp/cam++_256"
test_wav = "/root/private_data/g813_u1/dsk/3D-Speaker/egs/voxceleb/sv-camPPECA/data/raw_data/voxceleb1/test/id10270/5r0dWxy17C8/00001.wav"



import torch
import torchaudio
from speakerlab.utils.builder import build
from speakerlab.utils.config import build_config



# ============ 打印 hook ==============
def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            print(f"[{name:40s}] "
                  f"input: {tuple(i.shape for i in input)} "
                  f"output: {output.shape}")
    return hook


# ============ 加载模型（CPU） ==============
def load_model(exp_dir):
    config = build_config(f"{exp_dir}/config.yaml")
    model = build("embedding_model", config)
    config.checkpointer["args"]["checkpoints_dir"] = f"{exp_dir}/models"
    config.checkpointer["args"]["recoverables"] = {"embedding_model": model}
    checkpointer = build("checkpointer", config)
    checkpointer.recover_if_possible(epoch=config.num_epoch, device=torch.device("cpu"))
    return model.to("cpu").eval(), config


def attach_hooks(model):
    for name, module in model.named_modules():
        # 跳过整个模型本身，否则输出太多
        if name == "":
            continue
        module.register_forward_hook(make_hook(name))


def test_forward(model, config, wav_path):
    wav, fs = torchaudio.load(wav_path)
    assert fs == config.sample_rate

    # 只裁切 / padding 到 3 秒，不做 sliding
    target_len = int(3.0 * fs)
    cur_len = wav.shape[1]
    if cur_len < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - cur_len))
    else:
        wav = wav[:, :target_len]

    feature_extractor = build("feature_extractor", config)
    feat = feature_extractor(wav).unsqueeze(0)

    print("\n========== 开始 forward，下面会打印所有层的 shape ==========\n")
    emb = model(feat)
    print("\n========== 最终输出 shape ==========")
    print(emb.shape)


if __name__ == "__main__":
    model, config = load_model(exp_dir)
    attach_hooks(model)
    test_forward(model, config, test_wav)
