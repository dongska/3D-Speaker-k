import torch
import torchaudio
from speakerlab.utils.builder import build
from speakerlab.utils.config import build_config

# ===== 修改为你自己的路径 =====
exp_dir = "/root/private_data/g813_u1/dsk/3D-Speaker/egs/voxceleb/sv-camPPECA/exp/camPPECA_alpha0"
test_wav = "/root/private_data/g813_u1/dsk/3D-Speaker/egs/voxceleb/sv-camPPECA/data/raw_data/voxceleb1/test/id10270/5r0dWxy17C8/00001.wav"



# ===========================
#   加载模型（CPU）
# ===========================
def load_model_cpu(exp_dir):
    config_file = f"{exp_dir}/config.yaml"
    config = build_config(config_file)

    embedding_model = build("embedding_model", config)

    config.checkpointer["args"]["checkpoints_dir"] = f"{exp_dir}/models"
    config.checkpointer["args"]["recoverables"] = {
        "embedding_model": embedding_model
    }
    checkpointer = build("checkpointer", config)

    # 强制加载到 CPU
    checkpointer.recover_if_possible(
        epoch=config.num_epoch,
        device=torch.device("cpu")
    )

    embedding_model.to("cpu").eval()
    return embedding_model, config


# ===========================
#   裁切音频为 3 秒 segment
# ===========================
def cut_to_3s_segments(wav, fs):
    target_len = int(3.0 * fs)
    hop_len = int(1.5 * fs)
    cur_len = wav.shape[1]

    segments = []

    # 长音频 → 滑动窗口
    if cur_len > target_len:
        start = 0
        while start + target_len <= cur_len:
            seg = wav[:, start:start + target_len]
            segments.append(seg)
            start += hop_len

        # 最后一段不足 3s → padding
        if start < cur_len:
            last = wav[:, start:]
            pad_len = target_len - last.shape[1]
            last = torch.nn.functional.pad(last, (0, pad_len))
            segments.append(last)

    else:
        # 短音频 → padding 到 3s
        pad_len = target_len - cur_len
        seg = torch.nn.functional.pad(wav, (0, pad_len))
        segments.append(seg)

    return segments


# ===========================
#    执行一次完整前向传播
# ===========================
def test_forward_cpu(exp_dir, wav_path):
    # 加载模型
    embedding_model, config = load_model_cpu(exp_dir)

    # 加载 wav
    wav, fs = torchaudio.load(wav_path)
    assert fs == config.sample_rate, f"sample rate mismatch: {fs}"

    print(f"原始长度: {wav.shape[1] / fs:.2f}s")

    # --------------------------
    # 裁切成 3 秒 segments
    # --------------------------
    segments = cut_to_3s_segments(wav, fs)
    print(f"切出的 segment 数: {len(segments)}")

    feature_extractor = build("feature_extractor", config)

    emb_list = []

    # --------------------------
    # 逐段计算 embedding
    # --------------------------
    for idx, seg in enumerate(segments):
        feat = feature_extractor(seg)          # (T, F)
        feat = feat.unsqueeze(0)               # batch
        emb = embedding_model(feat)            # (1, C)
        emb_list.append(emb.detach().cpu())

        print(f"segment {idx}: 特征 {feat.shape} → embedding {emb.shape}")

    # --------------------------
    # 多段平均
    # --------------------------
    emb_final = torch.mean(torch.stack(emb_list), dim=0)

    print("\n====== 最终 embedding ======")
    print("shape:", emb_final.shape)
    print(emb_final)

    return emb_final


# ===========================
#           运行
# ===========================
if __name__ == "__main__":
    test_forward_cpu(exp_dir, test_wav)
