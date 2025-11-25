import re
import csv

input_file = "/root/private_data/g813_u1/dsk/3D-Speaker/egs/voxceleb/sv-camPPECA/exp/camPPECA_alpha0/models/CKPT-EPOCH-5-00/alpha"      # 输入文件
output_file = "/root/private_data/g813_u1/dsk/3D-Speaker/egs/voxceleb/sv-camPPECA/exp/camPPECA_alpha0/epoch5_alpha.csv"  # 输出 CSV 文件

pattern = re.compile(
    r"(xvector\.block[1-3]\.tdnnd\d+\.cam_layer\.alpha)\s+(-?[0-9]*\.?[0-9]+)"
)


rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            layer_name = match.group(1)
            alpha_value = float(match.group(2))
            rows.append([layer_name, alpha_value])

# 写入 CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Layer", "Alpha"])
    writer.writerows(rows)

print("提取完成！输出文件：", output_file)
