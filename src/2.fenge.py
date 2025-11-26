import os
import shutil

import pandas as pd
from tqdm import tqdm

# 路径设置
base_dir = "/home/mao/Pictures/0928mv/20250928_075419"
output_dir = f"{base_dir}/output"
basename = os.path.basename(base_dir)
range_start = 7921  # 起始帧
range_end = 8069  # 结束帧

# 创建输出目录
os.makedirs(output_dir + "/left/csv", exist_ok=True)
os.makedirs(output_dir + "/left/frame", exist_ok=True)
os.makedirs(output_dir + "/right/csv", exist_ok=True)
os.makedirs(output_dir + "/right/frame", exist_ok=True)

# 获取所有带 left 或 right 的 CSV 文件，并按 left 和 right 分组
# 参考文件名格式 0928mv_20250928_075419_left_json.csv
csv_files = sorted(
    [
        f
        for f in os.listdir(base_dir)
        if f.lower().endswith(".csv") and ("left" in f or "right" in f)
    ]
)
assert len(csv_files) == 2
left_files = [f for f in csv_files if "left" in f.lower()]
right_files = [f for f in csv_files if "right" in f.lower()]
assert len(left_files) == len(right_files) == 1


# 批量处理 CSV 文件
for file in tqdm(csv_files, total=len(csv_files), desc=base_dir):
    file_path = os.path.join(base_dir, file)

    # 打印每个文件的路径，确保文件被正确读取
    print(f"正在处理文件: {file_path}")

    # 添加 skip_blank_lines=True 来跳过空行
    try:
        df = pd.read_csv(file_path, skip_blank_lines=True)
        print(f"成功读取文件 {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件 {file_path} 是空的，跳过处理该文件")
        continue  # 跳过这个文件，继续处理下一个文件
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误: {e}")
        continue

    if df.empty:
        print(f"文件 {file_path} 读取后为空，跳过处理该文件")
        continue

    df_range = df[(df["Frame"] >= range_start) & (df["Frame"] <= range_end)].copy()
    print(df_range)

    if "left" in file.lower():
        prefix = "left"
    else:
        prefix = "right"

    # 生成新的文件名
    new_csv_filename = f"{output_dir}/{prefix}/csv/{basename}_{range_start:06d}_{range_end:06d}_ball.csv"
    new_jpg_dir = (
        f"{output_dir}/{prefix}/frame/{basename}_{range_start:06d}_{range_end:06d}"
    )
    shutil.rmtree(new_jpg_dir, ignore_errors=True)
    os.makedirs(new_jpg_dir, exist_ok=True)
    # df_range["Frame"] = df_range["Frame"].astype(str)

    for pno, (index, row) in enumerate(df_range.iterrows()):
        df_range.loc[index, "Frame"] = pno
        image_name = f"{pno}.jpg"

        shutil.copy(
            row["path"],
            os.path.join(new_jpg_dir, image_name),
        )

    df_range.to_csv(new_csv_filename, index=False)
    print(f"保存 CSV 文件: {new_csv_filename}")
