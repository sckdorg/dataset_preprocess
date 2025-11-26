import os

import cv2
import pandas as pd

# 路径设置
csv_dir = "/workspaces/TrackNetV3/datasets-faqiu/wangqiu/4926/csv"
frame_root_dir = "/workspaces/TrackNetV3/datasets-faqiu/wangqiu/4926/frame"
output_root_dir = "/workspaces/TrackNetV3/datasets-faqiu/wangqiu/output1"

# 遍历 CSV 文件
for csv_file in os.listdir(csv_dir):
    if not csv_file.endswith(".csv"):
        continue

    csv_path = os.path.join(csv_dir, csv_file)
    name_parts = os.path.splitext(csv_file)[0].split("_")
    if len(name_parts) < 3:
        print(f"⚠️ 文件名格式错误：{csv_file}，跳过")
        continue

    frame_folder_name = "_".join(name_parts[:3])
    frame_dir = os.path.join(frame_root_dir, frame_folder_name)
    output_dir = os.path.join(output_root_dir, frame_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(frame_dir):
        print(f"⚠️ 找不到帧目录：{frame_dir}，跳过")
        continue

    print(f"\n🟢 正在处理：{csv_file}")
    df = pd.read_csv(csv_path)

    # 筛选出可见目标
    df_visible = df[df["Visibility"] == 1]

    # 为了避免重复读写，每帧只读一次图像并画所有框
    frame_groups = df_visible.groupby("Frame")

    for frame_id, group in frame_groups:
        frame_name = f"{int(frame_id)}.png"
        frame_path = os.path.join(frame_dir, frame_name)

        if not os.path.exists(frame_path):
            print(f"❌ 缺失图像：{frame_path}，跳过")
            continue

        img = cv2.imread(frame_path)
        if img is None:
            print(f"❌ 图像无法读取：{frame_path}，跳过")
            continue

        for _, row in group.iterrows():
            cx = row["X"]
            cy = row["Y"]
            w = row["Width"]
            h = row["Height"]

            # 由中心点计算出边界框坐标
            xmin = int(cx - w / 2)
            ymin = int(cy - h / 2)
            xmax = int(cx + w / 2)
            ymax = int(cy + h / 2)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 保存图像
        output_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(output_path, img)
        print(f"✅ 已保存: {output_path}")

print("\n🎉 所有目标框绘制完成！")
