import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
import cv2
import pandas as pd
from tqdm import tqdm

# 全局变量
S3_BUCKET = "lstudio"
S3_PREFIX = "badminton20250708"  # 更新为你的日期
IMAGE_EXTENSIONS = {".jpg", ".png"}

# 输入数据
DATA_DIR = "/home/mao/jzd/tracknetv3/datasets/0928mv/20250928_075419"
# 输出 CSV 目录
CSV_OUTPUT_DIR = DATA_DIR


def folder_contains_image(folder_path):
    """判断一个文件夹是否包含图片文件"""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                return True
    return False


def find_folders_with_images(root_folder):
    """递归遍历文件夹，筛选包含图片的文件夹（仅 left 和 right）"""
    folders_with_images = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if os.path.basename(dirpath) in ["left", "right"]:
            if any(
                os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS for f in filenames
            ):
                folders_with_images.append(dirpath)
    return folders_with_images


def make_dir(output_dir):
    """创建目录，如果存在则删除后重新创建"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


# 初始化背景减除器
left_bgs = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=6, detectShadows=True
)

right_bgs = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=6, detectShadows=True
)


def json_to_csv(image_dir, json_path):
    """将 JSON 文件转换为 CSV 文件，包含边界框的宽和高"""
    json_files = sorted(
        [f for f in os.listdir(json_path) if f.lower().endswith(".json")]
    )
    boxes = []
    basname = os.path.basename(json_path)
    indexs = [int(i[: -1 * len(".json")]) for i in json_files]
    max_index = max(indexs) if indexs else 0
    min_index = min(indexs) if indexs else 0

    for current_frame in tqdm(range(min_index, max_index), desc=basname):
        json_file_path = os.path.join(json_path, f"{current_frame:06d}.json")
        try:
            with open(json_file_path, "r", encoding="utf-8") as file1:
                data = json.load(file1)
        except Exception:
            # print(f"处理 JSON 文件失败：{json_file_path}, 错误：{e}")
            boxes.append(
                {
                    "Frame": current_frame,
                    "Visibility": 0,
                    "X": 0,
                    "Y": 0,
                    "W": 0,
                    "H": 0,
                    "path": f"{image_dir}/{current_frame:06d}.jpg",
                }
            )
            continue

        if not data or "annotations" not in data[0]:
            print(f"跳过无效 JSON 文件：{json_file_path}")
            boxes.append(
                {
                    "Frame": current_frame,
                    "Visibility": 0,
                    "X": 0,
                    "Y": 0,
                    "W": 0,
                    "H": 0,
                    "path": f"{image_dir}/{current_frame:06d}.jpg",
                }
            )
            continue

        predictions = data[0]["annotations"][0]["result"]
        value = predictions[0]["value"]
        x, y, width, height = (
            value["x"],
            value["y"],
            value["width"],
            value["height"],
        )
        original_width = predictions[0]["original_width"]
        original_height = predictions[0]["original_height"]
        center_x_pixel = int(round((x + width / 2) * original_width / 100))
        center_y_pixel = int(round((y + height / 2) * original_height / 100))
        # 计算边界框的像素宽和高
        width_pixel = int(round(width * original_width / 100))
        height_pixel = int(round(height * original_height / 100))

        boxes.append(
            {
                "Frame": current_frame,
                "Visibility": 1,
                "X": center_x_pixel,
                "Y": center_y_pixel,
                "W": width_pixel,
                "H": height_pixel,
                "path": f"{image_dir}/{current_frame:06d}.jpg",
            }
        )

    dirname = "_".join(json_path.replace("\\", "/").split("/")[-3:])
    if boxes:
        df = pd.DataFrame(boxes)
        os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(CSV_OUTPUT_DIR, f"{dirname}.csv")
        df.to_csv(csv_path, index=False)
        print(f"生成 CSV 文件：{csv_path}")
    else:
        print(f"无有效数据生成 CSV：{dirname}")


def batch_mog2(input_dir, json_dir, s3_path, test_dir, mask_dir=""):
    """处理图像文件夹，生成 JSON 和测试图像"""
    make_dir(json_dir)
    make_dir(test_dir)
    image_files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    image_dir = input_dir
    MIN_AREA = 2
    bgs = left_bgs if "left" in input_dir.lower() else right_bgs
    dirname = "/".join(input_dir.replace("\\", "/").split("/")[-3:])
    for idx, file in tqdm(enumerate(image_files), total=len(image_files), desc=dirname):
        img_path = os.path.join(input_dir, file)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"跳过无效图片：{file}")
            continue

        h, w = frame.shape[:2]
        raw_frame = frame.copy()
        fg_mask = bgs.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.erode(fg_mask, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel2, iterations=1)
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        if mask_dir:
            os.makedirs(mask_dir, exist_ok=True)
            out_mask_path = os.path.join(mask_dir, file)
            cv2.imwrite(out_mask_path, fg_mask)
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        box_id = 1
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            offer = 5
            x, y, bw, bh = cv2.boundingRect(cnt)
            x -= offer / 2
            y -= offer / 2
            bw += offer
            bh += offer
            px = round(x / w * 100, 2)
            py = round(y / h * 100, 2)
            box = {
                "id": str(box_id),
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": w,
                "original_height": h,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": px,
                    "y": py,
                    "width": round(bw / w * 100, 2),
                    "height": round(bh / h * 100, 2),
                    "rectanglelabels": ["ball"],
                },
            }
            boxes.append(box)
            box_id += 1

        if len(boxes) != 1:
            continue

        filename = file
        prefixed_file = s3_path + "/" + filename
        # out_img_path = os.path.join(output_dir, filename)
        # shutil.copy(img_path, out_img_path)
        if test_dir:
            test_img_path = os.path.join(test_dir, filename)
            cv2.rectangle(
                raw_frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 1
            )
            cv2.imwrite(test_img_path, raw_frame)

        json_data = [
            {
                "data": {"image": f"s3://{S3_BUCKET}/{S3_PREFIX}/{prefixed_file}"},
                "annotations": [
                    {"model_version": "one", "score": 0.5, "result": boxes}
                ],
                "predictions": [
                    {"model_version": "one", "score": 0.5, "result": boxes}
                ],
            }
        ]
        json_name = file.rsplit(".", 1)[0] + ".json"
        json_path = os.path.join(json_dir, json_name)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2, ensure_ascii=False)

    print(f"处理完成：{dirname}")
    json_to_csv(image_dir, json_dir)  # 在处理完图像后立即生成 CSV


def iter_image_image(input_dir, mask_dir):
    """生成包含图片的文件夹路径（仅 left 和 right）"""
    for dirpath in find_folders_with_images(input_dir):
        if "_test" in dirpath or "_output" in dirpath or "_json" in dirpath:
            continue

        if not mask_dir:
            mask_dir = input_dir

        s3_path = f"{dirpath[len(mask_dir) + 1 :]}"
        test_dir = dirpath + "_test"
        # output_dir = dirpath + "_output"
        json_dir = dirpath + "_json"
        yield (dirpath, json_dir, s3_path, test_dir)


@click.command()
@click.option("--input_dir", default=DATA_DIR)
@click.option("--mask_dir", default="")
def main(input_dir, mask_dir):
    """主函数，批量处理文件夹"""
    batch_size = 4  # 设置并行进程数，建议根据 CPU 核心数调整
    tasks = [paths for paths in iter_image_image(input_dir, mask_dir)]
    if not tasks:
        print("未找到包含图片的 left 或 right 文件夹")
        return

    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for task in tasks:
            dirpath, json_dir, s3_path, test_dir = task
            futures.append(
                executor.submit(batch_mog2, dirpath, json_dir, s3_path, test_dir)
            )
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"处理失败: {e}")


if __name__ == "__main__":
    main()
