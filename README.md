# Dataset Preprocess

## 项目初始化

### 安装UV工具

安装文档 [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

```bash
pip install uv
```

### 初始化项目

uv自动下载依赖，并创建虚拟环境 **.venv**

```bash
uv sync
```

## 运行自动标注脚本

使用 openvcv mog2 模块，粗略标注数据样本

打开一号脚本 **src/1.frame_mog2.py**
修改输入输出目录

```python
# 输入数据
DATA_DIR = "/home/mao/jzd/tracknetv3/datasets/0928mv/20250928_075419"
# 输出 CSV 目录
CSV_OUTPUT_DIR = DATA_DIR
```

运行脚本

```bash
uv run python src/1.frame_mog2.py
```

## 针对轨迹范围下标，分割样本数据

针对完整视频帧分割成单条轨迹的数据样本

打开一号脚本 **src/2.frames_split.py**
修改输入输出目录和范围

```python
# 路径设置
base_dir = "/home/mao/Pictures/0928mv/20250928_075419"
output_dir = f"{base_dir}/output"
basename = os.path.basename(base_dir)

# 切割视频帧范围
range_start = 7921  # 起始帧
range_end = 8069  # 结束帧
```

运行脚本

```bash
uv run python src/2.frames_split.py
```
