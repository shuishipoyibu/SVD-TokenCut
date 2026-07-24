# main_tokencut.py 命令与参数手册

本文档整理当前 `main_tokencut.py` 的运行命令、命令行参数、参数约束和输出路径。所有命令均应从**仓库根目录**运行。

## 1. 路径约定与可配置目录

本项目的 `main_tokencut.py` 使用相对于仓库根目录的固定数据集位置；请按照 [DOWNLOAD_DATA.md](DOWNLOAD_DATA.md) 准备数据：

```text
VOC07:  datasets/VOC2007/VOCdevkit/VOC2007
VOC12:  datasets/VOC2012/VOCdevkit/VOC2012
COCO:   datasets/COCO/images/train2014 或 datasets/COCO/images/val2014
        datasets/COCO/annotations/
```

`--save-feat-dir` 和 `--load-feat-dir` **没有代码默认值，也不要求使用某个机器的绝对路径**。它们由使用者自行指定：

- `--save-feat-dir <目录>`：将当前运行生成的每张图像特征保存为 `.npy`；目录不存在时程序会自动创建。
- `--load-feat-dir <目录>`：从该目录读取已有 `.npy` 特征，并跳过 DINO 模型加载与特征提取；该目录必须与此前保存时使用的模型、补丁大小、特征类型和图像集合一致。
- 本文示例使用 `feature_cache/` 作为仓库内缓存位置，例如 `feature_cache/VOC12_KEY` 与 `feature_cache/COCO2014_VALKEY_10k`。这只是推荐的相对路径，可替换为任何可读写目录，例如外接磁盘或服务器高速盘。

## 2. 运行命令
### 2.1 VOC2012 基准实验

直接提取 DINO 特征并运行 TokenCut：

```bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset VOC12 \
  --set trainval \
  --which_features k \
  --run-name voc12_baseline_01
```

第一次运行可能需要下载模型权重。输出位于：

```text
outputs/voc12_baseline_01/VOC12_trainval/TokenCut-vit_small16_k/
```
### 2.2 提取并保存特征

VOC2012 全量 `trainval` key 特征：

```bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset VOC12 \
  --set trainval \
  --which_features k \
  --save-feat-dir feature_cache/VOC12_KEY \
  --no_evaluation \
  --run-name voc12_feature_export_01
```

COCO 2014 `val2014` 前 10000 张 key 特征，适合先试跑：

```bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset COCO20k \
  --set val \
  --which_features k \
  --save-feat-dir feature_cache/COCO2014_VALKEY_10k \
  --no_evaluation \
  --run-name coco2014_val_key_export_100 \
  --max-images 10000
```

如需 query 或 value 特征，将 `--which_features k` 改成 `q` 或 `v`，并相应调整保存目录名。

每张图像会产生一个 `.npy` 文件。当前实现仍可能额外生成一个空的 `preds.pkl`；真正需要的特征位于 `--save-feat-dir`。

### 2.3 从缓存特征运行，无 PCA
VOC12:
```bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset VOC12 \
  --set trainval \
  --which_features k \
  --load-feat-dir feature_cache/VOC12_KEY \
  --run-name voc12_baseline_cached_01
```
使用前 10000 张缓存特征跑 COCO `val2014` TokenCut 评估：
```bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset COCO20k \
  --set val \
  --which_features k \
  --load-feat-dir feature_cache/COCO2014_VALKEY_10k \
  --run-name coco2014_val_cached_100 \
  --max-images 10000
```
缓存目录中的文件必须按图像 ID 命名，例如：

```text
VOC12_KEY/2008_000002.npy
VOC12_KEY/2008_000003.npy
```

输出结构为：

```text
outputs/voc12_key_pca_sweep_01/VOC12_trainval/
├── TokenCut-vit_small16_k_pca8/
├── TokenCut-vit_small16_k_pca16/
├── TokenCut-vit_small16_k_pca32/
└── TokenCut-vit_small16_k_pca64/
```


### 2.8 不同模型和补丁尺寸

```bash
# ViT-S/8
python main_tokencut.py \
  --dataset VOC12 \
  --set trainval \
  --arch vit_small \
  --patch_size 8 \
  --run-name voc12_vits8_01

# ViT-B/16
python main_tokencut.py \
  --dataset VOC12 \
  --set trainval \
  --arch vit_base \
  --patch_size 16 \
  --run-name voc12_vitb16_01
```

### 9.2 降维方法对比命令


**中心化 PCA 批量实验：**

~~~bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset VOC12 \
  --set trainval \
  --which_features k \
  --load-feat-dir feature_cache/VOC12_KEY \
  --pca-dims 1 8 16 32 64 128 \
  --run-name voc12_centered_pca_sweep_01
~~~

COCO 对照命令：

~~~bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset COCO20k \
  --set val \
  --which_features k \
  --load-feat-dir feature_cache/COCO2014_VALKEY_10k \
  --pca-dims 1 8 16 32 64 128 \
  --run-name coco2014_val_centered_pca_sweep_01 \
  --max-images 10000
~~~


**仅中心化基线：**

~~~bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset VOC12 \
  --set trainval \
  --which_features k \
  --load-feat-dir feature_cache/VOC12_KEY \
  --center-only \
  --run-name voc12_center_only_01
~~~

COCO 对照命令：

~~~bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset COCO20k \
  --set val \
  --which_features k \
  --load-feat-dir feature_cache/COCO2014_VALKEY_10k \
  --center-only \
  --run-name coco2014_val_center_only_01 \
  --max-images 10000

~~~
**未中心化 PCA（仅SVD） 实验：**
~~~bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset VOC12 \
  --set trainval \
  --which_features k \
  --load-feat-dir feature_cache/VOC12_KEY \
  --svd-dims 1 8 16 32 64 128 \
  --run-name voc12_uncentered_svd_sweep_01
~~~

如果只导出了前 10000 张 COCO 特征，使用：

~~~bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset COCO20k \
  --set val \
  --which_features k \
  --load-feat-dir feature_cache/COCO2014_VALKEY_10k \
  --svd-dims 1 8 16 32 64 128 \
  --run-name coco2014_val_uncentered_svd_sweep_100 \
  --max-images 10000
~~~

**批量运行 Cosine-SVD 维度**
VOC12
~~~bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset VOC12 \
  --set trainval \
  --which_features k \
  --load-feat-dir feature_cache/VOC12_KEY \
  --cosine-svd-dims 1 8 12 16 20 24 28 32 \
  --run-name voc12_cosine_svd_sweep_01
~~~

前 10000 张 COCO 特征，使用：

~~~bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset COCO20k \
  --set val \
  --which_features k \
  --load-feat-dir feature_cache/COCO2014_VALKEY_10k \
  --cosine-svd-dims 64 128 \
  --run-name coco2014_val_cosine_svd_sweep_100 \
  --max-images 10000
~~~

**批量运行随机投影维度**

~~~bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset VOC12 \
  --set trainval \
  --which_features k \
  --load-feat-dir feature_cache/VOC12_KEY \
  --random-projection-dims 1 8 16 32 64 128 \
  --random-seed 4 \
  --run-name voc12_random_projection_sweep_seed4
~~~
前 10000 张 COCO 特征，使用：

~~~bash
OMP_NUM_THREADS=4 python main_tokencut.py \
  --dataset COCO20k \
  --set val \
  --which_features k \
  --load-feat-dir feature_cache/COCO2014_VALKEY_10k \
  --random-projection-dims 1 8 16 32 64 128 \
  --random-seed 4 \
  --run-name coco2014_val_random_projection_sweep_seed4_100 \
  --max-images 10000
~~~
## 3. 全部参数

| 参数 | 类型/取值 | 默认值 | 当前作用 |
| --- | --- | --- | --- |
| `-h`, `--help` | 开关 | 关闭 | 显示命令帮助并退出。 |
| `--arch` | `vit_tiny`, `vit_small`, `vit_base`, `moco_vit_small`, `moco_vit_base`, `mae_vit_base` | `vit_small` | 选择主干模型。 |
| `--patch_size` | 正整数 | `16` | 补丁大小，并用于计算特征网格；常用值为 8 或 16。 |
| `--dataset` | `VOC07`, `VOC12`, `COCO20k` | `VOC07` | 选择数据集。帮助信息中的 `None` 由单图模式内部使用，不应手工传入。 |
| `--set` | `train`, `val`, `trainval`, `test` | `train` | 数据集划分。VOC2012 不支持 `test`；COCO20k 的 `train` 对应 train2014 20k 子集，`val` 对应 COCO 2014 val2014。 |
| `--image_path` | 文件路径 | 无 | 切换为单图模式，并自动关闭评估和预测字典保存。 |
| `--save-feat-dir` | 目录路径 | 无 | 将每张图像的特征保存为 `.npy`，随后跳过 TokenCut。 |
| `--load-feat-dir` | 目录路径 | 无 | 从 `.npy` 缓存加载特征，跳过模型建立和特征提取。 |
| `--use-pca` | 开关 | 关闭 | 对每张图像的补丁特征单独执行 PCA。 |
| `--pca-dim` | 正整数 | 无 | 单次 PCA 的目标维度，必须与 `--use-pca` 同时使用。 |
| `--pca-dims` | 一个或多个互不重复的正整数 | 无 | 顺序执行一组 PCA 实验，例如 `--pca-dims 8 16 32 64`。 |
| `--output_dir` | 目录路径 | `outputs` | 输出根目录。数据集名、批次名和实验名会继续追加在其后。 |
| `--run-name` | 单层目录名 | 无 | 隔离一次运行；不能包含 `/`，也不能是 `.` 或 `..`。 |
| `--no_hard` | 开关 | 关闭 | VOC 评估时排除困难图像，并在数据集目录名后添加 `-nohards`。 |
| `--no_evaluation` | 开关 | 关闭 | 跳过真实标注和 CorLoc 评估；数据集可视化需要此参数。 |
| `--save_predictions` | 布尔值 | `True` | 控制 `preds.pkl`。当前使用 `type=bool`，传入字符串 `False` 仍会被解析为真，暂不建议在命令行使用它关闭保存。 |
| `--visualize` | `pred`, `attn`, `all` | 无 | 保存预测框、特征图或两者；当前仅在无评估模式实际保存。 |
| `--which_features` | `k`, `q`, `v` | `k` | 选择最后一个注意力层的键、查询或值 特征。 |
| `--k_patches` | 整数 | `100` | 当前只完成参数解析，没有传给 `ncut`，因此修改它不会改变结果。 |
| `--resize` | 整数 | 无 | 当前只传给单图 `ImageDataset`；数据集模式不会使用该参数。 |
| `--max-images` | 正整数 | 无 | 只处理数据集顺序中的前 N 张图。用于小规模特征导出时，后续加载该缓存也应使用相同 N。 |
| `--tau` | 浮点数 | `0.2` | TokenCut 图构建的阈值。 |
| `--eps` | 浮点数 | `1e-5` | TokenCut 图构建的数值阈值。 |
| `--no-binary-graph` | 开关 | 关闭 | 禁用默认二值图，改用相似度权重。 |
| `--dinoseg` | 开关 | 关闭 | 请求 DINO-seg 基线；见下方已知限制。 |
| `--dinoseg_head` | 整数 | `4` | DINO-seg 注意力头编号。 |

## 4. 参数组合规则

以下组合会被程序拒绝：

- `--max-images` 小于或等于 0。
- `--save-feat-dir` 与 `--load-feat-dir` 同时使用。
- `--load-feat-dir` 与 `--dinoseg` 同时使用。
- `--use-pca` 缺少 `--pca-dim`。
- `--pca-dim` 缺少 `--use-pca`。
- PCA 维度小于或等于 0。
- `--use-pca` 与 `--dinoseg` 同时使用。
- `--pca-dims` 与 `--use-pca` 或 `--pca-dim` 同时使用。
- `--pca-dims` 与 `--dinoseg` 或 `--save-feat-dir` 同时使用。
- `--pca-dims` 中包含非正数或重复维度。

PCA 缓存特征必须满足：

```text
shape = [1, tokens, channels]
tokens = 1 个 CLS token + 若干补丁 token
```

对每张图像，目标维度不能超过：

```text
min(补丁 token 数量, 通道数量)
```

这里执行的是逐图 PCA，不是对整个数据集拟合一个全局 PCA。

## 5. 输出目录与覆盖规则

数据集模式的通用输出结构是：

```text
<output_dir>/<run-name>/<dataset_set>/<experiment>/
```

未提供 `--run-name` 时会省略该层。常见实验名：

```text
TokenCut-vit_small16_k
TokenCut-vit_small16_k_pca32
vit_small-16_dinoseg-head4
```

每次分割运行完成后都会保存：

```text
preds.pkl    # 每张图像的预测框（启用预测保存时）
results.txt  # 运行时间；评估模式下还包含 CorLoc
```

`results.txt` 中的时间字段为：

```text
segmentation_time_seconds              # 所有图像的分割流程总秒数
timed_images                            # 实际计时的图像数
segmentation_time_seconds_per_image    # 平均每张图像的秒数
```

计时从每张图像的 DINO 特征已经可用后开始，到分割函数返回为止。直接 TokenCut 计入 `ncut(...)` 的完整耗时；PCA、单中心化、SVD、Cosine-SVD 和随机投影实验还会计入对应的特征变换耗时。缓存 `.npy` 的读取、DINO 特征提取、IoU/CorLoc 评估、可视化以及文件保存均不计入。CUDA 模式会在计时边界同步设备，避免异步执行导致时间偏小。

同一输出路径再次运行时，`preds.pkl` 和 `results.txt` 会被覆盖。建议每组正式实验都使用新的 `--run-name`。批量 PCA 未指定 `--run-name` 时会自动生成唯一时间戳目录。

## 6. 已知限制

1. 当前 `--dinoseg` 分支调用 `dino_seg(...)`，但 `main_tokencut.py` 中没有定义或导入该函数，实际运行会触发 `NameError`。修复前不要将该命令用于正式实验。
2. `--k_patches` 当前未参与 `ncut(...)` 调用，因此只是占位参数。
3. `--save_predictions` 使用 Python `bool` 转换字符串，`--save_predictions False` 不能可靠关闭保存。
4. 数据集评估模式中的可视化调用目前被注释；若要真正输出图片，需要添加 `--no_evaluation`。
5. `--save-feat-dir` 路径会跳过 TokenCut，但程序结尾仍可能保存空预测字典。特征导出是否成功应以 `.npy` 文件为准。
6. 相对数据集路径依赖当前工作目录。请从仓库根目录运行，或者后续将数据集路径改为基于脚本位置的绝对路径。

## 7. 常见错误

### `bash: --dataset: command not found`

上一行缺少续行符 `\`。确保命令写成：

```bash
python main_tokencut.py \
  --dataset VOC12 \
  --set trainval
```

### `数据集不存在或已损坏`

确认目录层级完整，而不只是外层目录存在：

```text
datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages
datasets/VOC2012/VOCdevkit/VOC2012/Annotations
datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt
```

### `找不到缓存特征`

程序会把图像名的扩展名替换为 `.npy`，然后直接在 `--load-feat-dir` 下查找。缓存文件不能再额外嵌套一层目录。

如果导出缓存时使用了 `--max-images N`，加载缓存运行 TokenCut、PCA、SVD 或 Cosine-SVD 时也要使用相同的 `--max-images N`。否则程序会继续遍历完整数据集，并在第一张没有缓存的图片处报 `Cached feature not found`。

### PCA 维度超过上限

减小 `--pca-dim`，或从批量维度列表中移除过大的维度。VOC 的图像尺寸不同，因此上限按图像分别计算；一张小图不支持某个维度时，整个该维度实验会停止。

### 缓存特征与网格大小不匹配

确保加载缓存时使用与特征生成时相同的 `--patch_size`、模型和特征类型。尤其不能把补丁尺寸为 8 的缓存按默认补丁尺寸 16 加载。

## 8. 查看实时帮助

代码参数发生变化后，以当前程序帮助为最终依据：

```bash
python main_tokencut.py --help
```


