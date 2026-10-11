# HKquad — Heat-kernel fields for quadrilateral meshing

在**独立输入三角曲面**上逐模型优化可移动热核，构造 4-RoSy 方向场和尺寸场，再通过 MIQ 参数化与 QEx 提取四边形网格。参考 quad 只提供方向监督。


## 1. 创建环境

```bash
conda env create -f environment.yml
conda activate hktex
python -m pip check
```

环境文件包含 Python 3.11、官方 CUDA PyTorch 2.10.0+cu128、Triton 3.6.0、GPU FAISS 1.13.1、DiGeo 0.0.4 和 CPU 几何依赖。GPU FAISS 使用 pytorch / nvidia Conda 通道，不能用 `faiss-cpu` 代替；`rtree` 用于三角曲面与参考 quad 的最近点查询。CMake 限定为 3.x，避免旧第三方后端与 CMake 4 的兼容问题。

也可以分步安装，版本与环境文件一致：
```bash
conda create -n hktex python=3.11 numpy=2.2.6 pip 'cmake>=3.24,<4' ninja -c conda-forge
conda activate hktex
conda install --override-channels -c pytorch -c nvidia -c conda-forge \
  'faiss-gpu=1.13.1=py3.11_h1b31e9c_0_cuda12.6'
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'triton==3.6.0' 'digeo==0.0.4' -r requirements.txt
python -m pip check
```

## 2. 构建 Stage 2：MIQ 与 QEx

```bash
cmake -S cpp/miq_backend -B cpp/miq_backend/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/miq_backend/build -j 4
cmake -S cpp/qex_backend -B cpp/qex_backend/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/qex_backend/build -j 4
```

MIQ/QEx 为 CPU/C++ 后端，热核优化使用 GPU。若暂时只做 Stage 1，可以不构建后端并在训练命令中加 `--stage1-only`。

## 3. 下载 BiMDF 配对数据

使用作者发布的 [BiMDF quantization dataset / QuadWild-300](https://zenodo.org/records/7962513)。包中包含三角输入、中间结果和多种 quad 输出。本指南固定使用：

- **输入三角 mesh：** 每个模型的 `mesh_rem.obj`，是作者预处理后的上游三角曲面；不是由参考 quad 拆出来的输入。
- **参考 quad：** 同模型的 `flow/quad_smooth_flow.obj`。


```bash
mkdir -p datasets/downloads datasets/raw
curl --fail --location --retry 3 --continue-at - \
  'https://zenodo.org/records/7962513/files/bimdf_quantization_dataset.zip?download=1' \
  --output datasets/downloads/bimdf_quantization_dataset.zip
printf '%s  %s\n' 'd551991a1bdce803ddd2d38af899d805' \
  'datasets/downloads/bimdf_quantization_dataset.zip' | md5sum --check -
```

校验显示 `OK` 后执行 `unzip datasets/downloads/bimdf_quantization_dataset.zip -d datasets/raw`。

只继续处理校验成功且完整解压的数据。上述解压命令用于原始数据目录；再次解压可能覆盖或询问同名文件。

以 Dente 为例，解压后应有：

```text
datasets/raw/bimdf_quantization_dataset/quadwild-300/
└── Organic/dente/
    ├── mesh_rem.obj                 # 独立三角输入
    └── flow/quad_smooth_flow.obj     # 参考四边形
```

确认文件路径：

```bash
test -f datasets/raw/bimdf_quantization_dataset/quadwild-300/Organic/dente/mesh_rem.obj
test -f datasets/raw/bimdf_quantization_dataset/quadwild-300/Organic/dente/flow/quad_smooth_flow.obj
```

## 4. 完整流程：以 Dente 为例

```bash
python scripts/prepare_quad_supervision.py \
  --triangle datasets/raw/bimdf_quantization_dataset/quadwild-300/Organic/dente/mesh_rem.obj \
  --quad datasets/raw/bimdf_quantization_dataset/quadwild-300/Organic/dente/flow/quad_smooth_flow.obj \
  --output-dir datasets/bimdf_cross_mesh/dente \
  --source-url https://zenodo.org/records/7962513
```

输出：

```text
datasets/bimdf_cross_mesh/dente/
├── triangle.obj       # 输入三角文件的原样复制，顶点/面不重排
├── quad_targets.npz   # v2逐面参考边向量、有效性及对应信息
└── provenance.json    # 来源、文件哈希、映射参数与质量统计
```

`--triangle` 和 `--quad` 都必须提供。

两份网格须表达同一物体，已经位于共同坐标系。默认距离门槛为输入包围盒对角线的 1%，有向法向夹角门槛为 60°，至少 95% 的输入面积需要可靠方向标签。不能匹配的模型会报错，不自动移动输入几何或补标签。

## 批量处理
将对应同一曲面、使用一致坐标系的三角 mesh 与参考 quad 放在 datasets/raw 下：
```bash
python scripts/prepare_quad_dataset.py \
  --source-dir datasets/raw/bimdf_quantization_dataset/quadwild-300 \
  --pattern '**/flow/quad_smooth_flow.obj' \
  --triangle-relative ../mesh_rem.obj \
  --output-dir datasets/bimdf_cross_mesh_batch \
  --source-url https://zenodo.org/records/7962513
```




## 5. 运行 Stage 1 + Stage 2


```bash
python train.py --data-dir datasets/bimdf_cross_mesh/dente --shape dente --gpu 0
```

默认配置是 `configs/hktex2quad_cross_mesh_gpu.yaml`：128个 kernel、1500步、目标密度约3000个quad。位置、十字方向、幅值、支撑各向异性及支撑朝向继续优化；h、s、epsilon固定。默认h按表面积和目标quad数初始化为统一尺寸，s/epsilon保留每个kernel自身初始化值，完整参数保留。

常用参数：

```bash
python train.py --data-dir datasets/bimdf_cross_mesh/dente --shape dente --gpu 0 \
  --kernels 256 --iters 1500 --target-quads 3000
```

`--target-quads` 控制密度，实际面数可能不同。`--stage1-only` 只优化并导出场。Fandisk可用 `--config configs/hktex2quad_fandisk_512_gpu.yaml`，仍须先准备它的独立三角配对数据。


## 6. 查看输出与失败原因

输出结构为：

```text
outputs/hktex2quad-quad/dente_quad_时间戳_随机后缀/
├── configs/                  # 本次配置
├── ckpts/                    # kernel模型与参数
├── dense_face_field.pt       # 三角曲面上的方向/尺寸/有效性
├── visualizations/           # 场与kernel图
├── quad_pipeline/            # MIQ/QEx的UV、网格、质量报告
├── FINAL_QUAD.obj            # 用MeshLab打开的最终四边形
├── INPUT_TRIANGLE.obj        # 同坐标系输入三角网格
├── run_status.json           # 各阶段状态与命令
├── stage1_console.log
├── stage2_console.log
└── README_OUTPUT_FILES.md
```


