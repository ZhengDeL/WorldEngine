# SimEngine 昇腾 910C 适配评估

[English](simengine_ascend_assessment.md)

评估日期：2026-10-19。WorldEngine 基线为 `3eded7e8ac64e5bed817860765089b5914ab698f`，SimEngine 目录树为 `8eedf247a89ce5b022209e4cf0f3988cc2a10a2f`。本分支目前提供代码和依赖评估，尚未交付 SimEngine NPU 运行后端。

**原仓 SimEngine 不能直接在纯 NPU 环境完整运行。CANN 已有面向 Atlas A2/A3 的高斯渲染算子库，存在可验证的迁移路径，但这些算子与原仓 `gsplat==1.4.0` 的接口和渲染语义尚未证明兼容。** 本次没有执行 SimEngine NPU 渲染、闭环或 GPU/NPU 性能测试，速度差距、设备最低数量及完整可用性均未实测。

## 评估范围与证据

评估覆盖场景加载、车辆与交通行为、MTGS 图像渲染、AlgEngine 闭环客户端、NR/R 测试、带奖励 rollout、Ray 调度和合成数据导出。依据是固定基线的源码、原项目文档，以及下列项目的维护者源码。源码事实、候选能力和待验证判断在下文分别说明。

SimEngine 的默认渲染器是 MTGS。源码没有导入 CARLA 或启动 Unreal 服务；`CARLA` 仅出现在历史注释中，因此不能用 CARLA 的硬件要求判断本项目是否可迁移。这里的渲染是计算型高斯光栅化，核心依赖是 CUDA 扩展。[渲染器配置](../projects/SimEngine/worldengine/configs/common/renderer/mtgs.yaml#L1)、[渲染入口](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L14)。

AlgEngine 的既有 NPU 结果不等于 SimEngine 验收。原闭环脚本另行启动 `projects/AlgEngine/closed_loop/sim_test.py`，该入口仍使用 `model.cuda()` 和 `MMDataParallel`，必须单独适配和测试。[模型加载](../projects/AlgEngine/closed_loop/sim_test.py#L274)。

| 功能 | 固定基线的实现 | NPU 迁移判断与验收要求 |
| --- | --- | --- |
| 场景加载、车辆动力学、日志回放、IDM 与地图 | Python、NumPy、SciPy、nuPlan；[运行入口](../projects/SimEngine/worldengine/runner/run_simulation.py#L16)、[环境构建](../projects/SimEngine/worldengine/envs/base_env.py#L327) | 可保留 CPU 执行；仍需验证 ARM64 依赖、场景终止、R/NR 行为与确定性 |
| MTGS 静态背景、天空和动态刚体 | `Vanilla`、`Rigid`、`MirroredRigid` 高斯模型；[类型映射](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L31) | 张量变换可迁移；CUDA 球谐和光栅化必须替换，不能只改设备字符串 |
| 球谐颜色 | 三种模型直接调用 `gsplat.cuda._wrapper.spherical_harmonics`；[基础模型](../projects/SimEngine/worldengine/render/mtgs/gaussian_model/vanilla_gaussian_splatting.py#L123) | CANN 有同类算子；需核对维度、阶数、方向、截断及逐相机结果 |
| 多相机 RGB、可选深度 | [MTGS 实际调用](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L300) | 需要兼容层，保留抗锯齿、alpha、深度定义和原分辨率 |
| 几何与位姿 | 多处使用设备侧 FP64；[标定](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L128)、[位姿](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L395) | 必须实测目标算子与 dtype。可先保留 CPU FP64 几何、向 NPU 传 FP32 渲染输入，不能静默降精度后声称等价 |
| AlgEngine 闭环推理 | 独立进程读帧 PKL、写规划 NPY；[客户端](../projects/SimEngine/worldengine/components/agents/client/navformer_client.py#L81) | 需要 NPU 模型入口、精度/MSDA 设置、原子文件协议与超时；离线推理通过不能替代闭环测试 |
| 稠密奖励与候选轨迹评分 | 8192 轨迹词表；CUDA `TorchSimulator` 或 CPU `PDMSimulator`；[分支](../projects/SimEngine/worldengine/manager/dense_reward_manager.py#L275) | 原代码已有 CPU 路径，可作为初始方案；NPU 加速路径涉及 FP64/LQR，另行验证。保留奖励定义及候选顺序 |
| Ray 场景并行 | `torch.cuda.device_count()`、`num_gpus`、`ray.get_gpu_ids()`；[资源发现](../projects/SimEngine/worldengine/utils/multithreading/worker_ray.py#L49)、[场景分片](../projects/SimEngine/worldengine/runner/builders/env_builder.py#L23) | 需改为 NPU 资源、芯片可见性和一致的 worker ID；Ray 2.48 本身已有 NPU 管理器，不能直接将 `num_gpus` 当作 NPU 资源 |
| 测试、rollout、导出 | [测试脚本](../projects/SimEngine/scripts/run_ray_distributed_testing.sh#L1)、[rollout](../projects/SimEngine/scripts/run_ray_distributed_rollout.sh#L1)、[导出](../projects/SimEngine/scripts/export_simulation_data.py#L1) | 原脚本固定八路 CUDA，带清理与进程组终止；需要受控 NPU 启动器、退出码和产物检查。导出文件存在不代表奖励、图像及元数据均有效 |

关闭渲染不能作为完整迁移方案。环境明确要求启用 DataManager 时必须启用 RenderManager；仅运行 CPU 车辆状态，不会得到供视觉规划和训练使用的图像。[约束](../projects/SimEngine/worldengine/envs/base_env.py#L336)。

## 候选后端及版本条件

优先候选是 CANN 的 [gauss-splat](https://gitcode.com/cann/gauss-splat)。维护者 README 声明支持 Atlas A2/A3，提供球谐、投影、筛选、排序、渲染前后向等算子，要求 CANN 8.5 及以上、配套 PyTorch 2.7.1 及以上。README 中的 `torch_npu v7.3.0` 是插件发布线表述，不能直接与 Python wheel 的 `2.7.1.post8` 做版本大小比较。**CANN 9.1、现有 Torch-NPU wheel 与候选仓某个 tag 的编译及运行 ABI 尚待核对，未做兼容实测。**

较早的官方 [3DGS 实践](https://gitcode.com/cann/cann-recipes-spatial-intelligence/blob/master/algorithms/gaussian_splatting/README.md) 使用 `meta_gauss_render`，文档配置为 A2、CANN 8.2.RC1、PyTorch 2.1.0、Torch-NPU 2.1.0.post12。[SLARM 维护者](https://github.com/kevinchiu19/SLARM#installation) 另有 PyTorch 2.1/910C 使用记录。这些是技术可行性的依据，不是 WorldEngine 或当前软件栈的兼容证明。

[Ray 2.48 的固定版本源码](https://github.com/ray-project/ray/blob/ray-2.48.0/python/ray/_private/accelerators/npu.py) 已定义 `NPU` 资源和 `ASCEND_RT_VISIBLE_DEVICES`。基础 NPU 调度不要求先升级到新版 Ray 插件，但 WorldEngine 自己的资源发现、分片数和 worker 路径仍需适配。

原 WorldEngine 环境使用 Python 3.9、PyTorch 2.0.1/CUDA 11.8、gsplat 1.4.0，并把重建训练交给外部 MTGS。NPU 试验应使用独立环境和镜像，保留原 GPU 环境；不要在现有 AlgEngine 环境中直接替换依赖。[原环境说明](installation.md#environment-1-simengine-simengine)。切换到较新 PyTorch 后，还应检查场景 `.ckpt` 的序列化格式及 `torch.load` 默认行为，避免把反序列化问题误认为渲染算子问题。

以下是 2026-10-19 获取的 canonical CANN `master` 小文本快照。SHA256 标识本次读取的完整文件内容，不是 Git commit，也不代表其后 `master` 不再变化。构建时必须另选配套 tag/commit 并记录实际指纹。

| 源码 | 本次内容 SHA256 |
| --- | --- |
| [README.md](https://raw.gitcode.com/cann/gauss-splat/raw/master/README.md) | `8c640465d0c978b8d22b5b736cff4944226b037074c28e6d7778cef2a5f17483` |
| [ops/rendering.py](https://raw.gitcode.com/cann/gauss-splat/raw/master/gauss_splat/ops/rendering.py) | `bb4b4b2ea3115c28ce68fce8327dede71effad7c94aaae33b5fdd331bfd59294` |
| [ops/projection_three_dims_gaussian_fused.py](https://raw.gitcode.com/cann/gauss-splat/raw/master/gauss_splat/ops/projection_three_dims_gaussian_fused.py) | `a8020eb1bd1c7eebc68d6802f20b45e514ee71af7c0cdfa0806426434a84a281` |
| [ops/calc_render.py](https://raw.gitcode.com/cann/gauss-splat/raw/master/gauss_splat/ops/calc_render.py) | `55538ed671be7cefcee77ba3089956cb7a0a217ffa43e53cd3e12c5a55a431cc` |
| [build.sh](https://raw.gitcode.com/cann/gauss-splat/raw/master/build.sh) | `126616ed2406a29a8df77b3c9dc71a713a3554b9dec92e9c7d68b91f40d6c700` |

## 与原 gsplat 调用的接口差距

下表比较原 [MTGS 调用](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L338) 与上述 CANN 实现。候选的低层能力有助于实现兼容层，但高层 `Rasterizer` 不是可直接替换的 `gsplat.rendering.rasterization`。

| 参数或语义 | WorldEngine 原合同 | CANN 取样源码与所需工作 |
| --- | --- | --- |
| 高斯参数 | 世界坐标 means；已归一化 quats；`exp(scales)`；`sigmoid(opacities)` | 高层入口接收 log-scales/logit-opacities 并再次激活。直接传原值会重复激活；应明确转换或使用接收已激活值的低层投影 |
| 颜色 | 每相机 `C×N×3` 已计算 RGB，包含动态刚体/Fourier/镜像处理 | 高层要求 `sh0/shN` 系数，内部强制计算 SH。不能把 RGB 冒充 SH；其 `sh_degree=None` 验证分支也不构成可用 RGB 通路，需要实际桥接 |
| SH | 按相机与对象计算；degree 0 使用 sigmoid；正阶数为 SH 后 `+0.5` 并截断到 `[0,1]` | 候选有逐相机 SH 和 0–4 阶算子；高层 degree 0 仍走 SH，并只做下界截断。需保留原分支与上界截断 |
| 相机 | 已算好的 world-to-camera、`C×3×3` 内参 | 候选公开入口收 camera-to-world 再求逆；矩阵方向必须一致，避免二次反转 |
| 近远裁剪 | `near=0.01`、`far=1e10`、`radius_clip=0` | 候选高层写死 near=0.2；低层投影支持 near/far 参数，需要显式传入原值 |
| 抗锯齿 | 默认 `antialiased`，遵循 gsplat 的协方差/opacity 补偿 | 候选高层没有此参数，投影默认 `calc_compensations=False`；低层提供补偿选项，仍需逐输出对照 |
| tile / packed | tile 16、`packed=False` | 候选默认 tile 32、压缩投影结果。tile 16、排序及有效高斯映射需要测试，不能因可传参数就认定等价 |
| RGB、alpha 与背景 | 返回 RGB/可选深度通道及独立 alpha；原代码据此合成背景 | 候选高层返回 RGB 和深度，没有 alpha。低层额外返回累积状态，可研究导出 alpha，但必须核对其定义、裁剪和终止规则 |
| `RGB+ED` | 深度为归一化期望深度，空像素按原规则处理 | 候选低层文档定义的是加权深度和；高层虽接受 mode 字符串，却没有相应输出模式分支。不能把 D 直接当 ED，需要 alpha 归一化及空像素验证 |
| 动态场景 | 背景、天空、多动态 actor，含镜像与时间变化 SH | 先保留 MTGS 对象变换，再向后端传组合高斯和各相机颜色；单个静态点云通过不代表动态场景通过 |
| 梯度 | SimEngine `render()` 为 `@torch.no_grad()`；传入 `absgrad=True` | 渲染 pilot 可先验证前向；这不覆盖外部 MTGS 训练、反向或重建收敛 |
| 图像导出 | FP32 渲染后转 uint8、RGB→BGR、OpenCV 畸变重映射 | 保留原操作顺序和元数据，不用降分辨率或省相机来宣称同配置性能 |

原 gsplat 的 CUDA 构建与算子来源可直接查看其固定 [v1.4.0 backend](https://github.com/nerfstudio-project/gsplat/blob/v1.4.0/gsplat/cuda/_backend.py) 和 [rendering.py](https://github.com/nerfstudio-project/gsplat/blob/v1.4.0/gsplat/rendering.py)。

## 两条部署候选

**全 NPU 计算方案**：CPU 保留场景、地图、文件协议及必要的 FP64 几何，NPU 承担高斯渲染和规划模型。需要完成上述渲染合同、闭环客户端与调度适配。CPU 主机仍是运行组成部分；“全 NPU”指不依赖 NVIDIA GPU。目前未验证。

**GPU 渲染 + NPU 规划方案**：保留原 MTGS/gsplat CUDA 渲染，在独立 NPU 进程运行规划模型，通过原帧/轨迹文件接口连接。代码结构允许这种分工，但当前客户端、路径共享、同步、退出行为及性能都未测。GPU 仍是必要资源，不能称为纯 NPU SimEngine。跨主机时还需验证共享文件系统与传输时间。

## 单场景最小输入

先选择一个公开来源的真实场景，保留标识、时间戳、标定及所需 actor，不需要下载全部 WorldEngine 资产。

| 输入 | 最低验证范围 |
| --- | --- |
| 场景描述 | `all_scenarios.pkl` 中一个原始场景，保留完整 history/future、相机标定、地图及轨迹元数据；派生子集记录源文件 SHA 和所选 ID |
| MTGS 场景资产 | 与 `digitaltwin_asset_id` 对应的 `<asset-root>/<id>/background/<id>.ckpt`，其中包含该场景实际使用的背景、天空、刚体和镜像模型；[加载位置](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L471) |
| 地图 | 对应地区的 nuPlan 地图与地图版本 JSON；闭环指标/奖励需要实际地图数据 |
| 规划模型 | 与配置匹配的 VADv2 或 HydraMDP 官方 checkpoint；仅做渲染算子 pilot 不需要规划权重 |
| 轨迹词表 | 奖励/规划使用 `test_8192_kmeans.npy`，保留 8192 个候选及其顺序 |
| 输出空间 | 原分辨率各相机图像、逐帧元数据、轨迹、奖励及报告；容量按所选场景实测 |

2026-10-19 对固定 WorldEngine 数据版本 `d0e88d858533ca407a3387a26577c083aa7b3617` 的文件元数据查询显示：`navtest_failures/all_scenarios.pkl` 为 **911,923,264 B**；其资产发布为三包，分别为 **30,904,422,680 B、29,680,462,746 B、27,886,330,535 B**。这只是发布粒度，尚未下载正文、检查包内索引或确认一个场景位于哪一包，不能把任一包称为已验证 mini。[场景清单](https://huggingface.co/api/datasets/OpenDriveLab/WorldEngine/tree/d0e88d858533ca407a3387a26577c083aa7b3617/data/sim_engine/scenarios/original/navtest_failures?recursive=false&limit=1000)、[资产清单](https://huggingface.co/api/datasets/OpenDriveLab/WorldEngine/tree/d0e88d858533ca407a3387a26577c083aa7b3617/data/sim_engine/assets/navtest_failures/assets?recursive=false&limit=1000)。

AlgEngine 的 NAVTRAIN 相机包不能替代这些 Gaussian 场景 `.ckpt`。已有预构建场景和资产时，渲染 pilot 不需要重新执行整个 MTGS 重建流程；若要自己制作资产，还需外部 [MTGS](https://github.com/OpenDriveLab/MTGS) 的数据、重建训练与导出，属于额外验收范围。

## 可执行的研究预检

以下操作是待执行的研究流程，不是本分支已有的 SimEngine NPU 启动命令。设备操作应在指定独立环境内执行，保留每条命令的退出码和版本输出。

先记录基线与候选。`CANN_GAUSS_COMMIT` 必须是已核对 CANN/插件配套关系的实际提交，不能填 `master`：

```bash
git rev-parse HEAD
git status --short
npu-smi info
python -m pip freeze
python -m pip check

WORKDIR=/path/to/isolated-simengine-research
: "${CANN_GAUSS_COMMIT:?Select a compatible fixed gauss-splat commit}"
git clone https://gitcode.com/cann/gauss-splat.git "$WORKDIR/gauss-splat"
git -C "$WORKDIR/gauss-splat" checkout --detach "$CANN_GAUSS_COMMIT"
git -C "$WORKDIR/gauss-splat" rev-parse HEAD
```

在已有配套 PyTorch/Torch-NPU 的独立构建环境中加载 CANN，按维护者依赖准备编译环境。README 提供 `build.sh --python=<version>` 和 `pip install . --no-build-isolation` 两种方法；需有编译器、CMake、ACL/CANN 头文件及算子开发环境，不能仅安装上游 CUDA gsplat wheel。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$WORKDIR/gauss-splat"
python -m pip install numpy==1.23 decorator sympy scipy attrs cloudpickle \
  psutil synr==0.5.0 tornado cmake pyyaml expecttest protobuf
python -m pip install . --no-build-isolation
python -m pip check
cd "$WORKDIR"
python -c 'import acl, torch, torch_npu, gauss_splat.gsplat_c; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())'
```

构建/导入通过后，可用一枚空闲逻辑 NPU 执行下面的最小前向探针。它使用合成的 256 个高斯、单相机和 64×64 图像，不下载模型或场景。**该探针尚未实测；即使通过，也只证明候选后端的这一调用可用，不证明原 MTGS 兼容。**

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python - <<'PY'
import torch
import torch_npu
import gauss_splat

torch.npu.set_device(0)
device = 'npu:0'
n = 256
means = torch.zeros(n, 3, device=device)
means[:, 2] = 2.0
quats = torch.zeros(n, 4, device=device)
quats[:, 0] = 1.0
splats = dict(means=means, quats=quats,
              scales=torch.full((n, 3), -3.0, device=device),
              opacities=torch.zeros(n, device=device),
              sh0=torch.zeros(n, 1, 3, device=device),
              shN=torch.zeros(n, 15, 3, device=device))
pose = torch.eye(4, device=device)[None]
intrinsics = torch.tensor([[64., 0., 32.], [0., 64., 32.],
                           [0., 0., 1.]], device=device)[None]
with torch.no_grad():
    rgb, depth, info = gauss_splat.Rasterizer().rasterization(
        cam=(pose, intrinsics, 'RGB'), size=(64, 64), tile_size=32,
        active_sh_degree=0, splats=splats, camera_model='pinhole')
torch.npu.synchronize()
assert rgb.shape == (1, 64, 64, 3)
assert depth.shape == (1, 64, 64, 1)
assert torch.isfinite(rgb).all() and torch.isfinite(depth).all()
print({'rgb': list(rgb.shape), 'depth': list(depth.shape),
       'rgb_max': float(rgb.max()), 'meta_keys': sorted(info)})
PY
```

随后才实现独立渲染兼容层，使用同一份原始 gsplat 1.4.0 输入和完整输出对照 SH、投影、排序、RGB、alpha、ED。覆盖单/多相机、near/far 边界、遮挡、空像素、抗锯齿、镜像及动态 actor。完成真实单场景后，再接两种规划模型，验证 R/NR、奖励 rollout、恢复和导出产物。原脚本的 `simulation_completed.flag` 及进程退出码不能单独作为成功依据，应核对每个 `RunnerReport.succeeded`、实际帧数、有限值、轨迹和奖励覆盖。

## 设备与性能验收

算子 pilot 从 **1 个逻辑 NPU** 开始。首次闭环建议渲染与规划各用一个独立逻辑芯片，先隔离内存和执行问题；两进程共用一芯片是否可行、最低 HBM/CPU 内存、16 芯片扩展能力均需实测。原文档的 8 GB 最低/24 GB 推荐 GPU 与 32/64 GB 主存要求是 GPU 指南，不能直接换算为 NPU 容量。

应先通过输出质量门禁，再做三层同条件测量：

| 层次 | 必须固定的输入与配置 | 报告内容 |
| --- | --- | --- |
| 算子 | 同一高斯、相机、dtype、clip、排序与抗锯齿设置 | 预热后同步耗时、峰值 HBM、完整输出误差；编译/冷启动单列 |
| 八相机渲染 | 同一真实资产、动态 actor、时间戳、原分辨率与后处理 | 每个完整八相机帧的 p50/p95 延迟、帧/s、相机图像/s、内存及图像质量 |
| 完整闭环 | 同一模型 checkpoint、场景集合、R/NR、奖励与导出范围 | 总 wall time、完成/异常场景数、规划轨迹、闭环指标、奖励完整性，以及 IO/通信时间 |

保留原始日志、版本、源码和输入哈希。GPU 单卡、NPU 单芯片、NPU 双芯片物理卡及多进程配置分别标明；不能用 16 个 NPU 芯片与 8 张 GPU 的数量直接推出倍率。当前没有本框架的成对实测数据，因此没有可发布的快慢百分比或成本/能耗优势结论。

## 原仓已有的功能边界

- `DeformableSubModel` 在原 [类型映射](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L36) 中为 `None`；不能把任意外部 MTGS 模型都视为当前可加载资产。
- [渲染输出](../projects/SimEngine/worldengine/render/mtgs/mtgs.py#L380) 的 LiDAR 字段为空并标注 TODO。原数据目录中的 LiDAR 文件不代表运行时已实现 LiDAR 渲染。
- 原 [README 路线图](../README.md#roadmap) 的 BWM integration 尚未完成。未发布的 BWM 生成能力及外部 MTGS 重建训练，不应计入已实现 SimEngine 的迁移验收。

本分支应以单场景真实渲染和闭环证据逐步收敛适配范围；在对应门禁完成前，保持“研究/未验证”的状态。
