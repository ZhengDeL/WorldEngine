# 使用固定 CUDA 张量验证 NPU MSDA

本工具通过选定的 WorldEngine 后端调用实际注册的 MX、MMCV MSDA 入口，将完整前向输出和三个输入梯度逐元素对比四次原版 CUDA 执行结果及单独保存的 CPU32 参考。工具不重新生成随机输入或参考结果。

金样目录不随仓库提供。需准备可信的完整 schema-1 CUDA 导出目录，以及从独立来源取得的清单 SHA256。当前验证器覆盖以下两个正常 L1 用例，每例包含四次 CUDA 执行：

| 用例 | Batch | Queries | 空间形状 | Heads / channels / points |
|---|---:|---:|---|---|
| 时序 | 2 | 40,000 | 200 × 200 | 8 / 32 / 4 |
| Query | 1 | 65,536 | 200 × 200 | 8 / 32 / 4 |

每个用例包含 `inputs.pt`、`original_CPU32_reference.pt`、`cuda_repeat_000.pt` 至 `cuda_repeat_003.pt`、`case_manifest.json`。根目录的 `manifest.json` 记录文件大小、SHA256、张量元数据及原始张量 SHA256。验证器执行前后均核对这些身份，并使用 Torch 的受限 `weights_only=True` 加载方式。便携导出还包含 `source_proof.json`，共 16 个文件：根清单、源码证明、两个用例清单及 12 个张量文件。验证器检查源码证明的完整 SHA256，并将其中的验证器、源码、精度、版本和用例定义与根清单逐项关联；每个张量载荷绑定同一证明摘要。不含 `prior_report_file` 的旧 schema-1 导出仍使用原来的 15 文件接口。

可先在 CUDA 主机使用[金样导出器](README_export_cuda_msda_goldens_zh.md)，再传输整个目录，并独立核对生成端清单摘要。本工具不下载金样张量。

请在具备 `torch_npu`、MMCV、MX Driving 的 WorldEngine Ascend 环境运行。设置 `$WORLDENGINE_ROOT`、`$GOLDENS`、`$GOLDEN_SHA256`，以及位于源码和金样目录外的全新 `$REPORT` 路径；报告父目录须已存在。设备编号为进程可见的逻辑编号。

```bash
python tools/validation/verify_npu_cuda_msda.py \
  --source "$WORLDENGINE_ROOT" \
  --goldens "$GOLDENS" --manifest "$GOLDEN_SHA256" \
  --output "$REPORT" --device npu:0 \
  --backend cuda-compatible --maximum-seconds 180
```

上述命令仅预览，不导入 Torch、不初始化设备、不写报告。确认参数后，在独立进程中增加 `--execute`，并由外层超时器约束运行：

```bash
timeout --signal=TERM --kill-after=10s 210s \
  python tools/validation/verify_npu_cuda_msda.py \
    --source "$WORLDENGINE_ROOT" \
    --goldens "$GOLDENS" --manifest "$GOLDEN_SHA256" \
    --output "$REPORT" --device npu:0 \
    --backend cuda-compatible --maximum-seconds 180 --execute
```

`--maximum-seconds` 是内部时间预算，不能保证中断正在执行的原生算子；Python 信号处理可能需要等待原生调用返回。请保留外层超时器或等效进程监管。超时终止的退出状态不代表完整数值测量。默认选择两个注册入口，共完成 80 项完整张量比较：CUDA 目标 64 项、CPU32 目标 16 项。`--entries mmcv` 仅选择六参数 MMCV 入口；`--backend reference` 使用同一固定目标测量默认历史后端。

原容差保持不变：前向 `atol=2e-5, rtol=2e-5`，梯度 `atol=1e-3, rtol=1e-4`。报告分别列出 CUDA、CPU32 的计数，以及原始张量字节一致性（SHA256，与序列化文件字节分别记录）和容差内一致两种结果；同时记录实际源码 SHA、后端策略、版本、FP32／禁用 autocast／关闭 HF32 设置及严格确定性开关读回。全局严格开关不等于所有外部算子均已注册确定性保证。

| 退出码 | 含义 |
|---:|---|
| 0 | 完整比较完成，CUDA、CPU32 所有项目均在原容差内 |
| 1 | 完整比较完成，至少一项超出原容差 |
| 2 | 输入无效、执行阻断或不完整、完整性校验失败 |

CPU32 与原版 CUDA 在部分插值边界存在差异。CUDA 项可以全部在容差内，而 CPU32 项仍使程序返回 1，应查看各目标的独立计数。单纯逐位不同不会触发退出码 1。结果仅覆盖所测算子用例，不构成整模型重放、收敛、质量、性能或其他 NPU 数值差异的根因证明。跨版本使用相同随机种子不代表输入相同，应以完整保存张量的 SHA 为准。

模型命令及检查点合同见[后端使用指南](../../ascend/MSDA_CUDA_zh.md)；已测范围和目标差异见[原版 CUDA 基线](../../ascend/CUDA_BASELINE_zh.md)。
