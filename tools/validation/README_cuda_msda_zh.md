# 原 CUDA MSDA 验证

固定原CUDA输入和完整参照的导出、注册入口NPU对照，见[金样导出器](README_export_cuda_msda_goldens_zh.md)及[NPU验证器](README_npu_cuda_msda_zh.md)。

[English](README_cuda_msda.md) · [CUDA 实测基线](../../ascend/CUDA_BASELINE_zh.md)

该独立 Linux 工具比较原 `mmcv-full==1.6.2` CUDA MSDA、未改的 CPU FP32
参考实现及 CUDA 自身重复结果，不修改模型或已安装算子。[实际结果](../../ascend/results/cuda-msda-original-20261017.json)
完整保留 CPU32 超差项，执行完成不等于数值通过。

请使用已有的 CUDA 环境、兼容的 CUDA 版 MMCV 1.6.2、PyTorch 和 NumPy。
本次实测为 H20 上 PyTorch 2.0.1+cu118／CUDA 11.8／NumPy 1.23.4。工具不安装依赖，
硬核原 MMCV Python 参考及前后向源码哈希，记录 native DSO 完整 SHA 和版本。
允许不同构建的 DSO，但不能继承本次实测的来源结论。不同 Torch 版本即使 seed
相同，也不能假定输入字节相同；须比较六个输入及四个 CPU 参考 Tensor 哈希。

从仓库根目录运行。输出父目录须存在，报告文件须尚不存在。默认仅预览，不导入
框架、不创建文件：

```bash
mkdir -p /tmp/msda-result
python tools/validation/validate_original_cuda_msda.py \
  --output /tmp/msda-result/report.json
```

选择空闲 GPU 显式执行。CUDA C++ 调用不返回时，仍需外层只管理自身作业的超时：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=20s 360s \
  python tools/validation/validate_original_cuda_msda.py \
  --output /tmp/msda-result/report.json --device cuda:0 \
  --maximum-seconds 300 --execute
```

默认运行两个正常 L1 完整大形状，每个 case 在相同存储上执行四次，`seed=29`、四个
CPU 线程、严格确定性开关开启、TF32 关闭。可用 `--case temporal40000` 或
`--case query65536` 选一个 case，`--repetitions` 支持 4–8 次。完整比较前向及
value、location、attention-weight 三梯度的所有元素。全部重复结束后才将保留的
GPU 结果转到 CPU；不导出 Tensor 载荷，也不向执行主机外传输数据。

原前向阈值保持 `atol=2e-5, rtol=2e-5`，原梯度阈值保持 `atol=1e-3, rtol=1e-4`。
重复结果的字节逐位一致性单独记录，不等同于容差判定。全局严格开关不能证明
外部 CUDA 扩展实现了确定性算法。两个 case 不能代表历史 raw-L4／spatial-L1
特殊布局的 GPU 实测。

退出码 0表示完整比较且 CPU32／自重复两类原容差均通过；1 表示完整比较但至少
一类超差；2 表示阻断或未完成。外层 timeout 可能返回自身退出码。退出 1 仍须
保留实际 JSON，不能当作没有结果。该工具不验收模型训练、收敛、NPU 根因或
跨平台一致性。原本机报告保留实际运行路径，公版证据仅脱敏安装路径和库搜索路径。
