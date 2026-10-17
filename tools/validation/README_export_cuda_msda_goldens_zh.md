# 导出原版 CUDA MSDA 金样

本可选工具生成 [NPU 验证器](README_npu_cuda_msda_zh.md)所需的 schema-1 文件。它加载公开的 `validate_original_cuda_msda.py`，固定 SHA256 为 `af7309e616f9a02648f9a5b35f8960ea97f51e108b1915bca7843448bd733fea`；输入生成器及原 MMCV CPU32／CUDA 公式保持不变。环境需提供 `mmcv-full==1.6.2` 的真实 CUDA 扩展、NumPy 和兼容的 PyTorch CUDA。

两个正常 L1 完整形状及同一组输入存储上的四次执行均固定。导出器记录实际 Torch／CUDA 版本和完整张量 SHA。同一随机种子不保证跨 Torch 版本的张量值相同。CPU32 容差比较与重复执行逐位一致性分别记录，CPU32 超差不会阻断完整金样导出。

```bash
python tools/validation/export_original_cuda_msda_goldens.py \
  --validator tools/validation/validate_original_cuda_msda.py \
  --output "$NEW_GOLDEN_DIRECTORY" --device cuda:0 \
  --maximum-seconds 270
```

上述命令仅预览，不导入 Torch、不创建输出。确认参数后，在独立进程中实际导出，并由外层超时器约束运行：

```bash
timeout --signal=TERM --kill-after=10s 300s \
  python tools/validation/export_original_cuda_msda_goldens.py \
    --validator tools/validation/validate_original_cuda_msda.py \
    --output "$NEW_GOLDEN_DIRECTORY" --device cuda:0 \
    --maximum-seconds 270 --execute
```

`--maximum-seconds` 检查内部时间预算，不能保证中断正在执行的 CUDA 或序列化调用。请保留外层超时器或等效进程监管。超时或清单不完整时，不应使用该导出。输出目录须全新，父目录须已存在且路径规范；所选设备当前至少空闲 4 GiB 显存。文件约占 2 GB，另保留程序强制的 8 GiB 磁盘余量。完整张量留在运行主机，执行前用 `CUDA_VISIBLE_DEVICES` 限定目标设备。

完成后，按实际授权的传输流程将整个金样目录复制到 NPU 主机。在生成端计算 `manifest.json` 的 SHA256，并在传输后独立核对；将该摘要传给 NPU 验证器的 `--manifest`。清单绑定全部 12 个完整张量文件及两个用例清单，包括六输入元组和四个输出角色。

额外的不可变 `source_proof.json` 记录执行前实际原源码和精度。为兼容 schema 1，载荷字段 `original_prior_report_SHA256` 绑定该执行前源码报告，明确不代表先前的数值测量。工具不依赖任何私有旧报告，请将源码证明与金样一并保留。完整便携导出共 16 个文件：根清单、源码证明、两个用例清单和 12 个张量文件。NPU 验证器除检查完整文件与张量身份外，还核对源码证明的摘要及源码／精度／用例身份。

导出器退出码 0 表示完整且绑定源码的文件生成成功，不表示 CPU32 所有容差项通过；退出码 2 表示阻断或不完整，不应使用对应清单。NPU 验证器会独立保留 CUDA、CPU32 的数值结果，完整比较存在超差时返回 1。

NPU 后端选择见[后端使用指南](../../ascend/MSDA_CUDA_zh.md)，算子及训练的已测边界见[原版 CUDA 基线](../../ascend/CUDA_BASELINE_zh.md)。导出这两个正常 L1 用例不代表覆盖所有形状或整模型。
