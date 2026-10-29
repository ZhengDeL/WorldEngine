# 完整 NAVTEST 输入检查

`ascend/tools/verify_navtest_inputs.py` 检查官方 12,146 个 NAVTEST anchor、136 个 log、原始当前帧与前三帧、8 路相机，以及全部 142,976 张必需的 1920×1080 RGB JPEG。工具复用原 `ascend.preflight.inspect_phase`，再独立重建 17,872 个必需队列帧。这是 CPU 输入验收，不执行模型、训练、收敛、GPU 对照或官方轨迹评分。

2026 年 10 月 20 日，公开 CLI 已在固定镜像中完成全部 142,976 张必需图像解码和官方 32 包字节关联检查，错误为 0、退出码为 0。随后独立账本审计重建并核对了全部图像／成员记录及有序摘要，没有再次解码图像。源码、原始 summary 和审计报告身份见[实测记录](navtest_public_input_results_zh.md)。此前私有 worker `aad3c0875a015dd4c51204dcf2534a080f03ef6c4bdb51534189c44c8ec5ca79` 的结果仍单独保留。

## 输入与固定来源

当前优化分支要求运行源码指纹 `7a8c50dce5f58b4a568f1dd7d6286ddb3370d6723c71ae3c2550045fc72555fb`，共 247 个 Python 文件。嵌套工具单独记录 SHA，不计入这个范围。10 月 20 日的原始记录使用归档源码 `0488bfb7…`／245 文件，保留原身份；复跑旧记录须取回匹配的工具版本。当前源码绑定已通过 32 项 CPU 测试（原 17 项、4 项源码／数量合同测试、11 项归档测试），本次绑定修复没有重新解码全量图像。两模型原 100% 配置的 test 数据集必须同为完整 `NavSimOpenSceneE2E`，queue4、history3、load_interval1。

数据目录内必须有下列发布者原文件，工具检查完整文件大小与 SHA：

| 相对路径 | 字节数 | SHA256 |
| --- | ---: | --- |
| `alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl` | 848516627 | `59a8bdf1a9a2c6066dd562333679f54ffee9cbfe9cb3420bab27dce62c5da21b` |
| `alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl` | 2791314758 | `58d82e73bc3dfaa6464ea7d90dee86f8e0139ea1679145d610a9babf86fe7316` |
| `alg_engine/test_8192_kmeans.npy` | 3932288 | `cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35` |

图像位于 `raw/openscene-v1.1/sensor_blobs/test/<log>/<camera>/<file>.jpg`。CPU 检查不需要模型 checkpoint，也不需要待生成的官方 metric cache。标注/PDM pickle 仅在完整发布者 SHA 核对后加载，不接受随意下载的 pickle。

正式像素来源验证还需要原始 test-camera 的全部 32 个压缩包，共 127,882,665,618 字节。公开清单为 `ascend/data_tools/navtest12146/ARCHIVES.lock.json`，整文件 SHA 为 `c62488d01a9fe43c254085aa009a06f1e001e5011f4081a56069d209b7e6ed54`。清单只包含发布者版本、文件名、固定版本 URL、大小和 SHA，无服务器路径或代理信息。

SHA 来源为官方 Hugging Face LFS 元数据，版本 `a76f840b65e972bc45e56c2adced897498e9a026`。魔搭链接使用逐文件匹配的固定版本 `9439af9e240a950dd0b6862ce3bc34233c5d8690`。原魔搭元数据请求参数是 `master`；固定版本来自返回的逐文件记录，不能误写成该 API 请求本身已经固定版本。

## 仅 CPU 运行

使用含 NumPy、Pillow、PyYAML 和原 MMCV Config 依赖的 Linux 容器，不挂设备、不请求 Ascend runtime。下面的固定镜像具备原 CPU 检查依赖。启动前关闭 Torch 后端自动加载；MMCV 可能导入 Torch，但本工具不初始化加速器。开始和结束都会拒绝设备节点或已初始化的 CUDA/NPU 后端。

先设置 `SOURCE`、`DATA`、`ARCHIVES`、`QC_PARENT`、`JOB_NAME` 及获准使用的未来 UTC 时间 `DEADLINE_UTC`。输出父目录应存在，`run1` 子目录和容器名称必须全新。由外部监督器保留实际镜像/CID、挂载、退出/OOM 状态，并执行本任务截止时间。

```bash
docker run --name "$JOB_NAME" --runtime runc --network none \
  --memory 64g --cpus 8 --pids-limit 1024 \
  -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e CUDA_VISIBLE_DEVICES= -e ASCEND_RT_VISIBLE_DEVICES= \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -v "$SOURCE:/source:ro" -v "$DATA:/data:ro" \
  -v "$ARCHIVES:/archives:ro" -v "$QC_PARENT:/qc:rw" \
  --entrypoint /opt/conda/envs/bevformer/bin/python \
  sha256:3522f3ac97044486686539405709fb7f833eac3c95767b93db7fff1956e54d48 \
  /source/ascend/tools/verify_navtest_inputs.py \
  --source /source --data-root /data --output-dir /qc/run1 \
  --archive-root /archives \
  --archive-lock /source/ascend/data_tools/navtest12146/ARCHIVES.lock.json \
  --archive-lock-sha256 c62488d01a9fe43c254085aa009a06f1e001e5011f4081a56069d209b7e6ed54 \
  --deadline-utc "$DEADLINE_UTC" --stop-file /qc/STOP --decode-workers 4
```

本次实测分配为 64 GiB RAM、8 个 CPU；检查器进程峰值 RSS 为 5,001,781,248 字节，约 4.66 GiB。这不是容器总内存或模型训练容量，也不代表已经验证更小配置。解码线程限 1–4，待处理队列不超过线程数两倍；压缩包同时流式读取也不超过 4 个，不在内存中保留整套图像 payload。

## 验证内容与结果边界

1. 核对官方标注、PDM、词表、split、源码和原配置。十六进制 log_token 与官方 log_name 通过已锁定标注建立一一映射，不直接比较这两套名称。
2. 每张必需 JPEG 都计算 SHA，执行 Pillow `verify()`，重新打开并 `load()` 完整解码，检查尺寸/RGB及紧接解码后的内容 SHA。缺历史、缺相机、PDM 非有限或坏图像均不能通过。
3. 全部 32 包读到 gzip EOF，核对完整压缩字节数/SHA及 gzip CRC/长度。检查每个 tar 头的路径逃逸、大小写别名、重名、父文件/目录冲突、链接、稀疏或特殊类型，以及成员和扩展元数据大小限制。不解压、不修改数据。
4. 必需图像在归档中的成员大小/SHA必须与本次解码账本相同。未选文件仍检查安全头并归属于完整已验证压缩包，无需逐个另算 payload SHA。跨包重复图像只有在匹配相同必需图像字节时才接受并记录；证明的是每张必需图像至少有一个官方同字节来源，不宣称唯一来源包。
5. 归档阶段后，再把全部必需数据图像逐张重算 SHA，与解码账本对照；标注/PDM/词表、源码、split、锁文件也在结束前复核。检查覆盖记录中的读取窗口；只读挂载不能阻止外部主机在之后继续改文件。

正式通过要求 `summary.json.result` 为 `pass_complete_original_NAVTEST_inputs_and_required_JPEG_decode_CPU_only`，actual_exit0，零解码错误，142,976 张全部通过，且两项像素/归档来源标志为 true。需一并保存 `images.jsonl`、逐包成员账本及整文件 SHA。exit1 表示拒绝，exit2 表示 STOP、信号或截止导致未完成，都不能当完整输入通过。

没有原压缩包时，可以显式选择 `--assembled-only`，并省略三个 archive 参数。它仅检查标注与已有图像树，结果是不同的 `pass_NAVTEST_metadata_and_JPEG_decode_without_publisher_pixel_provenance_CPU_only`，两项来源标志仍为 false，不能证明官方相机像素来源，也不能替代正式归档关联结果。

中断后可在全新输出中，使用同一个公开 worker 产生、SHA 已锁定的旧 summary 和 JPEG 账本，成组传入 `--resume-report`、`--resume-report-sha256`、`--resume-ledger`、`--resume-ledger-sha256`。旧通过图像仍重新校验当前内容；重复行、截断或变更会拒绝。压缩包会重新流审。不能把旧 aad3 私有报告隐式当成新 worker 的报告。

```bash
python -m unittest discover -s ascend/tools/tests -p 'test_*navtest*.py' -v
```

合成 tar/JPEG 夹具只验证检查机制。完整官方数据验收须保留本公开源码的独立实跑证据，不能由单元测试或旧私有结果替代。
