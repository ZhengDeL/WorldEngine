# 迁移文件审查

本次对照上游 `3eded7`，审查提交 `4946f15` 的 179 个迁移条目：178 个普通文件逐原始字节核 SHA，另 1 个 NAVSIM gitlink 核固定提交。按文件职责检查源码与 CLI 合同、依赖构建、数据来源、证据一致性和文档范围，不代表已执行全部代码路径。[逐文件记录](results/file-audit-20261018.json) 保留原审查 SHA、检查、发现及未覆盖范围。

审查后修订了下列六个已有 Python 文件，相关 CPU 回归已实际完成；原张量容差（`atol=1e-6`、`rtol=1e-5`）和原始损失严格门槛（`<1e-4`）均保持不变，详见[实际回归及源码证据](results/source-review-regression-20261018.json)。

| 已有文件 | 修订内容 |
|---|---|
| `ascend/compare_resume.py` | 隔离模型与优化器的失败统计，检查 dtype、有限值及日志／RNG 完整性。 |
| `ascend/run_container.py` | 校验训练后推理的任务／停止参数，并选实际 latest checkpoint。 |
| `ascend/split_mini.py` | 输出目录与源目录分离时，相机链接指向解析后的最终目录。 |
| `ascend/train.py` | 拒绝所选训练 runner 无法执行的提前停止参数组合。 |
| `ascend/validate_msda.py` | 算子比较显式拒绝非有限输入／结果。 |
| `ascend/validate_msda_layout.py` | 布局比较显式拒绝非有限输入／结果。 |

检查点比较器补齐 dtype、有限值、日志与 RNG 完整性检查，并隔离模型失败统计；两个算子比较器显式拒绝非有限值。CLI 门禁拒绝所选 runner 无法执行的停止参数组合；提前停止后的推理选用实际 latest checkpoint。相机链接修复采用两个 CPU 容器及合成两日志夹具验证：按原 launcher 的挂载规则，中间软链不可读，最终目标链接可读；标签、PDM 顺序、六份元数据及相机目标字节不变。此文件系统单元检查与官方数据、模型验收分别记述。

六项修订后，同一 245 个 Python 文件的源码指纹为 `40997952fdc967296ac1bc15d19f3ef89054d1f3e75e25f23cacb48a8197308b`，其余 239 文件规范 LF 字节保持一致。此前两更新设备对照仍归属 `3d46/245`。新源码另经未加数值包装的公开 VADv2 FP32 入口，在 16 个逻辑 NPU 上完成一次更新，随后只读重验全部 16 个 runtime 记录和保存的 checkpoint。私有后置检查最初误用了模型 profile 标签，保留 controller 退出码 2 的原始记录，同时记录模型实际退出码 0 和后续 CPU 复核退出码 0。本次追加运行只证明单更新兼容性。

新文件 `tools/validation/test_validation_guards.py` 在原 179项之外独立审查；其 6 个 CPU unittest 方法全部实际完成且无跳过，采用真实微型 CPU checkpoint，并显式替代 CLI／Docker／设备调用，不执行模型或加速设备。

```bash
CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tools/validation/test_validation_guards.py --source . --verbose
```

本次交付另新增两份审查文档、三份结果 JSON 和该 CPU unittest，其最终字节由所在 Git 提交及已审发布清单绑定，与原 179项审查分别记述。逐文件审查及有界测试不代表独立初始化重放、长周期收敛、全量数据质量或平台等价验收通过；已有 CPU32 差异和历史数值负例继续保留。普通 SDK RECORD 声明也保留明确边界：实际 SDK 字节未改变且精确迁移项已经检查，但未额外核验普通 RECORD 行声明的 hash／size。

逐文件 JSON 分别保留原 `4946` 审查状态，以及六个修订文件当前 `corrected_and_actual_CPU_verified` 状态和新 SHA。VADv2 与 HydraMDP 的原保存 checkpoint 也已经过修正版 checkpoint 比较器重验，不据此改写原设备运行的源码归属。

另对历史文件 [bf16-v7-gradient-coverage-20261016.json](results/bf16-v7-gradient-coverage-20261016.json) 澄清字段含义：`actual_train_anchors:139` 表示所配置子集的锚点库存，并非该次运行实际消费的独立锚点计数。原 JSON 数值及 SHA 保持不变；不由此字段推算消费量，也不借用其他运行的计数。

逐文件 JSON 保留检查项、已有实测、风险级别和未覆盖范围；私有位置已脱敏，证据 SHA256 保留。

| 审查路径 | 审查身份 | 类型 |
|---|---|---|
| `.gitattributes` | `69f2128700376d85a8ac84ab4217df0c04ea7aa735112fb33e7bf0c7d9551f0d` | 配置或依赖锁 |
| `.gitignore` | `6d9926136ba276f191b53c366c892dc307ee858e1e6590692b3bd9ddcaea5d6a` | 配置或依赖锁 |
| `.gitmodules` | `e355a63344bf596a6500689ee27f1f7f0748c323976153fc752e196582f0dadb` | 配置或依赖锁 |
| `README.md` | `2f5c76d185680cb9db391a75b816fece6a3caf9fae94bbfea8580bbb322ae786` | 文档 |
| `ascend/CUDA_BASELINE.md` | `632bdd4c4b8a292708ffb094d21cdf0ff6ce5cbee53556f7333765ff422d4404` | 文档 |
| `ascend/CUDA_BASELINE_zh.md` | `a1d15204d9c24d82618838fa297cb2d621eb63f5556fef4dfad322cd0d0da866` | 文档 |
| `ascend/FULL_ALGENGINE.md` | `319b9ae282a7d40760dde2723d93a69a9c5cbef46b1d31fdf403e573aa89f94b` | 文档 |
| `ascend/MSDA.md` | `5ab7ef3766ae119869d2ff906caeb1a774f0038b1259fc11c9fd9ccb515fe1a0` | 文档 |
| `ascend/MSDA_CUDA.md` | `7c14d3732f31f1c6d8f2005ed56a66048a5628de281ade10e14bc9973e21bba5` | 文档 |
| `ascend/MSDA_CUDA_zh.md` | `273f6a5202b7eb2c8a8052ff3a998598248b19c4aeef103bbd2c9caeb6d8b397` | 文档 |
| `ascend/MSDA_zh.md` | `4cf6b9226de11af617bd2391f9d99abeb9eba7d8d28a7abe4393e5205d1d7466` | 文档 |
| `ascend/README.md` | `79d39570cdc259d9b399cdd8421c0d4300d449c3401a9550b97b2ab35d14f0a4` | 文档 |
| `ascend/README_zh.md` | `7bec4463062fb01921a53bb3944819385c41e43fec785b7ede6f26959c772057` | 文档 |
| `ascend/VALIDATION_zh.md` | `0f7284ae3917a7aeffe1e1bfdaec690899e96c63662af72ea76a5a825ce8a35c` | 文档 |
| `ascend/__init__.py` | `51d65c07c6adde7c65f617aef9f665d333e1f1f88710824f8704033f1f8a6105` | Python 源码 |
| `ascend/assets.lock.json` | `d9803fc42ddbda274f703dcda6d831db2658c1d403dc616704ad6da600bf8062` | 证据或锁定 JSON |
| `ascend/autocast.py` | `0ad97c93cbaeaaf72201eafc9ff61f57aebd7227965566d55148ac238c671afe` | Python 源码 |
| `ascend/compare_resume.py` | `9f2cb1cd4a2638cba9a4be24d8a56242909c638ca0bde616445435982ec35330` | Python 源码 |
| `ascend/compat.py` | `791f78d434d33bd6bfcf94ea4581f4ced66c90845df53bca8f554d2d97ee6152` | Python 源码 |
| `ascend/cpu_memory.py` | `04bf1c7d0991940ef4567b60826b5ec24d4d92e01bec0e2179d30787c86f1c12` | Python 源码 |
| `ascend/data_tools/CPU_REPRODUCTION_VERIFIED.json` | `e6ac2354540ff98cb557d7628dbfc520adcab77d4721022c1ff8b7a799375472` | 证据或锁定 JSON |
| `ascend/data_tools/DELIVERY_VERIFY_REPORT.json` | `50bc96be738189b53a60abb17726b7623ec6ede08ce4f9165664df8142ef7977` | 证据或锁定 JSON |
| `ascend/data_tools/EXPORT_LOCAL_CHECK.json` | `f37434bc5e88fd568924cff001f8ad65451a731c02f0dee3359e785473fca530` | 证据或锁定 JSON |
| `ascend/data_tools/GETITEM_CPU_VERIFIED.json` | `349dc2b222fe3b097ee9c38496d34d3a6db89fc633551928758a3f83f5fd0a4c` | 证据或锁定 JSON |
| `ascend/data_tools/IMPLEMENTATION_CONTRACT.json` | `c4c9b6e6dc2ded8a3c8a522223d7513566c34cdc235a386003bc6ea033fa0a5c` | 证据或锁定 JSON |
| `ascend/data_tools/INPUT_LOCKS.json` | `48b003fea137809b88f5f92c9ebc65e102ec9df504806f5eebc7ba10840cbfb3` | 证据或锁定 JSON |
| `ascend/data_tools/LOCAL_CHECK.json` | `533994ceb0caca89f33ed2e992a1c9ea1112a0e3d4f2d04ce29f10159e88adfc` | 证据或锁定 JSON |
| `ascend/data_tools/NPU_SINGLE_RANK_VERIFIED.json` | `5fbf69a470d47b46cf784d6309d8aa3baaa76949f5afddef2536254114bbfa40` | 证据或锁定 JSON |
| `ascend/data_tools/NPU_TWO_RANK_VERIFIED.json` | `e9aa90ce10abf4e98657b6f51996b7b7068f2095f9ff56ac28ba778977a2276c` | 证据或锁定 JSON |
| `ascend/data_tools/README.md` | `715ef327c1fb20a4d1182c2cce44822468cd9d4e8fd27b05d9c6cb9242a00976` | 文档 |
| `ascend/data_tools/README_zh.md` | `2a8af3c75f85ca811afc696f7772e4152bf513dfb83da96da93a54a3f475cfaa` | 文档 |
| `ascend/data_tools/TOOLS_MANIFEST.json` | `978f73a2c89116a5991fdb612a8f828b1ddc89d346992b1bd339a6821f7ff07a` | 证据或锁定 JSON |
| `ascend/data_tools/export_simscale_minimum.py` | `23254336e52c8ea1ab6d4a1eba283c7f43aefcfec6b6ec5ca5c1e816ffe3c708` | Python 源码 |
| `ascend/data_tools/navtrain139/DATA_PROVENANCE.json` | `4d869939810e359ecf3379aeb26b54c32c4d54f1d96c393089ea57945a9887e2` | 证据或锁定 JSON |
| `ascend/data_tools/navtrain139/INPUTS.lock.json` | `a37413649ee37fee2971b44c9d983c8ab991aa5f7210edae1716464ec463886f` | 证据或锁定 JSON |
| `ascend/data_tools/navtrain139/README.md` | `bf86c039d76ab0bef9725e7ef2d5c49978bc3885c869e5d6e3b3ce1ed78ecc72` | 文档 |
| `ascend/data_tools/navtrain139/README_zh.md` | `9d02948da8f77a8256fd7560ea22a8b9d076cbe790c649529ccdaacf5c4e9271` | 文档 |
| `ascend/data_tools/navtrain139/SUBSET_REFERENCE.json` | `b50b11d4892c43d9c4cd04d8f687973a7a340ab714fb6cee87da91c7efa46956` | 证据或锁定 JSON |
| `ascend/data_tools/navtrain139/TOOLS_MANIFEST.json` | `c3924d3bceda09bb947cf9f0230d650cdf02d241b713c1dd4097bf1a09409589` | 证据或锁定 JSON |
| `ascend/data_tools/navtrain139/VALIDATION.json` | `52739f2e937981f211ad3ab98ed89cbcc658171420f8aeb740d1351d7d25d44b` | 证据或锁定 JSON |
| `ascend/data_tools/navtrain139/prepare_navtrain139.py` | `3a9c43c8db54fe0bf022ae4ff260cf807424bd3e0a49c2032cd943688b535353` | Python 源码 |
| `ascend/data_tools/simscale_convert.py` | `3db9779124634a102f6ccedc7432704c425b9f1d80bcfa9859275156fe007ce6` | Python 源码 |
| `ascend/data_tools/worldengine_8192_logged_reward.py` | `82528bae108918c74ef6a896d12cf652f14bd751353479cfc63924e8d0a6223c` | Python 源码 |
| `ascend/docker/.dockerignore` | `1e3e30a9f69d3f3749d69dcc00cadfce0753e25661ff3b4a776f98a0c149ee1c` | 配置或依赖锁 |
| `ascend/docker/Dockerfile` | `eb6a84991d07ae07b8b7d18082162f05e9d0bec092edc35ba0f3deac859fdd81` | Image recipe |
| `ascend/docker/Dockerfile.post8` | `a6978ae1d1eca22ba5249d1c974868fc280adeb903623e15ca85675a9901219d` | Image recipe |
| `ascend/docker/README.md` | `ed555dbe8e5ee84708534661488843fd2fef210afc363c7164eb63197be9c721` | 文档 |
| `ascend/docker/README_zh.md` | `38d37a6dbe556f04d58fc7b2c8f823b0b08188ddc3a406c164c03a5fd7d24ec0` | 文档 |
| `ascend/docker/dependency_repair/BUILD_VALIDATION.json` | `eca8611549840674722bc31e55d98efbf9f3891bc8c1ae84bfc08f0348c2fb11` | 证据或锁定 JSON |
| `ascend/docker/dependency_repair/Dockerfile` | `0fc208fe76c83313b0cd2e4d745a670769f434a82022048dd90dafbc573f1f0d` | Image recipe |
| `ascend/docker/dependency_repair/README.md` | `44fd14231aa06ddeea95b0f0ae338c0fda2830266cb5fa36884adb487722c2dc` | 文档 |
| `ascend/docker/dependency_repair/README_zh.md` | `ce45a6ce05d3f258f878692e0b173c347a4fe7e7e0eac82b7443c8d23c7f2959` | 文档 |
| `ascend/docker/dependency_repair/STAGING_PROVENANCE.json` | `e384e431a7e03b4e955578937006cf96c0f36e2e68d19053d6257c056d6dbc22` | 证据或锁定 JSON |
| `ascend/docker/dependency_repair/TOOLS_MANIFEST.json` | `17deb06c3a7af4ce2fd28f0461e94cd8e216449272c031b18ab5ae45c65cc8a3` | 证据或锁定 JSON |
| `ascend/docker/dependency_repair/WHEELS.lock.json` | `87fa56d316f4892217c826bf641463697db17aef518ae27b2df668af1f88bbe2` | 证据或锁定 JSON |
| `ascend/docker/dependency_repair/download_wheels.py` | `54ec95cad1aff52524aa4fcc6747ae3a48f0a50c90345e8fd5c2c32e0ee33169` | Python 源码 |
| `ascend/docker/dependency_repair/environment/Dockerfile.environment` | `9bb0270629213816f4a5dcc958258a4a9e10cf66a4f8196eedafbeb1eee9279d` | Image recipe |
| `ascend/docker/dependency_repair/environment/SDK_RECORD_RELOCATIONS.lock.json` | `ffa29b2d01b17e4cff615bc9fe9ee6b676d81681beabf9fd547ba775f9461df5` | 证据或锁定 JSON |
| `ascend/docker/dependency_repair/environment/SDK_TOOLS_WHEELS.lock.json` | `6ce017b935708e8190f2721993417f080cf200c6bb1f59a0133a6169fd5150c5` | 证据或锁定 JSON |
| `ascend/docker/dependency_repair/environment/application_CPU_acceptance.py` | `374658d83e0a4e7c6b542f660078e6a02963f97f23d0cad0b40af35c00d74352` | Python 源码 |
| `ascend/docker/dependency_repair/environment/compiler_python311.sh` | `77c56af9e0d76b51931ec636ebb26867fdb6b406e864942b12b52212a474bdc7` | Shell entry |
| `ascend/docker/dependency_repair/environment/create_sdk_component_view.py` | `199ed55e8354e60c77660acc40dc8d12a26430c471ab9816b71ade81c323e3dd` | Python 源码 |
| `ascend/docker/dependency_repair/environment/geospatial_CPU_acceptance.py` | `c77319f13386bca103f7e0dd5c555547022fcfe7debe1079df2a8eff68110897` | Python 源码 |
| `ascend/docker/dependency_repair/environment/minimal_dependency_reference.py` | `5b32d48ac5da91c25324d15a56b4c05622bd323e12c0fd251312e9d7de5d39eb` | Python 源码 |
| `ascend/docker/dependency_repair/environment/sdk-tools.lock` | `614fee19946f432ee5068b8789aea63d47ee55bd2da3599e0c4882a7575e1261` | 配置或依赖锁 |
| `ascend/docker/dependency_repair/environment/sdk_tools_CPU_acceptance.py` | `5e212aaa0cb4ef3859323711e0ec3f4df6a69d06323f0c86524e1582583a202e` | Python 源码 |
| `ascend/docker/dependency_repair/environment/verify_launch_environment.py` | `c1df8ec986700c81eca2e8d64d2aec6faf210de88ae9d948ad63021d22929ddd` | Python 源码 |
| `ascend/docker/dependency_repair/environment/worldengine_application_environment.py` | `b4fc53b695b01e8d91a94b2faecfbfc26531e52091b83e5e532a158ebf7883bd` | Python 源码 |
| `ascend/docker/dependency_repair/runtime/WHEELS.lock.json` | `ff150d78bc1815666468d71160629a46bf9dbf99951d78438979c750bb904ecd` | 证据或锁定 JSON |
| `ascend/docker/dependency_repair/runtime/apply_minimal_repair.py` | `69e90db1e3115555114cf52aa7b00e2854abc05f19472e0c061906ddff949146` | Python 源码 |
| `ascend/docker/dependency_repair/runtime/argparse.lock` | `9be00d046ec7620215de48471d112b55d3f4a75807922d3ad6683dcf3c71019d` | 配置或依赖锁 |
| `ascend/docker/dependency_repair/runtime/dependency_cpu_acceptance.py` | `5b32d48ac5da91c25324d15a56b4c05622bd323e12c0fd251312e9d7de5d39eb` | Python 源码 |
| `ascend/docker/dependency_repair/runtime/runtime-repair.lock` | `7af2947e5f7fa103596d62a807038027b2a2cd80d9ad95ce11c613f0c8b3596b` | 配置或依赖锁 |
| `ascend/docker/dependency_repair/verify_tools.py` | `4160f2ee2e54a34dcb65298bc0f04d6fe9911da910aaf627b5d5549041ea38f4` | Python 源码 |
| `ascend/docker/extra_requirements.txt` | `5334336e9d9f565f81e6980bdaedf48d006e91f492710bd7f46267faee3f806d` | 配置或依赖锁 |
| `ascend/docker/geo-linux-aarch64-20261013.explicit.txt` | `26c50a16e3be1f1fc9fa1a0c395b7d1f5f68fecb3f7e4daca1f39fe382da0093` | 配置或依赖锁 |
| `ascend/docker/versions-20261013.json` | `e47dd36e52239386a34d9435c4a20fbd84a0b2d5d5c45c318b7d5c296caf8c27` | 证据或锁定 JSON |
| `ascend/evaluation.py` | `6554545b44babb3558bdc645475eaa892ffa90edb2eed9c6d05e3408507eae4a` | Python 源码 |
| `ascend/evidence/eight-rank-resume-20261013-evening.json` | `782d8fc8136b802c46dc00fd21e8b3c4e3fa2bbbdaeb89141c18a8aa4389fe8b` | 证据或锁定 JSON |
| `ascend/evidence/full-algengine-20261013.json` | `42a45b9cd3c1c40be8eb3ce42ce5bb0b564e48cc00dccf7f0abebab596065330` | 证据或锁定 JSON |
| `ascend/evidence/msda-checkpoint-audit-20261013.json` | `29a5cff2e70060098f91c2c15848b6a0a56f44b5363e2633dd6bbfbe184722a9` | 证据或锁定 JSON |
| `ascend/evidence/msda-device-profile-20261013.json` | `3cfc4946c6a987940a8536df8a16173fc5f0319c137d11efe75d4c8193e8e7c8` | 证据或锁定 JSON |
| `ascend/evidence/msda-full-shape-20261013.json` | `eb4ecc4f5df544148c23b01bbb186b4116a8564a1bec84ec60cd68a7ec532596` | 证据或锁定 JSON |
| `ascend/evidence/msda-map-layout-20261013.json` | `b962f8aba45fba7cd0263901c30bf489503a1700d41c997b95cda83fab1d07a8` | 证据或锁定 JSON |
| `ascend/evidence/msda-planning-20261013.json` | `c34e75dd689b99392faef38e06919669baa9e52815f647dd04e07c626b59e74e` | 证据或锁定 JSON |
| `ascend/evidence/msda-precision-20261013.json` | `2dfa75092036b65633991224ca4a13726f385213a304ba0135b5044cadab3cd6` | 证据或锁定 JSON |
| `ascend/evidence/msda-replay-20261013.json` | `513f69d6f5a97fb9cf59253993422a260178f65e1cb8afd836ee98bf7d20533e` | 证据或锁定 JSON |
| `ascend/evidence/msda-source-assets-20261013.json` | `6bfce2a973a081673f97e20500d61e70ee463244372288e26476aa1fab801686` | 证据或锁定 JSON |
| `ascend/evidence/msda-strict-matrix-20261013.json` | `8429f274e3d0e068e872c0a0fd2a9c264046772909dbbbbed047ef4b55d268d7` | 证据或锁定 JSON |
| `ascend/evidence/msda-training-stability-20261013.json` | `ffbe0088d0937ea07f01b19171bb8d3a7bb65c4f73a2ee200b812d5bda0502dd` | 证据或锁定 JSON |
| `ascend/evidence/official-data-origins-20261016.json` | `bdf0feda4359dc424ad1016afc5818d163e219cccb0939fb35adef9da0a8cddc` | 证据或锁定 JSON |
| `ascend/evidence/original-configs-20261013.json` | `eb679340063e88cc0747449e2c564c1206cebcd57f2996301564c8988ba3360f` | 证据或锁定 JSON |
| `ascend/evidence/planning-checkpoints-20261016.json` | `7096fcb6ac62ee0ccb62622574452b55a649984b2847a9bb192158c99f611b34` | 证据或锁定 JSON |
| `ascend/execution/README.md` | `78a11c0304abb2830201df32e8dd0b68828fcf84a0a435cfc94a2528bc5cd33b` | 文档 |
| `ascend/execution/README_zh.md` | `2a42e3c33276f9831773d7346feb9a58409eac5bf9d7a90b76bf4a0283654a68` | 文档 |
| `ascend/execution/synchronous_launcher.py` | `b6c5d067cc7ba54efa4650e8278c721d6d4187a65bc91870a8b4b1fca4405b5f` | Python 源码 |
| `ascend/fetch_assets.py` | `fbaea8896c016c372179aa0ab91a97535aff3130745c45d1b9c2886e708e9ffb` | Python 源码 |
| `ascend/inference.py` | `0f72247eda2dd779acc2c65694082f599cf1fbdca778bd97b578ac82d4b984fb` | Python 源码 |
| `ascend/msda.py` | `6d8c36426e902dd2fc63039a87f1dff745523b6da7b462aef6fe35780f128638` | Python 源码 |
| `ascend/msda_cuda_compatible.py` | `feb35fc82b67dec1fbf8d24c0f5f60d6ff30356e8f419cf37e90d92481e854d9` | Python 源码 |
| `ascend/official_score.py` | `b660903fa70089b8faa679c79455e86731fdcef27b3ae8ab62ec6288a8f74156` | Python 源码 |
| `ascend/perception.py` | `1dbdf23f370deb7ded368824845b7d7667490ef0030f99afc860efffb9037566` | Python 源码 |
| `ascend/precision.py` | `ca51230f1a2bc82b78c9ee15346f58fdd7c73bd42a4eff1a4718990a977c6e8e` | Python 源码 |
| `ascend/preflight.py` | `f39bd2cbe8d6351b25bacd4c5b28632cd5b26e76052ea0d3e3f67f98c11cb85e` | Python 源码 |
| `ascend/probes/full_shape_gate_probe.py` | `a7421b6835e8cf8903398f71259b0bd1a455d883c96c6ade595f41d912cbf449` | Python 源码 |
| `ascend/results/actual-public-reproduction-16chip-20261014.json` | `456fbece0b66018b4b3b3763fa06ec1553d67984e0b61b4ee7c5e7d3b67ad4be` | 证据或锁定 JSON |
| `ascend/results/bf16-autocast-history-cache-cpu-20261016.json` | `dcd71d3fe487c1a58053acc0bfc1fb36ed200010fcb34afa64e5353e1b393743` | 证据或锁定 JSON |
| `ascend/results/bf16-navtrain139-short8-20261016.json` | `bf40ef04f8221d3926939ca4def72d8b42750ce92e085c6617c22cf2e6aaf075` | 证据或锁定 JSON |
| `ascend/results/bf16-official100-navtrain139-cold-20261016.json` | `9a2c4b8e03e274c872a1ac0fbebc1289510a845c9dc6d84a9ec64210543e4b40` | 证据或锁定 JSON |
| `ascend/results/bf16-v6-optimizer-coverage-20261016.json` | `351b2479114c88c979a7c9f8e766ad5e3d09b5254572cc48867c4065e336caa4` | 证据或锁定 JSON |
| `ascend/results/bf16-v7-gradient-coverage-20261016.json` | `1a3ff05a2823964c8f3daaca787f39859a78a23e2edef4b0e53ee98a2d03b3e0` | 证据或锁定 JSON |
| `ascend/results/bf16-v7-navtrain139-short8-20261016.json` | `24b8898ed4f878a3cebd38d4e22f20e61bbb4e454c0cd17bacc29b34fdf4bb83` | 证据或锁定 JSON |
| `ascend/results/camera-layout-original-input-20261014.json` | `127fa17e1793bf0d83087a538d7127ba48b1696c93a9539701475d5c21a64034` | 证据或锁定 JSON |
| `ascend/results/cuda-compatible-navtrain139-short2-20261017.json` | `05f970b36d5693eb18dc11fb3539c46227383cef9e8e693515aaba8b5d9590aa` | 证据或锁定 JSON |
| `ascend/results/cuda-msda-original-20261017.json` | `1d6a862b17a72d9a213eb7c4c5950d6b3ba6f1d387d8b04bb56030e94783b529` | 证据或锁定 JSON |
| `ascend/results/cuda-training-original-20261017.json` | `567e92097d72bbe10ecf1c783411200902a4ad9614170c32b3b9691d3e7eb50b` | 证据或锁定 JSON |
| `ascend/results/default-cpu-full-input-hydra-16chip-20261014.json` | `aa6cbe5dd7fc9f00b035b72f9446215e55236ffee05830f4ac9e576ead41c02b` | 证据或锁定 JSON |
| `ascend/results/eight-hour-perception-stability-20261014.json` | `007f82bd57d34b308179857cb7d2924505a4bef063f1322074f17bb0d81f4eed` | 证据或锁定 JSON |
| `ascend/results/fp32-bf16-v7-matched8-planning-20261016.json` | `e4390f78a6d2986da4a828a6ac6852b3f9309857c53344ca4bbbb3371d69d5a9` | 证据或锁定 JSON |
| `ascend/results/fp32-navtrain139-short8-20261016.json` | `d7d10a782d511d80ab5e34e5e3666a02b7bb5478445a84ea397cca4ea0bf8fd9` | 证据或锁定 JSON |
| `ascend/results/full-navtrain-dataset-construction-20261014.json` | `8436fb0b1ba07ec676e5bb3297320e33b610b3dc5ea4132b6471c326566fd0d4` | 证据或锁定 JSON |
| `ascend/results/full-navtrain-metadata-qc-20261014.json` | `1c47ee0ddb60381ee8fac667d22c13f69ca5613d40fd2ada0bbff8728ff76631` | 证据或锁定 JSON |
| `ascend/results/full-rank-state-diagnostic-20261014.json` | `a2390ebccf3b92e00d6dc9ba58bef172fa9a6d8398a92bab760b35cd0b59c70d` | 证据或锁定 JSON |
| `ascend/results/gpu-npu-official100-eval22-20261016.json` | `1e7f79f613853a62cb5d531aa001e0a4286c0a26d3ad0ffa217c85eefaa8f591` | 证据或锁定 JSON |
| `ascend/results/hydra-ft37-bf16-eval22-20261017.json` | `0d4a2a6b6e072587d4b5d480fc67c8427c4c34a98169f378a3428c8fd91e3b9a` | 证据或锁定 JSON |
| `ascend/results/npu-cuda-msda-20261017.json` | `28af9fdff4ef0501318b0e992a7f09b579d1e7cecb795d0872380aad826da1d6` | 证据或锁定 JSON |
| `ascend/results/npu-private-msda-vad-fp32-replay2-20261017.json` | `b27f6b33810c740b59857aaf6228fea84d23bc1ef977b28601957338f41f169b` | 证据或锁定 JSON |
| `ascend/results/public-sync-repeated-16chip-20261014.json` | `e011d834f4827a699b40159c340da1e7ef2310b86d02631dee15cdf6037c2708` | 证据或锁定 JSON |
| `ascend/results/real-adam-vad-diagnostic-20261014.json` | `7273559148b39d146f992b76725f2bfea48a3e247204bb13f1d75b52d7cc97f3` | 证据或锁定 JSON |
| `ascend/results/real-log-ft37-bf16-v7-short8-20261016.json` | `5b07d23def0b589d4e234b14e21a0208d2622c0794f476d60b61c8f1b91515a5` | 证据或锁定 JSON |
| `ascend/results/strict-replay-post8-20261014.json` | `69ad91ac42142b7870233908031637d0e86cee760b3851bcb3592bf0c0920a56` | 证据或锁定 JSON |
| `ascend/results/sync-hydra-16chip-20261014.json` | `6fdb35dd5db82e812161a3d5830055fcbb9ea9132a0b79a499c9261c21a0b1ff` | 证据或锁定 JSON |
| `ascend/results/sync-vad-16chip-20261014.json` | `40a3a0c6270aa65c8a47cfab78ae09a83548a76ac9fd2c27a959014b81294fb9` | 证据或锁定 JSON |
| `ascend/run_container.py` | `5b0a750625599430b25cc1762cae10816ea75f40560b374b5e1a878144594e0e` | Python 源码 |
| `ascend/runtime.lock.json` | `011a6849c72afd9352ac3be4a4ed7df946b1507ca3fda8f95c4c1efeef3a9362` | 证据或锁定 JSON |
| `ascend/runtime.py` | `bf48ddb35cca4aecc6d15007fee0d107c8463af55ba679441146ccb395693369` | Python 源码 |
| `ascend/scoring.py` | `491702df001102ff358064156e450fe0536fb34ced4d90ffb0d8632f78ec9f16` | Python 源码 |
| `ascend/split_mini.py` | `f095d6a829636880ac226a78a8deca6fa8d816d1eeecef584096a0c99613a82c` | Python 源码 |
| `ascend/third_party/navsim` | `gitlink:3e8291bfa89ff247231e0227778840cd0a036896` | Pinned submodule |
| `ascend/tools/download_algengine_assets.py` | `2c8e4f800860575b6cf7a110716dec679edc792f23882b4d6b3b736e48f47916` | Python 源码 |
| `ascend/tools/launcher_common.sh` | `db6c6d0609ff5b85aa27e165cd92be24457b536122832866d9ba0167cee1216e` | Shell entry |
| `ascend/tools/prepare_real_mini.py` | `cdb7e7b36e170e429f4f95d60fdb2764ba2252d13801d2aec1d0694a57b68d17` | Python 源码 |
| `ascend/tools/score_cached.py` | `c74dbcb53026fa790521b592357b766a8d14cc617b7864728704ada37c4b2965` | Python 源码 |
| `ascend/train.py` | `579349d31766fd4ef9b0eaabdb6408e7a4907902ecc3595a961a6a4f3c0f52a2` | Python 源码 |
| `ascend/validate_msda.py` | `5a898191f716dec2536ab1b7e208c0651ef750b2fdf929c4eaec37d2e1b78d5d` | Python 源码 |
| `ascend/validate_msda_layout.py` | `6b4c946c5881eb2f398582c7eb7c7c9155de7517f3f82e4bb5170e9c623224b1` | Python 源码 |
| `ascend/validation_hook.py` | `a777fe664289316f4aa37aa90ecd9531b26ee9a089dd602e31c40710e583a9c8` | Python 源码 |
| `ascend/validation_results_20261013.json` | `7aafdfca3449efb8a12717ec26bbba68e3d95991ad7d25143a5d44d1599d5a72` | 证据或锁定 JSON |
| `docs/algengine_ascend.md` | `111280412aa11dc19e4f8b2e150ff9ef70485dce49b8825211dd6a5872474484` | 文档 |
| `docs/algengine_ascend_zh.md` | `5edfc397fdbda8f5aa34cad8a3ef46932b550366d5aab8cea394ee363376c5ce` | 文档 |
| `docs/algengine_usage.md` | `c841912204e46213c321ae5987ae9a67c106d93f264b45ec8a8c87a21b9c0bdf` | 文档 |
| `docs/data_organization.md` | `70daa09816478d1e4f74368099be8d33ea834841dc4bcf0add21d0fdaf2491a1` | 文档 |
| `docs/installation.md` | `afc2f1095c062bfcd9221b2705987dea1ce04cbb0401ba5be2b0fbe13236a884` | 文档 |
| `docs/quick_start.md` | `76108c3919a150fc62edb7db2e90d48f556d781a5873ae26337709ef67be0887` | 文档 |
| `projects/AlgEngine/mmdet3d_plugin/datasets/eval_utils/map_api.py` | `39ff26f4c25411893b9fe88a55cb4a991dc1463b6a6fb98b6c900d57b8a4dcab` | Python 源码 |
| `projects/AlgEngine/mmdet3d_plugin/datasets/navsim_openscene_nuplan.py` | `564e9de48056dd8dd88426ce20f238961ce698f96135ec3da084d43fea8f6805` | Python 源码 |
| `projects/AlgEngine/mmdet3d_plugin/datasets/navsim_openscene_nuplan_det.py` | `49283a3610c29ddf0607aa1ae67b94f08c0b33619281055769af522832ca2f70` | Python 源码 |
| `projects/AlgEngine/mmdet3d_plugin/datasets/navsim_openscene_simscale.py` | `70e757e12792b55f1a0c84c7518b32e860bc00ec591c9a8bbac1ea31322af2c8` | Python 源码 |
| `projects/AlgEngine/mmdet3d_plugin/datasets/navsim_openscene_synthetic.py` | `ebe2fdde7769e397ed82ef18ba0325aa5c80cc2b7f4e5bd07362780ff05d954e` | Python 源码 |
| `projects/AlgEngine/mmdet3d_plugin/uniad/apis/mmdet_train.py` | `33373e5032c01f0218bd093aa17e898c5520301a007778ed62f569ea295565df` | Python 源码 |
| `projects/AlgEngine/mmdet3d_plugin/uniad/dense_heads/motion_head_plugin/motion_deformable_attn.py` | `d6b26544e91d3631e9705119e19808c6dfa267847348fd3de19a3a55495a571f` | Python 源码 |
| `projects/AlgEngine/mmdet3d_plugin/uniad/modules/decoder.py` | `4332be57c848b4ec084ce2fbc6acff27e4e21712ed547dee2d498737237c108c` | Python 源码 |
| `projects/AlgEngine/mmdet3d_plugin/uniad/modules/spatial_cross_attention.py` | `4b11b1d66d0da47cfea5663b7130f85925354b890a63238a140e25c6bddbcf17` | Python 源码 |
| `projects/AlgEngine/mmdet3d_plugin/uniad/modules/temporal_self_attention.py` | `1b8398c5691585825a6e867288f252e52ca08eb42c78f4722ae9bfc1a277542c` | Python 源码 |
| `projects/AlgEngine/scripts/e2e_dist_eval_npu.sh` | `a7b82bd8bb2b99963ce0614a1bfd7cfa1881870e4864d88c621ab6ce3da808ee` | Shell entry |
| `projects/AlgEngine/scripts/e2e_dist_train_npu.sh` | `0f4785b8ffdde1e6a69dcd240af4c835665852623faf5c82703a45cdeffaa273` | Shell entry |
| `projects/AlgEngine/scripts/test.py` | `fd4a0ca9fdd5f2a4950e7221cec220e964e8e7f98f7bf1fda4303b81b152e76d` | Python 源码 |
| `projects/AlgEngine/scripts/train.py` | `12377468ea587bec2bce9a976695b14c01097f41f427d6af13ca2edeb1fb2cbf` | Python 源码 |
| `tools/validation/README.md` | `9c91f5086e9bde4e1056f1b983e498a31441f47efadc5f201b3eb6ca422d7911` | 文档 |
| `tools/validation/README_cuda_msda.md` | `9a27f7d9cb177e2dac19aab343d6fdd26899410a950026eecdfb149dbcb8653c` | 文档 |
| `tools/validation/README_cuda_msda_zh.md` | `95c9a794d4909c0cdfd685b03e292d230a6863cfdf1a8eea3a341b2319b3d6f2` | 文档 |
| `tools/validation/README_export_cuda_msda_goldens.md` | `1dd6bd68d90fdde05301d45d9e1fafcd875d0328a4303cddd6f53763dfc9bee2` | 文档 |
| `tools/validation/README_export_cuda_msda_goldens_zh.md` | `19fe74e4373bd5252c6e5710302c243f024f69a4497e7986299c35fa347abc0d` | 文档 |
| `tools/validation/README_npu_cuda_msda.md` | `59c12946381a57bfd1b4b6b405ccb8195ba12cbc094bd1721798473d94001848` | 文档 |
| `tools/validation/README_npu_cuda_msda_zh.md` | `d4dbbec39384defbf0131f64be276e52a8a8f8b635c6e37c94e04b4952358978` | 文档 |
| `tools/validation/compare_planning_outputs.py` | `0ff90c7a44d6fb0b996a4e86594ff9d237ef1684c45da82c16b82c47bea9d3f7` | Python 源码 |
| `tools/validation/export_original_cuda_msda_goldens.py` | `1070635ec466e637adc6f91325972bba8b5567bcee6082447584518200598141` | Python 源码 |
| `tools/validation/validate_original_cuda_msda.py` | `af7309e616f9a02648f9a5b35f8960ea97f51e108b1915bca7843448bd733fea` | Python 源码 |
| `tools/validation/verify_npu_cuda_msda.py` | `bda759e69180c652f2a02e4774641a4a5ddc076536a0e38828d2c02bd74caf55` | Python 源码 |
