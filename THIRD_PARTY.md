# MMDetection fixture provenance

The candidate source does not include copies of MMDetection function bodies.
`extract_journal_test_fixtures.py` extracts them statically from the user's already installed,
SHA-locked source files for local CPU regression tests. Generated fixtures are not
repository assets and should not be committed with runtime outputs.

Upstream: [OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection).
Reference API source: [mmdet/apis/test.py at v2.24.0](https://github.com/open-mmlab/mmdetection/blob/v2.24.0/mmdet/apis/test.py).
License: [Apache-2.0](https://github.com/open-mmlab/mmdetection/blob/v2.24.0/LICENSE),
whose upstream file identifies OpenMMLab as copyright holder. The reference tag matches the actual installed 2.24.0 package measured in the
frozen image. Compatibility still requires the recorded whole-file and function hashes.

The extractor requires LICENSE from the same installed source root as the
statically located functions, preserves it byte-for-byte, and accepts the original NOTICE
when that distribution includes one. Its generated provenance records package
version read as a literal without execution, source path, whole source/version-file hashes, exact function-source hashes and
the copied license/NOTICE hashes. If generated fixtures are redistributed
separately, retain the applicable upstream license, notices and attribution.

The extractor does not import Torch, torch_npu, MMDetection or DrivingSDK.
It does not execute the installed files. Neither static extraction nor the CPU
tests establish device equivalence. Static installed-file extraction and all 27
public schema2 CPU tests were exercised in the fixed image without accelerator
devices or driver mounts. That CPU evidence does not validate NPU inference or
interrupted/resumed equivalence.
