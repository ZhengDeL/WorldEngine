# Public NAVTEST CPU input validation — October 20, 2026

The public input checker completed the original NAVTEST input audit in the fixed
Linux image: 12,146 anchors from 136 logs, four-frame input queues and eight
cameras. All 142,976 required JPEGs passed complete decoding, with no errors.
The container exited with code 0 and no OOM. A separate local audit subsequently
reconstructed the original image and archive-member ledgers and confirmed their
exact size/SHA256 linkage, coverage and ordered digests.

This result covers CPU input validation and its recorded evidence. It does not
establish NPU inference, training convergence, GPU parity, model quality or
official NAVSIM scores. The local ledger audit did not decode the images again
or reread the publisher archives.

| Check | Recorded result |
| --- | --- |
| Official inputs | Annotation, PDM cache and vocabulary whole-file hashes unchanged before/after; 12,146 anchors and 136 logs covered |
| Input frames and cameras | 17,872 unique queue frames, eight cameras; 142,976 unique required JPEGs |
| JPEG content | 142,976 RGB images at 1920 × 1080 fully decoded; zero errors; 30,491,135,031 file bytes |
| Publisher archives | Original worker verified all 32 complete archive SHA256 values and gzip EOF/CRC; 127,882,665,618 compressed bytes; entries match the fixed public publisher lock |
| Member linkage | Separate ledger reconstruction confirmed all 142,976 JPEG size/SHA256 pairs and exact union coverage, with zero duplicate matches |
| Final dataset rehash | Original worker rehashed all 142,976 required files after the archive stage; the recorded ordered digest was independently reconstructed from the original ledger |
| Source and CPU execution | Source `0488bfb7…` / 245 Python files unchanged; no accelerator devices or initialized CUDA/NPU before/after |
| CPU regression | 28 Linux input-checker tests and 14 ledger-tool tests passed without skips before their respective actual runs |

The run used public worker `d5b5de5c…`, archive helper `7c5ccd3b…`, archive lock
`c62488d0…` and image `3522f3ac…`. The six code, fixture and lock artifacts checked by the launch retain their
recorded identities. The result pages and operation guides were updated after
the independent audit; they do not change those operational artifacts. The
earlier private-worker input result remains a separate record.

The tested container allocation was 64 GiB RAM and eight CPUs, with four decode
workers and up to four archive workers. The checker process reported a peak RSS
of 5,001,781,248 bytes (about 4.66 GiB); this is neither total container memory nor
a training-memory requirement. Worker elapsed time was 1,093.80 seconds, recorded
for this run rather than as a performance benchmark. Hash checks cover their
read windows and do not prevent later external changes to the dataset.

For the independent check, the completed container's original records were
packaged without modifying the source output. The 17,175,318-byte package
contains 36 regular files, including its manifest and a CPU container identity
proof. Auditor `c10bb5f9…` checked the package SHA256, original summary SHA256,
all 33 raw image/member ledger SHA256 values, parsed record bytes and post-read
stability. It then reconstructed the exact image/member union, reference digest
and final ordered JPEG digest. The actual local command exited with code 0;
its report SHA256 is `21a79c1c55d59afd4d2c016638ff1b44d4efecf8496224b26e6d78b755f9bc63`.

The original summary SHA256 is
`0131f7cebcf65ab1183b54233291d03bfeae99bbdee7888aafe841569b490cb2`.
The [machine-readable record](../ascend/results/full-navtest-public-inputs-20261020.json)
contains per-archive identities, the evidence-package SHA and both ordered
image digests. See the [operation guide](navtest_input_audit.md) for reproduction
of the public input check. Public records omit internal host paths and
credentials; raw deployment receipts remain with the local evidence.
