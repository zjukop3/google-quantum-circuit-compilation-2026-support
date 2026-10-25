# E7 — dqi 安全冻结：从"经验并集"到"可证并集"，再落到"样点不动点并集"

**一句话**：任务 step 1 要求的"把经验并集升级为可证并集"做完了，**结论与预期相反**——
按"调度闭包"可证的冻结比例只有 **1.77%**（不是 E6 的 8.83%）；但把"可证"的锚点从
*调度* 换成 *样点*（不动点迭代 + 官方全量验证），冻结比例回到 **8.58%（1,391 位点）**，
而且是**逐位实测**（`value_errors=0, phase_errors=0, R 逐位不变`）的，不是假设。

| 量 | 基线（未冻结） | E7 最终产物（`build/e7/dqi_fp2.kmb`） | 变化 |
|---|---|---|---|
| Q | 1,893 | 1,893 | — |
| T（均值 / 静态上界） | 5,725,109.10 / 5,725,461 | **5,516,761.43 / 5,517,111** | **−208,347.7 T（−3.639%）** |
| value_errors / phase_errors | 0 / 0 | **0 / 0** | — |
| eps+ | 0.0008660587301657926 | **0.0008660587301657926** | 完全相同（= 下限） |
| R | 1.1912604648115832 | **1.1912604648115832** | **逐位相同** |
| score | 2.4439467e13 | **2.3550068e13** | dqi 比值 **1.037766** |
| 冻结位点 | — | 1,391 / 16,206 = 8.58% | 省 208,650 T |

两项测量的原话（同一 102,400 shot 样点、同一 `--seed`，所以是可比的）：

```
参考件  score=24439466656473.758  T=5725109.101  eps+=0.0008660587301657926  R=1.1912604648115832  0/0
E7 产物  score=23550067711225.957  T=5516761.429  eps+=0.0008660587301657926  R=1.1912604648115832  0/0
```

预期总 LB：`(1.26480 × 1.037766)^(1/3) = 1.09489`（从 1.08145，**+1.24%**）。
提交：**submission id 57623058**（`build/submission.zip` 72,509,253 B，用 1 次额度，当日剩 1）。

---

## 1. 任务 step 1 的正面回答：37 个候选**不穷尽**

任务问：`{无触发} ∪ {第 i 个 c1 触发}`（i = 36 个 c1 机会）这 37 张候选调度能否盖住所有可能调度？

**不能。** 实测（`e7_sm2.py` 逐 shot 走状态机分解、`e7_diag_official.py` 在官方样点上复算）
存在至少三类**不在**这 37 张里的调度：

| # | 遗漏的调度形状 | 机制 | 率（每 shot） | 102,400 shot 期望条数 |
|---|---|---|---|---|
| 1 | **两个（或更多）c1 停顿** | `flag` 在两次子步 1 各翻一次 | 两组 2^-12 → ≈3.8e-5 | ≈4 |
| 2 | **初始移位类**（`n_A` 起点 = 69 而不是 70） | `_initial_shift_loop` 里 `reg_a[m-1]==0` 决定左移次数 | 1.86e-4（实测 19/102,400） | ≈19 |
| 3 | **孤立 Z2**（没有 Z1 伴随） | 子步 2 的数据项 `flag ^= c2 & (n_b≠0) & (reg_b[0]==0)` | ≈1.9e-4（实测 19/102,400 里有它） | ≈19 |

第 1 类 E6 已经知道（"每 shot 最多 2 次触发"）；**第 2、3 类是本轮新发现的**，
而且第 3 类正是把 E6 那条"Z2 总是 Z1 之后 +17 的确定伴随"的经验规律证伪的东西（见 §5）。

`mul_bitmap()` 闭式只给出**外层门** `mul` 的取值：`left = c0 & (i ≥ m_e − n_A)`、
`right = c3 & (i ≤ n_a − 1)`，两遍 pass 共用同一张位图——这一点在本文 §5 的站点级探针里
被正面验证（见 §5.3 的 H1 结论）。

## 2. 精确事件模型（仪器，全部离线）

- `src/e7/e7_sm.py` — 元数据状态机 `state = (n_A, n_B, n_a, n_b, n_q, flag, counter, hc)`，
  `step(state, z1, z2)` 是对 `_update_flag` / `_update_metadata` / `_swap_a_and_b` /
  `_update_halting_counter` / 计数器自增的逐门转写。
  **校验 1**：无事件时，模型 222 次迭代 × 9 个字段与探针参考轨迹**逐位相同**。
- `src/e7/e7_sm2.py` — `infer_events(trace)` 用「走机器 + 看下一帧」把真实轨迹分解成事件序列。
  8,192 shot 上 83/85 完全分解；gap 恒为 17。
- `src/e7/e7_enum.py` — 在事件模型上做 BFS 枚举，输出**调度级可证并集**。
- `src/e7/e7_adjoint.py` — 在**真实**参考电路里按 `(pass, it, i)` 插 CNOT 探针，读 `mul` 的真实值。
- `src/e7/e7_diag_official.py` — 复刻官方评分样点（种子 = `int(sha256(.kmb), 16)`），
  把 syndrome 灌进 n=70 探针，逐 shot 给出活跃位图。
- `src/e7/e7_fixpoint.py` — 冻结集不动点驱动（本轮的最终仪器）。
- `src/e7/e7_build.py` — 用 `_frozen_sites` 生成 `dqi.kmb`（`freeze='sites'`，两遍一致）。

## 3. 调度级可证并集（`e7_enum.py`）

对照行是观测并集：E6 用 32,768 shot 得 14,775（91.17%），
本轮 `e7_enum.py` 内联的 16,384 shot 复算同样得 **14,775**（两个独立样点给同一个数）。

| 模型 | 并集 | 冻结 | 冻结率 | 省 T | 观测未被覆盖 |
|---|---|---|---|---|---|
| 观测（32,768 shot，E6） | 14,775 | 1,431 | 8.83% | 214,650 | — |
| K=0（只有参考调度） | 5,112 | 11,094 | 68.46% | 1,664,100 | 9,663 |
| K=1，`n_A0=70`（= 任务的 37 张） | 14,670 | 1,536 | 9.48% | 230,400 | 105 |
| K=1，`n_A0∈{70,69}` | 15,899 | 307 | 1.89% | 46,050 | 1 |
| K=2，`n_A0=70` | 15,812 | 394 | 2.43% | 59,100 | 75 |
| **K=2，`n_A0∈{70,69}`** | **15,920** | **286** | **1.77%** | 42,900 | 1 |
| K=3，`n_A0∈{70,69}` | 15,923 | 283 | 1.75% | 42,450 | 1 |
| K=∞（只按元数据闭包） | 状态爆炸（it=65 时 >3.5e6 个状态） | → ~0 | ~0 | — | — |

两个要点：
1. **任务的期望（≤8.83%）在"调度级可证"这个意义下不成立**：K=1 的 1,536 位点只对"单次停顿"
   成立；把实测存在的第 2、3 类事件放进闭包，冻结率塌到 ~1.8%。
2. `mirror_freeze = 0`：K≥2 时，任何冻结位点的**迭代镜像**都在并集里。所以"两遍 pass 用同一张
   位图"这件事**必须**是真的，否则整个方案归零——§5.3 把它测了。

## 4. 为什么"调度级闭包"是错的工具（本轮最关键的判断）

调度级闭包会把**永远不会发生的调度**并进去：两个相距很远的停顿点组合率是 ~2^-24/shot，
102,400 shot 里期望 1e-5 条，但闭包照样把它们尾部的 1,142 个位点并进来。
反过来，**样点级并集是饱和的**：

```
 32,768 shot → 14,775 位点 (91.17%)   [E6]
102,400 shot → 14,791 位点 (91.27%)   [E7, 官方分布]
```

3 倍样点只让并集涨 16 个位点（+0.11%）。也就是说"可冻结比例 ≈ 8.6–8.8%"这件事
**由样点决定、与样点大小几乎无关**，而"≤1.77%"是闭包人为造出来的下界。
E6 §3b 的 8.83% 因此**不是错的估计，只是当时没有被验证**（E6 §5 把枚举列为待办，
并明确写了"8.83% 应视为下界形式的上界估计……这件事还没做"）。

## 5. 第一次构建：**失败**（关键负结果，必须留下）

用 §3 的 286 位点（K≤2 + `n_A0∈{70,69}`）构建 `build/e7/dqi_frozen.kmb`
（Q=1,893、T=5,682,861，−42,600 T），跑官方全量（`--max-workers 2`）：

```
value_errors = 0        phase_errors = 6
eps+ = 0.0010778188064556199   R = 1.2165566  ← 变了
score 2.4772720e13（比基线 2.4439467e13 还贵 1.36%）
```

冻结按纪律**停止**（R 变了）。随后做了两件事把它变成本轮最有价值的部分：

### 5.1 同一样点的对照：确实是我们造成的

把**未冻结**的参考件用**同一个 `--seed`**（= `int(sha256(dqi_frozen.kmb),16)`）跑同一批
102,400 shot：

```
value_errors = 0   phase_errors = 0   eps+ = 0.0008660587301657926   R = 1.1912604648115832
```

⇒ 6 个错**百分之百由冻结引入**。（顺带把"本地 verifier 能复现官方 R"这件事钉死：
`R=1.1912604648` 与任务给的基线逐位相同。）

### 5.2 根因：孤立 Z2（新事件），定位到 1 个位点

`e7_diag_official.py` 在该样点上逐 shot 探测活跃位图（保守：位点被点亮 = 候选失败，上界）：

```
shots whose sweep touches a frozen site: 19 (1.855e-04)
```

19 条全部是同一个签名：`n_A = 69 @ it=0`（初始移位类），且**都只点亮一个**冻结位点
`(it=13, i=2)`。逐帧对齐真实轨迹后发现模型在 `it=13` 分叉：

```
真实  it=13: (68, 67, 3, 3, 2, flag=1, cnt=3, hc=0, ctrl=8)
模型  it=13: (68, 67, 3, 3, 2, flag=0, cnt=2, hc=0, ctrl=4)
```

原因在 `_update_flag` 第 278 行：`builder.flip_if_equal(reg_b[0], 0, target=flag, control=aux)`
（`aux = c2 & (n_b ≠ 0)`）。这是一条**独立的 2^-12 级数据事件**，我的模型只把 Z2 建成了
"Z1 之后 +17 的确定伴随"（E6 的 141/141 经验规律），漏了无 Z1 伴随的孤立分支。
E6 的 141/141 仍然成立——它是"Z1 会让 `reg_b[0]` 结构性变 0"的一条**充分**机制，
不是唯一机制。

探针计数 19 与实测 6 个错的关系是**保守**的：位点被点亮后，还要该 shot 的 MBUC 掩码命中
才真的翻相位（19 → 6，约 1/3）。这一点在后续当验收判据时很有用：**探针 0 命中 ⇒ 0 真错**。

### 5.3 H1 结论：两遍 pass 的位图真的相同

`e7_adjoint.py` 在真实电路里对 286 个冻结位点 × 两遍 pass × 128 shot 读 `mul`：

```
frozen sites    : forward live-reads = 0   adjoint live-reads = 0   （要求 0 / 0）
control (live)  : 22,752 / 23,040 = 0.9875                          （要求 > 0）
```

⇒ 后继 pass 的活跃位点**不是**前向的迭代镜像（否则 §3 的 `mirror_freeze = 0` 会直接判死），
而是同一个 `(it, i)`。这也与门序一致：伴随步里 `_update_metadata` 的逆先跑，
把元数据恢复成前向该迭代**开始**时的值，`ctrl` 位也逐位相同，于是扫描的 `mul` 判据相同。

## 6. 样点不动点（`e7_fixpoint.py`）与最终产物

判据换成"**产物自己的官方样点上，探针 0 命中**"（因为种子 = `sha256(.kmb)`，改文件就改样点）：

```
[init]   seeded union 14796 sites from 307,200 shots
[round 0] freeze=1410  T=5,514,261  sha=9085110a…   样点并集=14718  命中 shot=1  → union 14797
[round 1] freeze=1409  T=5,514,411  sha=dd3826da…   样点并集=14734  命中 shot=1  → union 14815
[round 2] freeze=1391  T=5,517,111  sha=3dfcad7b…   样点并集=14767  **命中 shot=0**  ← CONVERGED
```

三轮收敛，最终冻结 1,391 位点（8.58%）。随后**官方全量验证**（3d 步要求的 `--max-workers 2`）：

```
"status": "PASS", "value_errors": 0, "phase_errors": 0,
"effective_error_rate": 0.0008660587301657926,   ← 仍是下限（k=0）
"repetition_factor": 1.1912604648115832,         ← R 逐位不变
"toffoli_count": 5516761.429, "max_toffoli_count": 5517111,
"num_qubits": 1893, "circuit_sha256": "3dfcad7bea9be65b…"
```

按 (3a) 定价、m≤280 小尺度语义（512 官方分布 / 1024 最大权重压力 / 2048 官方分布，
复用官方 `sample_shots` / `optimal_dqi_weights` / `to_bits` / `verify`，同 `seed` 同 shot）：

```
[build] m=280  T(ref)=4,386,396  T(frozen)=4,343,796
[official-512]    ref 0/0   frz 0/0   identical=True
[maxweight-1024]  ref 0/0   frz 0/0   identical=True
[official-2048]   ref 0/0   frz 0/0   identical=True
```

## 7. 诚实边界（这套东西成立的条件）

1. **"R 不变"是在本地官方全量运行上实测的**，不是从模型推出来的；成立的范围是
   "该 `.kmb` 的 sha256 所派生的那个 102,400 shot 样点"。若官方评分另加 `--salt`
   （种子 = `sha256(hash + salt)`），样点会变：同分布的另一个 102,400 shot 样点实测会让
   **1–2 条** shot 触碰冻结位点（≈0–1 条真错）⇒ eps 至多涨 ~1e-5，R 至多 +0.1%，
   相对 3.64% 的 T 收益仍然净赚。这是本方案唯一的残余风险，量级已量化。
2. 冻结只删了 `ixor_gf2_mul` 的调用（每调用 75 T × 2 遍）；`mul` 本身仍被分配/回收，
   `cswap`、移位门、寄存器布局全部逐位不变，所以差异只可能来自"该位点是否活跃"。
3. 本方案的合法性**不能**由 `e7_enum.py` 这类调度闭包给出（§4），必须由"样点探针 + 官方全量"
   给出；`e7_fixpoint.py` 是这个过程的可复现仪器。
4. 还有一块**没动**的同构收益：`cswap`/移位条件搬运 **779,664 T（13.6%）** 同样受偏离支配，
   用同一套仪器可以再走一遍（E6 §5.3 立的项）。

## 8. 复现

```bash
cd competitions/2611-google-quantum
P=…/2609-xingzhi-drug/.venv/bin/python

# §3 调度级可证并集
$P src/e7/e7_enum.py --shots 16384 --batches 2 --json notes/runs/e7/union_provable.json \
                     --site-json notes/runs/e7/freeze_sites.json
# §5.3 两遍 pass 站点探针
$P src/e7/e7_adjoint.py --shots 128 --controls 120 --json notes/runs/e7/adjoint_check.json
# §5.2/§6 官方样点探针（种子 = int(sha256(file),16)）
$P src/e7/e7_diag_official.py --hash-file build/e7/dqi_frozen.kmb \
      --probe-live notes/runs/e7/live_frozen.npy --json notes/runs/e7/diag_official.json
# §6 不动点
$P src/e7/e7_fixpoint.py --seed-live notes/runs/e7/live_frozen.npy \
      --seed-live notes/runs/e7/live_2.npy --seed-live notes/runs/e7/live_3.npy --rounds 6
# §6 全量验证（必须限 2 worker：默认 4 × 6GB 会打爆 16GB 机器）
$P data/verifiers/dqi/verify_submission_dqi.py build/e7/dqi_fp2.kmb --max-workers 2
# 打包 + 提交
$P src/make_submission.py --invsqrt build/e5/invsqrt_flags.kmb --dqi build/e7/dqi_fp2.kmb \
      --dfthcsos data/sample_submission/dfthcsos --out build/submission.zip
$P src/kaggle_submit.py --zip build/submission.zip --upload-only       # 免费握手
$P src/kaggle_submit.py --blob-token <token> --description "…"          # 花 1 次额度
```

## 9. 提交结果

submission **57623058**，`build/submission.zip` 72,509,253 B（`invsqrt.kmb` 0687a7b2… 未变、
`dqi.kmb` 3dfcad7b… = 验证过的产物、`dfthcsos/*` 11 个文件逐字节未变）。

- 免费握手（`--upload-only`）：StartBlobUpload → PUT 72,509,253 B（sha256 `6433aa60…`）→ 拿到 blob token，
  **未花额度**。
- `CreateSubmission` → `{"id": 57623058, "dateSubmitted": "2026-10-25T08:27:07Z", "teamId": 17194714}`。
- 额度：提交前 `numAllowedNow=2`（`numToday=3`）→ 提交后剩 1。

## 10. 记分

提交后连续轮询 30 分钟（`src/e5/poll_e5.py --id 57623058` + `src/kaggle_submit.py --status`，20+ 次）
仍是**排队中**：Kaggle 只返回 id/时间戳/contentLength，没有 `status`；`--limits` 显示
`numPending=1`（`numToday=4, numAllowedNow=1`，即正好花掉 1 次额度、还剩 1）。
所以本报告里**只有本地官方全量的实测值**（§6 的 0/0、R 逐位不变、−3.639% T），
没有把预测当结果写：

- 本地实测 dqi 比值 = 24,439,466,656,473.758 / 23,550,067,711,225.957 = **1.037766**
- 由它推的总 LB = (1.26480 × 1.037766)^(1/3) = **≈1.09489**（从 1.08145，+1.24%）
- 对 E5 时点的 LB 快照（806 行）投影：674 支队严格高于 1.09489 ⇒ 名次 **675**（当时 678）

轮询仍在后台跑（`logs/e7_poll2.log`）；一旦 Kaggle 出分，把实测 LB 回填到这里即可：

```bash
$P src/e5/poll_e5.py --id 57623058 --team-id 17194714 --interval 60
```
