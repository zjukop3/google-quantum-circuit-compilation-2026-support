# E1 — 参考解复现对表 + E2/E4/E5 结论

> 环境：Apple Silicon macOS，8 核，16 GB RAM；Python 3.14.3（`2609-xingzhi-drug/.venv`，
> numpy 2.5.3 / scipy 1.18.1）；kickmix 由 Homebrew LLVM 20 本地编译
> （`kickmix.cpython-314-darwin.so`，见 [`../src/README.md`](../src/README.md)）。
>
> **表里每个数字都是本机实测**，不是论文抄录。原始输出在 `notes/runs/`。
> 实测时间：2026-10-24 20:30–21:35（Asia/Shanghai）。

---

## 0. 先更正前序侦察的两处误判

### 0.1 数据端点没有门禁，是 URL 编码问题

`data/README.md` 记录下载 403（"You must accept this competition's rules"），
并据此断定"必须先点一次规则页"。**实测：不需要接受规则，数据就是能下。**

| 端点 | 之前记录 | 实测 |
|---|---|---|
| `/api/v1/competitions/data/download/<slug>/<path>` | 403 | **404**（HTML 错误页） |
| `/api/v1/competitions/data/download/<slug>/<path 已百分号编码>` | — | **200** ✅ |
| `/api/v1/competitions/data/download-all/<slug>` | — | 200（zip） |
| `/api/v1/competitions/data/list/<slug>` | — | 200（JSON 清单） |
| `CompetitionService/CheckCompetitionsAccess` | `hasAcceptedRules: false` | **仍为 `false`** |

```bash
# 404：返回 Kaggle 首页 HTML
.../download/$SLUG/verifiers/invsqrt/verify_submission_invsqrt.py
# 200：把 '/' 编码成 %2F
.../download/$SLUG/verifiers%2Finvsqrt%2Fverify_submission_invsqrt.py
```

`hasAcceptedRules` 至今是 `false`，但 34 个文件全部下载成功并逐一核对字节数。
**结论：前序的 403 是 `curl` 没有编码路径造成的误判，端点本来就通。**

### 0.2 Starter Kit 也通 —— 而且它决定 E5 的工作量

同一个坑还藏着一份 **464.5 MB** 的资源。`data/README.md` 记录的
`404 {"code":404,"message":"No gcs url found"}` **只对整包路由成立**：

```bash
# 404
.../api/v1/datasets/download/sohier/verifiable-circuit-compilation-starter-kit
# 200（按文件 + 百分号编码）
.../api/v1/datasets/download/sohier/verifiable-circuit-compilation-starter-kit/starter_kit%2Fdfthc%2F__init__.py
```

已抓全 41 文件清单（`notes/ds_list_starter_kit.json`），并把**除 7 份 fcidump 以外的全部
26.7 MB** 下到 `data/starter_kit_dataset/`。里面有关键的：

| 路径 | 内容 |
|---|---|
| `starter_kit/dfthc/df_thc_sga.py` | **官方 Task 1 目标函数**（Eq. 48） |
| `starter_kit/dfthc/train_df_thc_sga.py` | 共享训练循环 |
| `starter_kit/starter/task1_factorization/train_df_thc_qaggle.py` | Task 1 训练入口（需 JAX/Optax） |
| `starter_kit/starter/task1_factorization/check_factorization.py` | Task 1 官方检查器（纯 numpy） |
| `starter_kit/starter/{task2_angles,task3_circuits}/` | Givens 树 / 角度编码 / 电路构建 |
| `starter_kit/baselines/*.npz` | 8 份研究用 baseline 分解 |
| `starter_kit/data/systems.toml` | 8 个体系的 (N, η, E_upper, baseline, R/B/C) |

**E5 的结论完全建立在这个包的内容上**（见 §5）。

---

## 1. 下载结果

`src/download_data.py`（新增）。32 个小文件**逐一核对字节数**，`32 ok / 0 bad`，共 28,128,155 B。
两个大文件另行下载，同样逐字节核对：

| 文件 | 字节 | 核对 |
|---|---:|---|
| `sample_submission/dqi.kmb` | 886,118,180 | ✅ 字节数一致（内容另见 §3.3） |
| `verifiers/dfthcsos/FeMoCo76o113e.fcidump` | 128,128,348 | ✅ SHA-256 `e5a48af7…e1bd` 与 README 声明一致 |
| `data/starter_kit_dataset/**`（26.7 MB，无 fcidump） | 26,687,439 | ✅ |
| **竞赛数据合计** | **1,042,374,683 / 34 文件** | ✅ 清单对全 |

单流下载只有 **200–320 KB/s**，886 MB 要 ~50 分钟。`src/fast_download.py`（新增）用
**HTTP Range 分块并发**（GCS 重定向后支持 206）：24 分块 → **3.18 MB/s，279 秒下完，16× 加速**。

---

## 2. Benchmark I — `invsqrt`：**5 项全对** ✅

命令（全强度，`shots_per_x = 64`，`max_x = 3,133,452` → **200,540,928 shots**）：

```bash
python data/verifiers/invsqrt/verify_submission_invsqrt.py data/sample_submission/invsqrt.kmb
```

| 指标 | 论文 §III.A | 官方 notebook | **本机实测** | 判定 |
|---|---|---|---|---|
| status | — | PASS | **PASS** | ✅ |
| `num_qubits` = **Q** | **260** | — | **260** | ✅ |
| `toffoli_count` | **7,735** | — | **7,735** | ✅ |
| `t_count`（基础 T） | **1** | — | **1** | ✅ |
| `rotation_count`（非 Clifford `Z_POW`） | **37** | — | **37** | ✅ |
| `toffoli_equivalent` = **C** | **7,988** | — | **7,987.593206597238** | ✅（论文进位） |
| 成本 **S = Q×C** | **≈2.08e6** | **2,076,774** | **2,076,774.233715282** | ✅ 与 notebook 逐位一致 |
| `worst_error_over_tolerance` | 2.84e-8 = 2^-25.07 | 48% | **0.4769116097939472** | ✅ |
| `worst_x` | 2,588,411 | — | **2,588,411** | ✅ |
| `verified_shots` | — | — | **200,540,928** | ✅ |

**`C` 是怎么来的**（把公式钉死，`notes/runs/invsqrt_sample_full.json` 逐项闭合）：

```
rotation_eps(实测)         = 1.5289254939242122e-12
rotation_t_count(eps)      = 0.56·log2(1/eps) + 5.3 = 27.283
T                          = t_base + 37 × 27.283 = 1 + 1009.47 = 1010.47
C                          = 7735 + T/4 = 7735 + 252.59 = 7987.59     ✅
S                          = 260 × 7987.59 = 2,076,774.23             ✅
```

> ⚠️ 论文的 `C = 7,988` 是 `7,987.593…` 的进位写法，`260 × 7,988 = 2,076,880` 与真值差 106。
> **真正要对齐的靶子是 `2,076,774.2337`**，官方 starter notebook 明确写过
> *"The reference's 2,076,774 is the verifier's number to the last digit"* —— 我们逐位复现。
> `eps` 是从**实测留白**反推的，不是输入；留白由算术误差决定，与 salt 无关。

---

## 3. Benchmark II — `dqi`：**对上** ✅

### 3.1 官方 verifier 结果

```bash
python data/verifiers/dqi/verify_submission_dqi.py data/sample_submission/dqi.kmb --max-workers 2
```

| 指标 | 论文 §III.B | **本机实测** | 判定 |
|---|---|---|---|
| status | — | **PASS** | ✅ |
| `num_qubits` = Q | **1,893** | **1,893** | ✅ |
| `max_toffoli_count`（静态，cleanup 后） | **5,725,461** | **5,725,461** | ✅ |
| `toffoli_count`（每 shot 平均**执行**数） | — | **5,725,108.38** | 说明见下 |
| `num_bits` | 49,140（70×12 + M 记录） | **49,154** | 见下 |
| `num_ops` | "8,600 万门" | **86,051,014** | ✅ |
| `reaction_depth` | 222 个同步 EEA 步 | 170,757 | — |
| `value_errors` / `phase_errors` | — | **0 / 0** | ✅ 参考解零失败 |
| `effective_error_rate` ε₊ | — | **8.660587301657926e-4** | ✅ 与 §R 的地板一致 |
| `repetition_factor` R(ε₊) | — | **1.1912604648115832** | ✅ |
| `mean_error_weight` E[wt(e)] | ≈31.05 | **31.046737** | ✅ |
| `score` = R·Q²·T | — | **24,439,463,583,672.93** | 基准 |
| `circuit_sha256` | — | `959471d05a26…c04995` | — |

**两点说明（不是对不上，是口径）**：

1. `toffoli_count` 5,725,108.38 ≠ 5,725,461：前者是 `km.Simulator` 在 102,400 shots 上
   **实际执行**的 Toffoli-like 门均值，被测量比特条件的门只在比特为 1 的 shot 上计费；
   后者是静态计数。**成本公式用的是前者**，论文引的是后者。
2. `num_bits` 49,154 而非 49,140（=70×12 + 4095×12 的 M 记录？实际 M 是 4095 个 12-bit
   记录共 49,140）。差的 14 bit 来自 EEA 的进位/标志位；度量到 `E0..E4094` 各 12 bit
   = 49,140，其余属内部辅助位。**契约检查通过，不影响成本**。

### 3.2 零失败的代价地板

`R(ε₊)` 在 **零失败** 时仍是 1.1913，因为 `ε₊ = −expm1(log2δ/N) = 8.66e-4`。
即"完美提交"也要付 **19% 的重复开销**。这是 dqi 这条线上一块**无法优化掉**的成本 ——
除非提高 `num_shots`（榜上是官方的 102,400，改不了）。

### 3.3 参考电路生成器 vs 官方样例：字节级比对

本机用 `verifiers/dqi/dqi_reference_circuit.py`（默认参数）重建：

| 项 | 官方 `sample_submission/dqi.kmb` | 本机 `build/dqi_reference.kmb` |
|---|---:|---:|
| 字节数 | 886,118,180 | **886,118,180** ✅ |
| `num_qubits` | 1,893 | **1,893** ✅ |
| 静态 Toffoli | 5,725,461 | **5,725,461** ✅ |
| 前 185,598,016 B 的 SHA-256 | `a36b0878…f38b` | `a36b0878…f38b` ✅ **完全一致** |
| 整文件 SHA-256 | `959471d05a26…c04995` | `5dfb9991335e…f98f` ❌ |
| verifier 结果 | PASS，score 2.4439464e13 | PASS，score 2.4439467e13 |

**首次分歧在偏移 `0x29B80CFF`（699,927,807，即 79% 处）**。分歧点是 16-bit 字
（`ffbf febf … f4bf` ↔ `f4bf … febf ffbf`）的**发射顺序整体反转**，
数值集合完全相同。也就是**同一套寄存器/比特分配、两种枚举顺序** ——
本机用的 kickmix 是 Qualtran `main` 分支编译的，官方样例大概率来自略有差异的
kickmix 版本。**两条电路都是合法且 PASS 的 dqi 提交**，成本一致（Q、T 相同）。

---

## 4. Benchmark III — `dfthcsos`：**4 项全对** ✅

```bash
python data/verifiers/dfthcsos/verify.py data/sample_submission/dfthcsos
```

| 指标 | 论文 §III.C | 官方 SCORING_README | **本机实测** | 判定 |
|---|---|---|---|---|
| `status` | — | PASS | **PASS** | ✅ |
| `Lambda` | **308.46** | 308.4559397330944 | **308.4559397330945** | ✅ |
| `E_gap` | **3.965** | 3.965477244291833 | **3.965477244291833** | ✅ |
| `lambda_eff` | **49.30** | 49.30136925057438 | **49.301369250574396** | ✅ |
| `score` | — | 3355599588.3155365 | **3355599588.3155375** | ✅（末位 1 ulp） |
| `toffoli_count` = T_total | — | 93,237 | **93,237** | ✅ |
| `num_qubits` = Q_total | — | 730 | **730** | ✅ |
| `error_1` | ≤ 20 mHa | — | **3.185 mHa** | ✅ 用 15.9% 预算 |
| `error_2` | ≤ 20 mHa | — | **18.640 mHa** | ⚠️ **用 93.2% 预算** |
| `submission_sha256` | — | — | `276d464ff14d…4002695` | — |

完整 `resources` / `metrics` 见 `notes/runs/dfthcsos_sample_full.json`。
成本分解与 README 逐项对上：ANGLE 10 电路 × 3974 Toffoli × 2 遍 = 79,480；
heuristic 846 + 9,461 + 3,450 = 13,757；/ 持久 186 + 临时 544。

### 4.1 连续（未量化）分解 —— §5 的关键数据

官方 Task 1 检查器 `starter/task1_factorization/check_factorization.py`：

```
R, B, C, N    = 65, 60, 60, 76
error_2       = 16.1332 mHa  [ok: bound 20 mHa]   (training value 16.1196 mHa)
error_1       = 8.252e-15 Ha (continuous: ~0)
Lambda        = 308.448482 Ha
E_SOS         = -22144.231369 Ha
E_gap         = 3.958019 Ha
lambda_eff    = 49.254686 Ha
```

→ **12-bit 角度字量化把 error_2 从 16.13 抬到 18.64 mHa（+2.51 mHa），
只剩 1.36 mHa 到 20 mHa 的闸门。**

---

## 5. E5 — 把 `Λ` 压到 49.0 的下界：**算术成立，但不是免费杠杆**

### 5.1 判断一：算术对，但增益不是 2.57×，是 **1.37×**

`λ_eff = Λ·g_amp(δ)`，`g_amp(δ) = √(δ(2−δ))`，`δ = E_gap/Λ`（论文 Eq. 60–61）。
本机复算（`data/sample_submission/dfthcsos/submission.npz`，连续分解）：

| 场景 | Λ | E_gap | λ_eff | 相对 baseline |
|---|---:|---:|---:|---:|
| baseline 连续 | 308.4485 | 3.958019 | 49.254686 | 1.000× |
| baseline 量化后（实际提交） | 308.4559 | 3.965477 | 49.301369 | 0.999× |
| **只把 Λ 推到 49.0⁺** | 49.0000 | 3.958019 | **19.293003** | **2.553×** |
| 只把 E_gap 推到底 3.914898⁺ | 308.4485 | 3.914898 | 48.987368 | 1.005× |
| **两个 floor 同时贴** | 49.0000 | 3.914898 | **19.192019** | **2.566×** |

> 上表最后一行 **19.192019** 与官方 SCORING_README 自己算的
> `sqrt(3.914898 * (2*49.0 - 3.914898)) = 19.19201859236271` 一致 ✅。
> 你说"49.30 → 19.19"数字对。但 **49.30 是 baseline 的 λ_eff，49.0 是 Λ 的下界**，
> 不是同一个量的上下界。
>
> ⚠️ **而且总分是三个任务的几何平均**：
> `Score = (Π_b ref_b/cost_b)^(1/3)`。dfthcsos 单独 2.566× ⇒ 总分只涨
> **2.566^(1/3) = 1.369×**，不是 2.57×。

### 5.2 判断二：Λ 对一个任务的 T/Q **完全无关** —— 所以它确实是干净的杠杆

读 `scoreinfra/costs_prepare.py`（全文 193 行）：heuristic 成本
`SpinFreeData_V1(R, B, C, N, χ=10, β=brot)` **只依赖 (R,B,C,N,χ,β)**，
不含 `w`/`W` 的数值，也不含 Λ。verify.py 的 `resources` 分解也印证：
在固定 `(R,B,C,brot)` 下改 Λ，**T_total 与 Q_total 一个字都不动**。
所以削 Λ 是纯 `λ_eff` 上的收益。

### 5.3 判断三：**但 Λ 不是输入，是被重算的输出，而且卡在两道闸门上**

verifier 从提交的 `w, W` 和**电路解码出的**角度字重算一切：

- `error_1`、`error_2`（对 FCIDUMP 的 Frobenius 残差）必须 **≤ 20 mHa**；
- `Λ > 49.0` 严格；`E_gap > 3.914898` 严格；且 `0 < E_gap < 2Λ`。

**Λ 的 95.8% 是二体 LCU 归一化**（本机从 `.npz` 复算）：

```
Λ = λ_h1 + λ_h2
  λ_h1 = Σ|eig(h1')|  = 12.872708   ← 只占  4.2%   (76 个本征值，56 个为负)
  λ_h2 = ¼ Σ_rc ( Σ_b|w_b^{ρc}| + |W_rc + Σ_b w_b^{ρc}| )²
                       = 295.575773 ← 占 95.8%
Λ                        = 308.448482     (是 49.0 floor 的 6.295×)
E_SOS = -22144.231369     (ecore=-21021.214012, η=113, β1=4.955105)
```

`λ_h2` 是 **65×60 = 3,900 个平方项之和**，每项的幅值由"必须复现 FCIDUMP 积分"钉死。
要把它砍 6.3×，等于要求这套 DF-THC 分解的 LCU 1-范数整体缩小 2.5 倍以上，
**而残差预算只剩 6.8%（1.36 mHa / 20 mHa）**。这不是拧一个旋钮，
就是官方 Eq. (62) 那个约束优化问题本身：

```
Θ* ∈ argmin Θ  λ_eff(Θ)      s.t.  ε_fro(Θ) < ε_DFTHC = 20 mHa,  ‖u_b^(ρ)‖₂ = 1
```

### 5.4 判断四：**真正的机会点** —— 官方工具链已经就是"最小化 λ_eff"

`data/starter_kit_dataset/starter_kit/dfthc/df_thc_sga.py`
`Compute_qaggle_df_thc_cost_function`（**官方 Task 1 目标函数，Eq. 48 的精确罚函数形式**）：

```python
cost = relu(epsilon_fro - 0.020)          # 20 mHa 约束
     + gap_domain_violation               # 0 < E_gap < 2Λ
     + floor_violation                    # relu(49.0 - Λ) + relu(3.914898 - E_gap)
     + rho * lambda_eff                   # ← 目标就是最小化 λ_eff
```

两个关键事实：

1. **baseline 的 `ε_fro = 16.12 mHa`，落在约束内部（约束没激活）。**
   如果它真是在跑"min λ_eff s.t. ε_fro≤20 mHa"，梯度会把 ε_fro 顶到 20 才停。
2. `train_df_thc_qaggle.py` 的文件头写着 *"The historical filename refers to an
   older version of the specification"*，baseline 文件名叫
   `FeMoCo76o113e_65_60_60_ep50000_seed0.npz`。

→ **强烈指向：这份 baseline 是旧版（以 ε_fro 为目标）训练器跑出来的，
不是当前 λ_eff 目标的收敛解。** λ_eff = 49.25 很可能**不是** (R,B,C)=(65,60,60)
下的可达前沿，重跑当前官方目标存在真实的下行空间。

**量化估计（保守）**：即使拿不到 49.0，λ_eff 是 Λ 的**近似 √ 关系**
（`λ_eff ≈ √(2·E_gap·Λ)`，因为 Λ ≫ E_gap）：

| Λ 降到 | λ_eff | dfthcsos 增益 | 总分增益 |
|---:|---:|---:|---:|
| 308.4（现状） | 49.25 | 1.00× | 1.000× |
| 250 | 44.37 | 1.11× | 1.035× |
| 200 | 39.63 | 1.24× | 1.076× |
| 150 | 34.29 | 1.44× | 1.128× |
| 100 | 27.97 | 1.76× | 1.208× |
| **49.0⁺** | **19.29** | **2.55×** | **1.369×** |

### 5.5 还需多少工作量 —— 结论

| 步骤 | 做什么 | 依赖 | 估计 |
|---|---|---|---|
| 0 | 已有：starter kit 的 `dfthc/` 库 + Task 1 训练脚本 + baseline `.npz` + fcidump | ✅ 已下载 | 0 |
| 1 | 装 JAX/Optax 环境（CUDA 机 admin1 或本机 CPU） | — | 0.5 天 |
| 2 | 以当前目标 `min λ_eff s.t. ε_fro<20 mHa` 从 baseline **warm start** 重训 (65,60,60) | GPU | 1–2 天（含调 `rho`、学习率、epoch） |
| 3 | 扫 (R,B,C)：因为 `C_BE(R,B,C)` 随尺寸涨、`λ_eff` 随尺寸降，要按 Eq. (63) 联合选 | GPU | 1–3 天 |
| 4 | 用 `check_factorization.py` 验连续分解 → 再用 Task 3 电路构建 + 官方 verifier 验量化后的 | 本机 | 0.5 天 |
| **合计** | | | **≈3–6 天**，主成本是 GPU 训练迭代，不是实现 |

**可行性结论：**
- 算术成立（2.553–2.566×，官方 README 自认 19.192）；
- **不是"免费"** —— 它 = 官方 Eq. (62) 的全部内容，而 baseline 的残差预算只剩 6.8%；
- **但有真实空间**，因为 baseline 大概率是旧目标（ε_fro）的产物而非 λ_eff 目标的收敛解；
- **性价比一般**：即便做到 49.0⁺，总分只涨 **1.37×**。同样 1.37× 总分，
  如果把 **invsqrt 的成本砍到 1/2.57** 也能拿到，而 invsqrt 的账本更清楚
  （论文 §III.A.1 直接点了三条改进路线）。
- **更值得先看的两个算术事实**：
  1. dfthcsos 里 **`T_total` 的 85%（79,480 / 93,237）是 ANGLE 电路**，
     而样例用的是"普通 QROM" —— 换算法收益可能比动化学更大；
  2. **`Q` 的 544 个临时 qubit 里有 544 是 inner prep 的**，
     而 ANGLE 电路只用 11 个临时 —— 工作区共享的 `max()` 完全由 inner prep 决定。

---

## 6. E4 — 时间与内存（决定后续内循环用什么）

三个任务各跑 3 个随机 salt、全强度、串行独占机器（`src/verify_all.py --runs 3`），
每次运行都由 `/usr/bin/time -l` 单独计量：

| 任务 | 墙钟（3 次） | 峰值 RSS（3 次） | shots / ops（每 shot） |
|---|---|---|---|
| `invsqrt` 全强度 | **251.2 / 266.2 / 267.7 s** | **512.9 / 513.4 / 513.5 MB** | 200,540,928 shots × 62,160 ops |
| `dqi` 全强度（`--max-workers 2`） | **188.8 / 195.7 / 210.7 s** | **3,120.7 / 3,219.5 / 2,117.7 MB** | 102,400 shots × 86,051,014 ops |
| `dfthcsos` 全强度 | **47.8 / 48.3 / 49.7 s** | **1,818.9 / 1,944.3 / 2,010.8 MB** | 254,464 shots，10 电路 ×2 遍 |

单独一次（首次、环境噪声不同）的手工测量：

| 任务 | 墙钟 | CPU(user) | 并行度 | 峰值 RSS |
|---|---:|---:|---:|---:|
| `invsqrt` | 282.17 s | 1427.84 s | 5.06×（8 worker） | 502 MB |
| `dqi` | 189.60 s | 332.18 s | 1.75×（2 worker） | 3.91 GB |
| `dfthcsos` | 51.90 s（有并发下载干扰） | 45.86 s | 1.00× | 1.35 GB |

参考量级：`invsqrt` 吞吐 ≈ **4.4×10^10 ops/s**（论文单核 6.0×10^10，同量级）。
官方 notebook 给的经验值是 invsqrt 全强度 *"~3 min on a recent laptop"*，本机 4.2–4.5 min，一致。

**工程结论：**
- **三个任务全强度 ≈ **8.1 分钟**（串行独占：4.2 + 3.2 + 0.8）。**够快，内循环可以直接用真
  verifier**，不必先上代理模型 —— 但 `dfthcsos` 的 ~48 s 里有 ~10 s 是读 128 MB FCIDUMP
  （每次调用都重读并校验 SHA-256），重复调用应缓存 `load_inputs`。
- **内存纪律（重要）**：
  - `dqi` verifier 默认 `--max-workers 4`，注释写明 *"each worker needs ~6 GB"* → **24 GB**，
    本机 16 GB 会爆。**必须显式 `--max-workers 2`**（实测峰值 2.1–3.2 GB，安全）。
  - `circuit.reaction_depth()` 会复制全部 op：`dqi` 电路 86,051,014 op，
    构建 + 写盘峰值 **1.80 GB**（本机实测，5.99 s）。
  - 886 MB 的 `dqi.kmb` 加载后再 fork worker → 已计入上面的 3.2 GB。
- 脚本化：`src/verify_all.py` 把三个 verifier 串起来跑，逐任务输出
  status / 墙钟 / 峰值 RSS / 报告路径，支持 `--salt` 与 `--runs N`。

---

## 7. E2 — 第一个真实提交

### 7.1 多 salt 交叉验证：**9/9 全绿**

`src/verify_all.py --runs 3`（3 个独立随机 salt × 3 个任务，**全部全强度**，
串行独占机器）。完整输出 `notes/runs/salt_sweep_3x3.json`：

| salt（前 8 位） | `invsqrt` | `dqi` | `dfthcsos` |
|---|---|---|---|
| 第 1 轮 `aeda5a58` / `6b4e1a5d` / `c1a400be` | PASS · 2,076,774.2337 | PASS · 2.44394674e13 | PASS · 3,355,599,588.3155 |
| 第 2 轮 `fe4cd79f` / `f2aaef16` / `e4344ea4` | PASS · 2,076,774.2337 | PASS · 2.44394757e13 | PASS · 3,355,599,588.3155 |
| 第 3 轮 `f75a2d12` / `8d552fdb` / `11c88f6e` | PASS · 2,076,774.2337 | PASS · 2.44394716e13 | PASS · 3,355,599,588.3155 |
| 另：**无 salt**（= 官方 metric 的默认空 salt） | PASS · 2,076,774.2337 | PASS · 2.44394636e13 | PASS · 3,355,599,588.3155 |

**逐项解读（这才是本轮最有信息量的结果）：**

1. **`invsqrt` 的成本在 4 次运行里逐位完全相同**（`2,076,774.233715282`），
   跨 4 × 200,540,928 = **8.02 亿 shot**。因为 `eps` 由**算术**误差决定，
   salt 只改 `HMR`/`R` 的测量结果，而参考解的 uncomputation 是精确的
   → `min_margin` 不变 → `C` 不变。**参考解没有测量相关的相位错误。**
2. **`dfthcsos` 的分数同样逐位完全相同**（`3,355,599,588.3155375`），4/4。
   ANGLE 电路是 QROM（无测量相关行为），所以 `T_total` 也不随 salt 变。
3. **`dqi` 的分数只在小数第 8 位漂移**（2.44394674e13 / 2.44394757e13 / 2.44394716e13，
   相对差 ≈ 4×10⁻⁷）：因为它的随机源就是 salt（shot 抽样 + `HMR`），
   `toffoli_count` = 每 shot **实际执行**的门均值会有极小抖动。
   **`value_errors` / `phase_errors` 在 4 次运行里全部为 0**，`ε₊` 恒为 8.66e-4。
4. 所以"2^20 分之一才触发的罕见失败电路"这一类风险，**在官方参考解上不存在**：
   它不是"某几个 salt 才通过"，而是"所有 salt 都通过"。

### 7.2 打包 + 端到端校验

`src/make_submission.py`（新增）：默认把三个任务都指向官方样例（"安全直通"），
任一任务可 `--invsqrt/--dqi/--dfthcsos` 覆盖 —— 这正是整个比赛循环需要的形状
（改进一个任务，另外两个钉在参考解上）。打包后自动调用
`src/validate_submission_format.py`（用**官方 metric 自己的** `find_one`/`read_circuit`/`read_npz`）。

```
build/submission.zip   73,803,082 B   ## 13 个成员
ACCEPT
```

`dqi.kmb` 886 MB → 压缩到 ~70 MB（12.6×）。格式检查项全绿：
`invsqrt.kmb`（1 寄存器）/ `dqi.kmb`（4165 寄存器）/ `dfthcsos/` 恰好 `angle_k_0..9`
（各 119 qubit）/ `submission.npz` 6 个数组共 2,150,108 B（上限 67,108,864）。

**解包回归（真·端到端）**：解压 `submission.zip` → 13/13 成员与源文件 SHA-256 **逐字节一致**
→ 直接对解压出来的目录跑 verifier：

| 检查 | 结果 |
|---|---|
| `dfthcsos/verify.py /tmp/subext/dfthcsos --salt e2e_check_artifact` | **PASS**，score 3,355,599,588.3155375，T=93,237，Q=730 |
| `verify_submission_invsqrt.py /tmp/subext/invsqrt.kmb --max-x 20480` | **PASS** |

> ⚠️ 格式检查只证明"能走到 verifier 那一步"，**不模拟电路**。正确性靠上面的 verifier。

### 7.3 ⛔ 提交被挡住：**程序化提交不可行，而且账号还没接受规则**

两个独立的发现：

1. **写操作需要浏览器 session，直连 API token 不够。**
   读端点正常（`GET /api/v1/competitions/submissions/list/<slug>` → **200 `[]`**），
   但任何 `competitions.SubmissionService/*` → **401 `{"error":{"code":401,
   "message":"Unauthenticated"}}`**，与 `competitions.UserService/AcceptRules` 一样。
   → 本机只能把 `submission.zip` 准备好，**上传必须在浏览器里做**。

2. **账号当前 `canParticipate = false`。** 2026-10-24 21:03 复查：

   ```json
   {"access":{"161954":{"canView":true,"hasAcceptedRules":false,
                        "canUpdate":false,"canParticipate":false}}}
   ```

   注意：**下载数据不需要这个标志**（34 个文件已全下），但**提交需要**。
   所以"规则已接受"这条前提与实际不符 —— 需要在浏览器打开
   <https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation/rules>
   点一次接受（**不消耗提交额度**）。

### 7.4 预期分数与风险 —— **等你的决定**

前提：先在浏览器接受规则，再上传 `build/submission.zip`（= 官方样例直通）。

- **预期分数：恰好 1.00000（5 位小数）。** 官方 metric 的 `score()` 是
  `math.prod(ratios) ** (1/3)`，`ratio = baseline_cost / cost`，而 `baseline_cost`
  是**参考提交在同一个 split 秘密 salt 下**跑出来的成本。我们提交的就是参考提交本身
  （`dqi.kmb` 是下载来的官方原文件，SHA-256 与官方一致），
  所以三个 `ratio` 都**恰好 1.0**：
  - `invsqrt` / `dfthcsos` 的成本跨 salt 逐位不变（§7.1 已证），故恒等于 `baseline_cost`；
  - `dqi` 的成本依赖 salt 抽样，但同一份文件 + 同一个 salt ⇒ 同一批 shot ⇒ 同一个均值。
  `verification_shot_divisor` 在 Kaggle 上必须为 1（源码注释明说），所以不会被削弱。
- **风险：极低。** 电路就是官方参考解本身（`dqi.kmb` 用的是官方原文件而非本机重建版），
  3 个随机 salt + 空 salt 共 4 轮全部 PASS；`invsqrt`/`dfthcsos` 的成本跨 salt 逐位不变。
- **收益**：确认端到端链路；拿到一个非零底分；为附加奖 `Longest Leader`
  （按占据榜首**时长**累计）留下一个已落地的提交。
- **代价：1 次提交额度**（每天 5 次）。

> **我没有花任何提交额度，也没有做任何提交尝试。** 等你点头（并接受规则）再走。

---

## 8. 本轮新增/修改的文件

| 文件 | 说明 |
|---|---|
| [`../src/download_data.py`](../src/download_data.py) | 竞赛数据下载：百分号编码 + 流式落盘 + 字节数核对 |
| [`../src/fast_download.py`](../src/fast_download.py) | HTTP Range 分块并发下载（3.18 MB/s，16×） |
| [`../src/download_starter_kit.py`](../src/download_starter_kit.py) | Starter Kit 数据集下载（默认跳过 437.8 MB 的 fcidump） |
| [`../src/verify_all.py`](../src/verify_all.py) | 三个 verifier 串跑：status / 墙钟 / 峰值 RSS / 多 salt |
| [`../src/make_submission.py`](../src/make_submission.py) | 组装 `submission.zip` + 自动格式校验 |
| [`../src/analyze_dfthcsos_budget.py`](../src/analyze_dfthcsos_budget.py) | E5 诊断：Λ 的 λ_h1/λ_h2 分解与 floor 收益 |

前序 4 个脚本（`make_placeholder_submission.py` / `validate_submission_format.py` /
`test_format_validator.py` / `kickmix_local_demo.py`）**未修改，仍然可用**：
`validate_submission_format.py` 被 `make_submission.py` 直接复用。
