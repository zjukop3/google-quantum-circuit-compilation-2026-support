# E5｜invsqrt `flags`：一元温度计 → 二元 `s`（Q 237→231，T 7334→6852）

> 任务：把 invsqrt 的 `flags` 从一元编码改成更紧凑的编码。
> 结论：**GO。改完不是"−6 qubit 换 +80 Toffoli"，而是 Q 与 T 同时下降**：
> `Q 237 → 231`（−6）、`T 7334 → 6852`（**−482**）、`C 7590.127 → 7108.127`、`score 1,798,860.14 → 1,641,977.38`
> （gain **1.26480**，比 E4 再 +9.56%，比已提交的 E3 再 +10.38%）。
> 小 n_p 全枚举（n_p=2..6，全部 x）等价性 PASS；官方验证器三盐全量 **PASS 且逐位一致**。**未花任何提交额度。**

---

## 0. 结论先给

| 电路 | Q | T | C = toffoli_equivalent | score = Q×C | gain vs reference | 相对 E4 |
|---|---:|---:|---:|---:|---:|---:|
| 主办方 reference | 260 | 7,735 | 7,987.59 | 2,076,774.23 | 1.00000 | — |
| E3（已提交 id 57612302） | 239 | 7,327 | 7,583.13 | 1,812,367.40 | 1.14589 | — |
| E4（`build/e4/invsqrt_opt3.kmb`） | 237 | 7,334 | 7,590.13 | 1,798,860.14 | 1.15449 | 1.00000 |
| **E5（本文件，三盐实测）** | **231** | **6,852** | **7,108.127186961416** | **1,641,977.380188087** | **1.26480** | **1.09555** |

- 净变化：**−6 常驻 qubit，−482 Toffoli**（交换率为负 —— 这次不是"买 qubit"，是顺手一起赚）
- **E4 对这条路的估算（Q −6 / T +80 / gain 1.17）是错的**，错在两处，见 §4
- 三处控制逻辑里其实只有**两处**碰 `flags`（`_select_interval` 根本不用它，见 §1.3）
- 产出：`src/e5/invsqrt_flags_circuit.py`（新构造器）、`src/e5/e5_flags_check.py`（全枚举等价性）、
  `build/e5/invsqrt_flags.kmb`、`build/submission_e5.zip`（格式校验 ACCEPT，**未提交**）

---

## 1. `flags` 到底是什么、用在哪

### 1.1 语义：它是 `s` 的温度计，不是 one-hot

`flags[k] = [x 的高位第 k+1 个 pair 全为 0]`，等价地

```
flags[k] = [s ≥ k+1],        s = (x 的前导零 pair 数) ∈ [0, n_p] = [0, 10]
s = Σ_k flags[k]              （所以是 1^s 0^(n_p−s) 的"温度计"，不是 one-hot）
```

`_flag_zero_pair` 每次只 AND 当前**最高 pair**（先 X 再 `init_and` 再 X），因为归一化是**边移边算**的：
第 k 步的寄存器最高 pair 恰好就是原来的第 `n_p − k` 个 pair，而"前缀全零"这个信息由前一步的移位
自动带进来（若前面某一步 flag=0，寄存器不再移，最高 pair 不变，后续 flag 自然全 0）。
所以 10 个前缀 AND 只要 10 个 Toffoli —— 这是参考实现里相当精巧的一处。

### 1.2 两处消费点（第三处不存在）

| 位置 | `flags` 的用法 | E4 电路实测代价 |
|---|---|---:|
| `_normalize` | 每步 `flags[k]` 同时控制 **两条 lane**（偶/奇）的窗口旋转 `left_rotate(x[2k::2], 1, control=flags[k])`；反向前向共 20 次/趟 | **230 T**（两趟） |
| `_denormalize` | `y1[k : k+result_bits+2]`（长度 30）上 1 次受控旋转 / flag，共 10 次/趟 | **580 T**（两趟） |
| `_select_interval` / `_select_interval2` | **完全不用 flags**：区间由 `x_norm` 的最高 3 位直接算（`t_a=!top&third`, `t_b=top&second`），`half` 在 E4 里也改成 Horner 每步现算 | 2 T（与 flags 无关） |

> 任务书里说"三处最精巧的控制逻辑"，实际是**两处**。少一处 = 少一份风险。

### 1.3 为什么一元编码这么贵

kickmix 的 `left_rotate(target, shift=1, control=q)` 是一次受控循环移位，价签实测 **`len(target) − 1` 个 CSWAP**（`bit_rotate_test.py::test_controlled_left_rotate_cost`，本机复测一致）：

```
_normalize   : Σ_k [(11−k)−1]×2 = 110 /趟  + 10 个 init_and      = 115/趟 → 两趟 230  ✔实测
_denormalize : 10 次 × (30−1) = 290 /趟                          = 290/趟 → 两趟 580  ✔实测
```

**一元编码的 10 个常驻 qubit 本身不花钱，贵的是它强迫你"一个 flag 一次旋转"**：20 + 20 次窗口旋转，
每次都要付几乎整条 lane 的 CSWAP。

---

## 2. 二元 `s` 怎么做（而不是"怎么把温度计塞进 4 位"）

### 2.1 核心事实一：二元移位寄存器是**桶形移位器**，比逐次旋转便宜

kickmix 有 `left_rotate(target, shift=<比特数组>)`，实测价签（`e5_probe.py`）：

| 调用 | 实测 T | 公式 |
|---|---:|---|
| `left_rotate(22, shift=4b)` | 51 | `(n//2)(b+1) − b` |
| `left_rotate(22, shift=[False]+s)`（= 2s，5 个权重位但只有 4 个量子比特） | **50** | 常量位不收费 |
| `left_rotate(39, shift=4b)` | **95** | 同上 |
| `left_rotate(30, 1, control=q)`（一元版的一次） | 29 | `n−1` |

于是：`x ← x·2^(2s)` 一次 **50 T**（替掉 110 T 的逐次窗口移位）、`y1 ← y1·2^s` 一次 **95 T**（替掉 290 T）。
两趟合计 100 + 190 = **290 T**，替掉原来的 810 T。

**语义已实测确认**（`e5_shift_semantics.py`）：`left_rotate` 是**循环**旋转，`[False]+list(s)` 给出的正是 `x << 2s`。循环旋转不会丢位当且仅当"绕回来的那些位是 0"：

- `x`：绕回来的是最高 `2s` 位，而 `s` 就是"前导零 pair 数"，`2s ≤ 前导零位数` → 全是 0 ✔
- `y1`：`p` 住在 `y1[:29]`，最高 `s` 位本来就是 0（相位定标留下的空位）→ 绕回来的是 0 ✔

桶形移位器的**中间态**不需要满足这条（循环旋转的复合仍是循环旋转，只看最终总量），所以不存在"中间步骤把数据卷丢"的问题。

### 2.2 核心事实二：温度计 → 二元**只用 CX（0 Toffoli）**，而且是精确的

`s = Σ_k flags[k]`，而 `flags[k] = [s ≥ k+1]`，所以 `s` 的第 j 位正好是若干 flag 的 XOR：

```
bit_j(s) = XOR over t of flags[t·2^(j+1) + 2^j − 1]      （下标 < n_p）
```

n_p=10 时展开就是 `b0 = f0⊕…⊕f9`、`b1 = f1⊕f3⊕f5⊕f7⊕f9`、`b2 = f3⊕f7`、`b3 = f7`
（`e5_flags_check.py::check_thermometer_terms` 对 n_p=2..32 的**全部** s 穷举验证过，全部精确）。
CX 是自逆的，所以反向前向走同一段代码即可**双向复用**（前向算 s、反向清零 s）。

### 2.3 温度计只需要**瞬时**存在

这是让 Q 也降下来的关键：10 个 flag 不再常驻，只在 `_normalize` 内部活着，
算完温度计 → CX 折进 `s` → 立刻 `del_and` 清掉（AND 形状的垃圾是**零相位免费**回收的，见 E4 §5.3）→
再 `free` 归还 ancilla 池。常驻只剩 4 位 `s`。

### 2.4 新的 `_normalize` / `_denormalize`

```python
_normalize_bin(forward):
    flags/aux = 临时 ancilla（10 + 9 = 19 个，用完全部归还）
    for k in 0..n_p-1:                        # 19 T
        a = ¬x[2j] & ¬x[2j+1]  (j = n_p−k)    # 注意：见 §3.2 的坑
        flags[k] = flags[k−1] & a             # 前缀链
    s ^= decode(flags)                        # CX only, 0 T
    del_and(全部 flags/aux)                   # 0 T（测量式解算，AND 形状）
    left_rotate(x, shift=[False]+list(s))     # 50 T，一次搞定 x·2^(2s)

_normalize_bin(inverse):                      # 严格镜像
    right_rotate(x, shift=[False]+list(s))    # 50 T
    重建温度计（19 T）→ s ^= decode（CX，自动把 s 清回 0）→ del_and 清温度计

_denormalize_bin:
    forward: left_rotate(y1, shift=s)         # 95 T
    inverse: right_rotate(y1, shift=s)        # 95 T
```

算术逐位不变：`x_norm`、区间选择、`y1 = p·2^s`、相位 `y1/2^F` 全部与 E4 相同 → 精度模型一个字都没动。

---

## 3. 实施与实测

### 3.1 Q 账本

```
persistent  204 = x 22 + y1 39 + s 4 + selectors 2 + 3×coef 17 + square 28 + z 29 + scratch 29
                  （E4 是 210：flags 10 + selectors 2 → 这里 flags 10 → s 4，−6）
transient    27 = iadd/isub ancilla 峰值（E4 已定位在 _correct/_multiply 那几次，未变）
                  normalize 内部新增 19 个瞬时 ancilla（10 flags + 9 aux）< 27，**不触新峰值**
Q = 204 + 27 = 231                                                              ← 实测 231 ✔
```

实测旁证：`e5_flags_measure.py` 把 `_normalize_bin` 整个 stub 掉，Q 仍是 231 —— 说明峰值不在 normalize，
而是那 27 个加法器 ancilla。`s` 只有 4 位也是下界（`s ∈ [0,10]`）。

### 3.2 实施中真实踩到的坑（值得记下来）

**`init_and` 是"按 target 广播"的，不是多控 AND。** 第一版我写成
`init_and(flags[k−1], km.array([x[2j], x[2j+1]]), flags[k])`，以为那是"3 控 AND"（实测确实收 2 个 CCX）。
但 `broadcast_init_and_obj`（`py/circuit/ops/ccx.pybind.cc`）是**逐 target** 配对 `(c1[k], c2[k], target[k])`，
标量 target 被广播成长度 2 → 两个积 XOR 进**同一个** qubit，得到的是 `flag & (¬b0 ⊕ ¬b1)`。
症状：n_p=2 时 `x=4..7` 的 `s` 从 1 变成 2（`e5_debug.py` 逐寄存器对比抓到的），全枚举立刻 FAIL。
修法：把 3 控拆成两个 2 控 AND，多花一个 ancilla，Toffoli 数不变（仍是 19/趟）。

> 教训：kickmix 里凡是"数组参数"的元件（`init_and`/`cx`/`left_rotate(shift=…)`）都要先确认
> 到底是**广播**还是**多控**。`cx(array, scalar)` 是广播（= XOR 奇偶，0 T），这一点我用
> `_thermometer_xor_into_s` 的"stage saves 0 T"和 s 的逐位正确性双向确认过。

### 3.3 分阶段实测（n_p=10，精度 12/17/28/27/28）

| 阶段 | E4 T | E5 T | 差 |
|---|---:|---:|---:|
| `_normalize`（两趟） | 230 | **138** | −92 |
| ├ 温度计链（两趟） | — | 38 | 19/趟（设计值） |
| ├ CX 折编码（两趟） | — | 0 | 0 |
| └ 桶形移位 `x·2^(2s)`（两趟） | — | 100 | 50/趟 |
| `_denormalize`（两趟） | 580 | **190** | −390 |
| `_select_interval2` | 2 | 2 | 0 |
| `_correct_slice` | 56 | 56 | 0 |
| Horner / square / multiply_m / multiply_y0 | 6,466 | 6,466 | 0 |
| **合计** | **7,334** | **6,852** | **−482** |

---

## 4. 为什么 E4 的估算（+80 T、gain 1.17）错了

E4 §6 的模型是："10 个 flag 退化成临时 AND，只留 4 位 s；反向 pass 里把 s 还原成 10 个阈值比较
（每趟约 40 Toffoli，两趟 ~80）"。两处偏差：

1. **漏算了桶形移位器的红利。** E4 以为"20/20 次 lane 旋转变成 4 次按 2^j 的受控旋转（旋转是 Clifford，
   不花钱）"——**旋转不是 Clifford**：`left_rotate(target, shift=1, control=q)` 收 `len−1` 个 CSWAP，
   `left_rotate(target, shift=bits)` 收 `(n//2)(b+1)` 个。恰恰是"改成二元"这一步把 810 T 的窗口旋转压到 290 T。
2. **高估了解码成本。** 温度计→二元不是"每趟 40 T 的阈值比较"，而是**纯 CX（0 T）**，因为 `flags[k]` 本身就是
   阈值 `[s≥k+1]`，XOR 组合直接给出二进制位。真正要付的只有温度计链本身 19 T/趟。

净效果：`ΔQ = −6`，`ΔT = −482` ⇒ **交换率为负**，完全不存在"超预算"的问题。

---

## 5. 验证

### 5.1 小 n_p 全枚举等价性（`src/e5/e5_flags_check.py`，枚举每个 n_p 的全部 x）

对照物：`fixed_point_model`（y1 逐位）+ 官方 reference 电路（相位 mod 1 圈）+ x 还原 + 全部 scratch（含温度计）归零。

| n_p | Q | T | Q_e4 | T_e4 | y1==model | phase==ref (mod 1 turn) | x 还原 | 脏 qubit |
|---:|---:|---:|---:|---:|---|---|---|---|
| 2 | 205 | 4,096 | 205 | 4,118 | ✅ | ✅ | ✅ | 0 |
| 3 | 208 | 4,636 | 209 | 4,717 | ✅ | ✅ | ✅ | 0 |
| 4 | 212 | 5,186 | 213 | 5,288 | ✅ | ✅ | ✅ | 0 |
| 5 | 215 | 5,664 | 217 | 5,831 | ✅ | ✅ | ✅ | 0 |
| 6 | 218 | 6,114 | 221 | 6,346 | ✅ | ✅ | ✅ | 0 |

（相位比较单位已按 E4 的教训用"圈"：`((g−r)/2).denominator == 1` 才判全局相位。）
另外 `thermometer→binary` 解码对 n_p=2..32 的**全部** s 穷举精确。

### 5.2 官方验证器全量三盐

`verify_submission_invsqrt.py build/e5/invsqrt_flags.kmb --salt <salt>`（每次 200,540,928 shots，
覆盖面 = 域内**全部** x ∈ [1, 3(2^10−2)²]，每个 x 64 shots）：

```json
// notes/runs/e5_flags_saltA.json
{
  "status": "PASS",
  "score": 1641977.380188087,
  "toffoli_equivalent": 7108.127186961416,
  "toffoli_count": 6852.0,
  "max_toffoli_count": 6852,
  "t_count": 1.0,
  "rotation_count": 36.0,
  "rotation_eps": 3.6809236372811597e-13,
  "num_qubits": 231,
  "reaction_depth": 3737,
  "num_ops": 58403,
  "verified_shots": 200540928,
  "worst_error_over_tolerance": 0.9251076979764682,
  "worst_x": 1119994,
  "circuit_sha256": "0687a7b203d7ff042d07679f5b2a7164342a92cd0e70511450d2b3cf9852f8fd"
}
// notes/runs/e5_flags_saltB.json
{
  "status": "PASS",
  "score": 1641977.380188087,
  "toffoli_equivalent": 7108.127186961416,
  "toffoli_count": 6852.0,
  "max_toffoli_count": 6852,
  "t_count": 1.0,
  "rotation_count": 36.0,
  "rotation_eps": 3.6809236372811597e-13,
  "num_qubits": 231,
  "reaction_depth": 3737,
  "num_ops": 58403,
  "verified_shots": 200540928,
  "worst_error_over_tolerance": 0.9251076979764682,
  "worst_x": 1119994,
  "circuit_sha256": "0687a7b203d7ff042d07679f5b2a7164342a92cd0e70511450d2b3cf9852f8fd"
}
// notes/runs/e5_flags_saltC.json
{
  "status": "PASS",
  "score": 1641977.380188087,
  "toffoli_equivalent": 7108.127186961416,
  "toffoli_count": 6852.0,
  "max_toffoli_count": 6852,
  "t_count": 1.0,
  "rotation_count": 36.0,
  "rotation_eps": 3.6809236372811597e-13,
  "num_qubits": 231,
  "reaction_depth": 3737,
  "num_ops": 58403,
  "verified_shots": 200540928,
  "worst_error_over_tolerance": 0.9251076979764682,
  "worst_x": 1119994,
  "circuit_sha256": "0687a7b203d7ff042d07679f5b2a7164342a92cd0e70511450d2b3cf9852f8fd"
}
```

> 三个盐的 JSON **逐字节相同**（含 `circuit_sha256`）→ 不是运气。
> 更关键：`worst_error_over_tolerance = 0.9251076979764682` 与 `worst_x = 1119994` 与 E4 **完全相同**，
> `rotation_eps` 也完全相同（3.6809236372811597e-13）→ 精度与旋转合成预算一分未动。

### 5.3 精度

`worst_error_over_tolerance` 与 E4 **逐位相同**（同一套 `12/17/28/27/28`、逐位相同的 y1）→ **≤ 1.0 且未动**。
原因：这次改的是"怎么算 s、怎么移位"，不是"用什么精度算"；`fixed_point_model` 描述的电路一字未改。

### 5.4 提交包

```
build/submission_e5.zip  73,801,610 B
  invsqrt.kmb  Q=231 T=6,852 ops=58,403, sha256 0687a7b2…52f8fd
  （其余成员与 E4 包逐字节相同；包内 invsqrt.kmb 的 sha256 与三盐验证的 circuit_sha256 一致 ✔）
src/validate_submission_format.py build/submission_e5.zip → ACCEPT
```
构造器是确定性的：重跑一次产出的 `.kmb` sha256 与已验证产物逐位相同。
**本轮 0/5 提交额度**（纯离线）。

---

## 6. 建议配置与风险

**建议：提交。** 这是当前唯一一个"Q 与 T 同时下降"的改动，且精度模型完全没碰。

| 项 | 结论 |
|---|---|
| gain（实测） | **1.26480**（相对 reference），比 E4 的 1.15449 再 **+9.56%**，比已提交的 E3 再 +10.38% |
| score（实测） | **1,641,977.380188087** = 231 × 7108.127186961416（三盐逐位一致） |
| Q / T | 231 / 6,852（`toffoli_count` 验证器实测 6852 ✔） |
| 精度 | `worst_error_over_tolerance` 不变（≈0.9251），余量 7.5% 未动 |

风险清单（都已正面验证，但如实列出）：

1. **桶形移位的"零回卷"前提**是结构性的，不是统计性的：`x` 靠 `s` 的定义、`y1` 靠 `p < 2^29`。
   反证条件一旦不成立会立刻表现为全枚举 FAIL（不是偶发）。n_p=2..6 全枚举 + n_p=10 全量官方验证器都过。
2. **`del_and` 的相位中性**依赖"target 恰好等于两个控制位的 AND"。温度计的每一个 flag / aux 都是
   标准 AND 形状，且清理时用的是**未改动**的 x 位（X 共轭后立即还原），所以两条 `del_and` 都是零相位。
   相位一致性的验证是全枚举 + 官方全量，属于最强证据。
3. **shot 相关性**：本次改动没有引入任何新的 `R`/`HMR` 用法（温度计清理复用了 `del_and`，与
   `_add_row` 里已经在用的机制同类），所以不会引入 E4 §5.3 那种"shot 相关随机符号"风险。
4. **桶形移位在 `s=0`** 时仍执行全部 CSWAP（恒等），只是浪费 T（已计入 50/95 的实测值），无正确性影响。
5. **残留的瞬时 ancilla 数**（19）离 27 的峰值还有余量；若将来把峰值压到 <19，需要重新核算 Q。

**为什么值得提交而不是再等等**：这条路 E4 判定为"唯一剩下的结构性机会"，现在做完了，
而且它不是"花 T 换 Q"而是双赢；再往下压 Q 的候选（`square` 反向重算 32 T/qubit、Horner 降寄存器 42 T/qubit 不等价）
都已被证伪或临界无收益。

---

## 7. 复现命令

```bash
cd competitions/2611-google-quantum
PY=../2609-xingzhi-drug/.venv/bin/python

# 0) 元件价签标定（单元件成本表 + 桶形移位语义）
$PY src/e5/e5_probe.py
$PY src/e5/e5_shift_semantics.py

# 1) 构建产物
$PY src/e5/invsqrt_flags_circuit.py build/e5/invsqrt_flags.kmb \
    --delta-bits 12 --coef-bits 17 --square-bits 28 --product-bits 27 --result-bits 28

# 2) 全枚举等价性（几秒）
$PY src/e5/e5_flags_check.py --n-p 2 3 4 5 6

# 3) 分阶段账本（E4 baseline vs E5）
$PY src/e5/e5_flags_measure.py
$PY src/e5/e5_measure.py            # E4 baseline 的阶段消融

# 4) 逐寄存器调试钩子（y1/s/x_norm vs 模型）
$PY src/e5/e5_debug.py --n-p 3 --limit 24

# 5) 官方验证器（200,540,928 shots/盐；全核时约 4m42s/盐，三盐并行 2 worker 时约 19 min/盐）
for s in saltA saltB saltC; do
  $PY data/verifiers/invsqrt/verify_submission_invsqrt.py build/e5/invsqrt_flags.kmb --salt $s
done

# 6) 打包 + 格式校验（不花额度）
$PY src/make_submission.py --invsqrt build/e5/invsqrt_flags.kmb --out build/submission_e5.zip --no-check
$PY src/validate_submission_format.py build/submission_e5.zip
```

---

## 8. 一句话

**`flags` 从 10 位温度计换成 4 位二元 `s`，不是"−6 qubit 换 +80 Toffoli"，而是 −6 qubit 的同时把 T 也砍掉 482**
（`_normalize` 810→290 的窗口旋转被两次桶形移位替掉，温度计只在 `_normalize` 内部瞬时存在、
且用纯 CX 折成二元）。Q=231 / T=6,852，gain **1.26480**（三盐逐位一致），精度模型一字未动 —— **建议提交**。

---

# 附录 A｜E5 提交结果（已提交、已出分）

**状态：COMPLETE。** 本文件前面全部是提交前的构建/验证；这一节记录真实提交与竞争结果。
本轮是 E3→E4→E5 三连的第三棒，`dqi.kmb` 与 `dfthcsos/*` 与上次提交**逐字节相同**，
所以整体分必须**只**随 invsqrt 这一个成员的比例变化 —— 这就是本轮实验设计本身。

## A1 实际提交了什么

| 字段 | 值 |
| --- | --- |
| 包 | `build/submission.zip`（= `build/submission_e5.zip` 的字节副本，73,801,610 B） |
| sha256 | `18029c35c9eccebbda4eeb19b16c9fbbf0c643b9138ebf0d0ab57b180e2eee0f` |
| 成员 | `invsqrt.kmb`（541,511 B）· `dqi.kmb` · `dfthcsos/*`（11 文件 + `submission.npz`） |
| `invsqrt.kmb` | Q=**231**、T=**6852** |
| `circuit_sha256` | `0687a7b203d7ff042d07679f5b2a7164342a92cd0e705114d2b3cf9852f8fd` |
| description | `E5 invsqrt flags: binary s (4b) replaces 10b thermometer, Q=231 T=6852` |

### A1.1 提交前复核（本地，未花额度）

| 检查 | 结果 |
| --- | --- |
| `build/submission.zip` 是否已被 E5 覆盖 | **是** —— 覆盖前它是 E4 字节（`20b7df73…`），覆盖后与 e5 逐字节相同（`cmp` 通过） |
| `unzip -t` 完整性 | `No errors detected` |
| 从 zip 里解出的 `invsqrt.kmb` sha256 | `0687a7b2…2f8fd` —— 与设计一致 |
| `dqi.kmb` / `dfthcsos/angle_k_{0,4,9}.kmb` / `submission.npz` vs E4 包 | **逐字节相同**（`cmp`） |
| 三盐官方验证器 | **PASS ×3**，JSON **逐字节相同**（sha `d7eb3056…`） |
| `worst_error_over_tolerance` | `0.9251076979764682`（与 E4 同值） |
| 小 n_p 全枚举等价性（n_p=2..6） | `all_ok: true`（`notes/runs/e5_flags_check.json`） |

## A2 提交机制（实际发生了什么）

| 步骤 | 结果 |
| --- | --- |
| `cp build/submission_e5.zip build/submission.zip` | 完成（名字必须是 `submission.zip`，否则 CreateSubmission 400） |
| `--upload-only` 免费握手 | **OK** —— StartBlobUpload + PUT，PUT 回显 `sha256=18029c35…` |
| CreateSubmission（复用 blob token） | **OK** |
| **submission id** | **57618221** |
| teamId | 17194714 |
| `dateSubmitted` | `2026-10-25T05:32:54.983Z`（13:32:54 +08:00） |
| 出分时刻 | 14:07:23 → **约 34.5 分钟**（E4 ~44 min、E3 ~39 min） |

日志：`notes/runs/e5_uploadonly.txt`（含 blob token）、`e5_createsubmission.txt`、`e5_watch.txt`（轮询轨迹，以 `SETTLED` 结束）。

## A3 结果

| | 值 |
| --- | --- |
| **public LB 分数** | **1.08145**（E4 1.04905 → E3 1.04644） |
| **排名** | **678 / 806**（E4 时 681 / 801） |
| 剩余额度 | `{"numToday": 3, "numTotal": 4, "numAllowedNow": 2}` |
| 本轮花掉额度 | **1**（`--upload-only` 握手仍为 0） |

排名邻域（`notes/runs/leaderboard_e5.json`）：

```
rank  675  1.09357
rank  676  1.08972
rank  677  1.08258
rank  678  1.08145   <== 17194714（我们，submission 57618221）
rank  679  1.07752
rank  680  1.06809
rank  681  1.05676
```

榜单同期从 801 → **806** 行。净效果：分数 **+0.03240**、排名 **+3**（681 → 678）。

## A4 预测 vs 实际（偏离分析）

| 量 | 预测 | 实际 | 偏离 |
| --- | --- | --- | --- |
| invsqrt 比例 `2,076,774.23 / 1,641,977.380188087` | 1.264800755 | — | — |
| 整体 `ratio^(1/3)`（**正确算术**） | **1.081452309** | **1.08145** | **−0.000002** |
| 任务书口径 | 1.08163 | 1.08145 | −0.00018 |

**结论：偏离 0.00018，比 0.002 的容差小 11 倍 —— 不触发根因排查；但原因已定死，不是模型误差：**

任务书里的 `1.08163` 是一次**算术滑步**。`1.08163^3 = 1.265424`，而真实比例是 `1.264800755`——
要让立方根等于 1.08163，比例得是 1.265424（差 +0.049%）。直接用给定输入算：
`1.264800755^(1/3) = 1.081452309`，五位小数即 **1.08145**，与出分**完全一致**。

**打分模型第三次被精确确认**（E3、E4、E5 三点全部落在 5 位显示精度内）：

- E3：`(1.145898…)^(1/3) = 1.046_44` → 出分 1.04644
- E4：`(1.154494549)^(1/3) = 1.049_053` → 出分 1.04905（dev −0.000003）
- E5：`(1.264800755)^(1/3) = 1.081_452` → 出分 1.08145（dev −0.000002）

反推校验：把印出来的 `1.08145` 立方回去得 1.264792655，与验证器比值 1.264800755 相对差 **−6.4e-06**，
正好落在五位小数打印量化带 `[1.2647751, 1.2648102]` 之内。**没有残差需要解释。**

同时也再次排除"其余两个成员也贡献了分数"这一可能：若它们真的动了，要补上 +0.00018 的缺口，
它们的比例得偏离 1 到 1.000493（整体因子 1.000164）—— 而它们与已出分的 E4 包逐字节相同，不可能偏离。

## A5 这一轮买到了什么

- E5 相对 E4 的贡献：**`flags` 一元温度计 → 二元 `s`**，`Q 237→231`（−6）、`T 7334→6852`（−482），
  即**不是**"花 T 换 Q"而是两者双赢；invsqrt 比例 1.15449 → 1.26480（+9.56%），整体 +0.03240。
- 出分经验：排名只从 681 挪到 678 —— 我们所在band很稠（677 行只比我们高 0.00113），
  **分数是硬收益，排名得靠活榜单看**（A2/A3 期间榜单自身长了 5 行）。
- 剩余额度 2 次（今天），`numTotal` 已到 4。**不要把 Final 槽花在 invsqrt 的边际上**：
  同样的绝对 qubit 节省，在 `dqi`（成本 2.4e13）上值钱得多，invsqrt 只有 1.6e6。

## A6 文件

| 路径 | 是什么 |
| --- | --- |
| `build/submission.zip` | 实际提交的归档（73.8 MB，含 E5 invsqrt） |
| `build/submission_e5.zip` | 同一份字节，按实验名留档 |
| `src/e5/poll_e5.py` | 按 submission id 轮询至落定的脚本 |
| `src/e5/project_rank.py` | 给定分数在快照榜单上的落位推算 |
| `src/e5/e5_deviation.py` | A4 的偏离分析（输出存 `notes/runs/e5_deviation.txt`） |
| `notes/runs/e5_uploadonly.txt` · `e5_createsubmission.txt` · `e5_watch.txt` | 握手 / CreateSubmission / 轮询轨迹 |
| `notes/runs/leaderboard_e5pre.json` · `leaderboard_e5mid.json` · `leaderboard_e5.json` | 榜前 / 榜中 / 榜后快照 |
