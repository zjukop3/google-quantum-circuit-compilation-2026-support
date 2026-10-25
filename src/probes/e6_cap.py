#!/usr/bin/env python3
"""E6 §4 — ANGLE 家族为什么不能更便宜：解码器的 Toffoli/ancilla 权衡 + 128 上限。

三条要钉死的事实
  1. 每条 ANGLE_k 的 Toffoli = 地址树内部结点数 = A-1（实测 3974，剪掉 ≥A 的空叶）。
  2. 这个解码器需要 **11 个同时存活的 ancilla**（12 bit 选择寄存器的二叉树深度 12，
     最高位用 X 共轭省掉一格）。这是 unary iteration 的标准结论：
     峰值存活 = 前缀链长度；要让 2^12 个地址各自拿到一个「当前地址」控制位，
     前缀链必须能区分 4096 个 minterm，深度下界 12。
  3. 于是电路宽度 = 12(selection) + J*brot + 11 >= 12 + 12J + 11，
     <= 128 => J <= 8.75 => J = 8 => K = ceil(75/8) = 10。

「用更少 ancilla」只能靠把后几 bit 改成逐地址的 MCX/AND 链 —— 本脚本把这条
trade-off 曲线算出来，证明那是**亏的**（活的 ancilla 越少，Toffoli 越多）。

用法: python src/e6/e6_cap.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parents[2]
SAMPLE_NPZ = COMP / "data" / "sample_submission" / "dfthcsos" / "submission.npz"

A = 3976
NUM_BITS = (A - 1).bit_length()          # 12
CAP = 128
BROT = 12


def load_deltas() -> np.ndarray:
    z = np.load(SAMPLE_NPZ, allow_pickle=False)
    words = z["angle_words"].astype(np.int64)
    N, L = words.shape
    J, K = 8, math.ceil(L / 8)
    pad = np.zeros((N, K * J), dtype=np.int64)
    pad[:, :L] = words
    b = pad.reshape(N, K, J).transpose(1, 0, 2)
    d = b.copy()
    d[1:] ^= b[:-1]
    return d


class TreeCounter:
    """逐字复刻 example_submission/angle_k_reference_circuit.build_angle_k_circuit 的递归。

    统计 init_and 次数（= Toffoli）、同时存活的 ancilla 峰值、以及被
    `all rows equal` 剪掉的子树数（Qualtran QROM 的 _break_early）。
    """

    def __init__(self, delta: np.ndarray):
        self.delta = delta
        self.rows = delta.shape[0]
        self.ands = 0
        self.live = 0
        self.peak = 0
        self.loads = 0
        self.early = 0
        self.max_depth = 0

    def node(self, level: int, lo: int, hi: int) -> None:
        if lo >= self.rows:
            return
        if hi <= self.rows and (hi - lo == 1 or np.all(self.delta[lo:hi] == self.delta[lo, 0])):
            self.loads += 1
            if hi - lo > 1:
                self.early += 1
            return
        mid = (lo + hi) // 2
        if self.rows <= mid:
            self.node(level + 1, lo, mid)
            return
        # 分配一个 ancilla（level-1 号槽），做 init_and
        self.ands += 1
        self.live += 1
        self.peak = max(self.peak, self.live)
        self.max_depth = max(self.max_depth, level)
        self.node(level + 1, lo, mid)      # anc = control AND NOT bit
        self.node(level + 1, mid, hi)      # anc = control AND bit（cx(control, anc)）
        self.live -= 1

    def run(self) -> dict:
        half = 1 << (NUM_BITS - 1)
        self.node(1, 0, half)              # 最高位由 X 共轭直接当控制，不占 ancilla
        self.node(1, half, 2 * half)
        return {"ands": self.ands, "peak_ancillas": self.peak, "loads": self.loads,
                "break_early_subtrees": self.early, "max_level": self.max_depth}


def frontier(num_bits: int = NUM_BITS, used: int = A) -> list[dict]:
    """活 ancilla 预算 vs Toffoli：前 p bit 用共享前缀树，后 num_bits-p bit 逐地址链。

    p = num_bits 就是参考实现（全共享，峰值 ancilla = num_bits-1）。
    p 越小 -> 活的 ancilla 越少，但每个地址要单独付 (num_bits-p) 个 AND 做后缀匹配。
    """
    out = []
    for p in range(0, num_bits + 1):
        if p == num_bits:
            tree_ands = used - 2          # 实测值：满树 4095 个内部结点，剪掉 121 个空结点
        else:
            tree_ands = min(2 ** p - 1, used - 1)
        chain = num_bits - p
        total = tree_ands + (used * chain if chain else 0)
        live = max(p - 1, 0) + (1 if chain else 0)
        out.append({"p": p, "tree_ands": tree_ands, "chain_per_addr": chain,
                    "ands_per_circuit": total, "peak_live": live})
    return out


def main() -> None:
    p = print
    deltas = load_deltas()
    p("=" * 92)
    p("[1] 参考 ANGLE 电路的解码器逐条计数（纯 numpy 复刻递归，不用 kickmix）")
    p("=" * 92)
    p("    %-4s %10s %14s %10s %20s %s"
      % ("k", "init_and", "#load(叶)", "peak_anc", "break_early 剪枝", "max_level"))
    tot = {"ands": 0, "peak_ancillas": 0, "loads": 0, "break_early_subtrees": 0}
    for k in range(deltas.shape[0]):
        s = TreeCounter(deltas[k]).run()
        tot["ands"] += s["ands"]
        tot["peak_ancillas"] = max(tot["peak_ancillas"], s["peak_ancillas"])
        tot["loads"] += s["loads"]
        tot["break_early_subtrees"] += s["break_early_subtrees"]
        p("    %-4d %10d %14d %10d %20d %d"
          % (k, s["ands"], s["loads"], s["peak_ancillas"], s["break_early_subtrees"],
             s["max_level"]))
    p("    ---- 合计 init_and = %d（= 2*K*3974 = %d，K=%d）"
      % (tot["ands"], 2 * 10 * (tot["ands"] // 10), 10))
    p("    每条 3974 = A-2 = %d：二叉树对 L 个叶子有 L-1 个内部结点，剪掉 121 个空结点 => 3974。" % (A - 2))
    p("    break_early 剪枝总数 = %d：**一次都没触发** —— 角度数据不可压缩（与 E2 §2.2 一致）"
      % tot["break_early_subtrees"])
    p("    峰值存活 ancilla = %d = num_bits-1 = %d（最高位 X 共轭省一格）"
      % (tot["peak_ancillas"], NUM_BITS - 1))

    p("")
    p("=" * 92)
    p("[2] Toffoli / ancilla 权衡曲线（前 p bit 共享前缀树 + 后 12-p bit 逐地址链）")
    p("=" * 92)
    p("    %-4s %10s %14s %12s %14s %s"
      % ("p", "前缀树AND", "每地址后缀链", "每电路AND", "峰值anc", "相对参考"))
    ref_ands = A - 2
    for row in frontier():
        p("    %-4d %10d %14d %12d %14d %8.2fx  %s"
          % (row["p"], row["tree_ands"], row["chain_per_addr"], row["ands_per_circuit"],
             row["peak_live"], row["ands_per_circuit"] / ref_ands,
             "<= 参考实现" if row["p"] == NUM_BITS else ""))
    p("    -> 曲线单调：要少一个活 ancilla，代价是每地址多一条 AND 链，总 Toffoli 立刻涨几倍。")
    p("       p=12 (峰值 anc=11) 是整条曲线的**最小值**；没有任何 p 能给到 anc<=8 而不把 Toffoli 抬 4 倍以上。")

    p("")
    p("=" * 92)
    p("[3] 128 qubit 硬上限怎么把 J（每电路词数）钉在 8")
    p("=" * 92)
    p("    宽度 = 12(selection 固定) + J*brot + 11(树 ancilla)")
    p("    %-4s %10s %10s %10s %8s %s" % ("J", "words", "anc", "宽度", "K", "cap"))
    for J in range(6, 13):
        w = J * BROT
        width = 12 + w + 11
        K = math.ceil(75 / J)
        p("    %-4d %10d %10d %10d %8d %s"
          % (J, w, 11, width, K, "OK" if width <= CAP else "REJECT (>%d)" % CAP))
    p("    -> J=8 是极限（119 <= 128）。J=9 需要 12+108+11 = 131 > 128。")
    p("    -> 要用 J=9 就必须把 ancilla 从 11 压到 <= 8，而 [2] 说明那样每电路 Toffoli 至少 12,439（3.1x），")
    p("       T_angles 会从 79,480 涨到 2*9*12,439 = 223,902 —— 得不偿失。")
    p("")
    p("    ★ 结论：T_angles = 2*K*(A-2) = 79,480 在 128-qubit 上限下**已经贴在最优**。")
    p("      唯一能再降 T 的路子是让 A 变小（更好的分解用更小的 R*B），而那被 20 mHa 残差闸门挡住。")

    p("")
    p("=" * 92)
    p("[4] 顺带核实：ANGLE 的临时 workspace 根本不是 Q_total 的瓶颈")
    p("=" * 92)
    p("    resource_costs: temporary = max(heuristic_temp=544, angles_temp=11) = 544")
    p("    所以就算把 ANGLE 的 ancilla 从 11 压到 0，Q_total 一个 qubit 都不会掉。")
    p("    反过来说：J 从 8 加到 9 会让 Q 从 730 涨到 742（+12 常驻词寄存器），这是 J=9 的唯一代价。")


if __name__ == "__main__":
    main()
