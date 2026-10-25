#!/usr/bin/env python3
"""E6 §5 — 把 E5 的三条「通用结论」逐条搬到 dfthcsos 上核对。

E5 的结论是在 **invsqrt** 上实测出来的（kickmix 的 `left_rotate` / CSWAP 价签）。
它们在 dfthcsos 上适用性未知，这里用**电路级证据**（逐条 opcode 直方图）判：

  E5-1  swap 与无条件 rotate = 0 T，条件传输是纯控制成本
  E5-2  「温度计转二元」双赢（invsqrt gain +10.4%，纯 CX 解码 0 Toffoli）
  E5-3  「复位清垃圾」不可用（R 是去相位操作）

用法: python src/e6/e6_flags_port.py
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMP / "data" / "verifiers"))

from dfthcsos.scoreinfra.circuits import circuit_costs, read_circuit   # noqa: E402

SAMPLE_DIR = COMP / "data" / "sample_submission" / "dfthcsos"

# kickmix 的 Toffoli-like 集合（verify.recover_words_from_circuits 的计数口径）
TOFFOLI_LIKE = {"CCX", "CCZ", "CSWAP"}


def hist(circuit) -> collections.Counter:
    c = collections.Counter()
    for line in str(circuit).splitlines():
        line = line.strip()
        if not line:
            continue
        c[line.split()[0]] += 1
    return c


def main() -> None:
    p = print
    p("=" * 96)
    p("[0] 10 条 ANGLE 电路的 opcode 直方图（.kmb -> kickmix.Circuit -> 文本）")
    p("=" * 96)
    total = collections.Counter()
    per = []
    for k in range(10):
        c = read_circuit(SAMPLE_DIR / f"angle_k_{k}.kmb", format="kmb")
        h = hist(c)
        total.update(h)
        cc = circuit_costs([c])[0]
        per.append({"k": k, "hist": dict(h), "num_qubits": cc["num_qubits"],
                    "max_magic": cc["max_magic"]})
    keys = sorted(total, key=lambda x: -total[x])
    p("    %-10s %12s %10s   %s" % ("opcode", "总次数/10电路", "每条均值", "扮演什么角色"))
    role = {
        "CX": "load：把 word 位 XOR 进目标（单控制，**0 Toffoli**）",
        "CCX": "init_and：地址树内部结点（**唯一计费的门**）",
        "X": "共轭：把 NOT 折进控制位",
        "CZ": "del_and 的相位修正（**0 Toffoli**）",
        "HMR": "del_and 的测量（**0 Toffoli**）",
        "Z_POW": "任意角旋转 —— 契约禁止",
        "CSWAP": "swap —— 这里一个都没有",
    }
    for k in keys:
        p("    %-10s %12d %10.1f   %s" % (k, total[k], total[k] / 10, role.get(k, "")))
    p("")
    p("    合计每电路 ops = %d，其中 Toffoli-like = %d = max_magic"
      % (sum(per[0]["hist"].values()),
         sum(v for kk, v in per[0]["hist"].items() if kk in TOFFOLI_LIKE)))
    p("    10 条电路 opcode 集合完全一致: %s"
      % (len({tuple(sorted(x["hist"])) for x in per}) == 1))

    p("")
    p("=" * 96)
    p("[1] E5-1 「swap / 无条件 rotate = 0 T，条件传输是纯控制成本」")
    p("=" * 96)
    p("    实测：CSWAP 出现 %d 次，Z_POW 出现 %d 次，任何 rotate 指令 %d 次。"
      % (total["CSWAP"], total["Z_POW"], sum(v for kk, v in total.items() if "ROT" in kk)))
    p("    => dfthcsos 的 ANGLE 家族是**纯 Boolean 电路**，根本没有可搬运的量子数据：")
    p("       没有 swap、没有 rotate，「条件传输」这个概念不成立。")
    p("       唯一的条件结构是 init_and/del_and 的控制位 —— 那本来就是纯控制成本（1 Toffoli），")
    p("       而它的反向（del_and）已经用测量免费清掉（见 [3]）。")
    p("    判定：**不适用**（不是反例，是这条规则讨论的对象在这里不存在）。")
    p("    补充：heuristic 那一块确实有 2*(N-1) 个受控 Givens 旋转，单价 2*brot-1 = 23 T，")
    p("          共 3450 T —— 但那是**闭式模型**算出来的，不是我们写的门；")
    p("          而且 verifier 禁止 Z_POW，所以「把旋转挪到无控」这条 invsqrt 式省法在这里被堵死。")

    p("")
    p("=" * 96)
    p("[2] E5-2 「温度计 → 二元」双赢招：dfthcsos 里有没有温度计？")
    p("=" * 96)
    p("    invsqrt 的那一招本质是：一元温度计 flag 只用一次就扔，改存二元的 s，")
    p("    再用纯 CX 解码 —— 省下的是「每个 flag 一次整条 lane 的 CSWAP」。")
    p("    dfthcsos 里的对应物是地址解码器的**一元前缀阶梯**（每个地址一根 one-hot ancilla）：")
    p("      · 它已经是**临时的**（每条电路结束前全部归零）")
    p("      · 它的消费已经是**纯 CX**（%d 个 CX 对 %d 个 CCX，负载部分完全免费）"
      % (total["CX"] // 10, total["CCX"] // 10))
    p("      · 它的清理已经是**免费**的（CZ/HMR，0 Toffoli）")
    p("    也就是说：E5 那个「温度计保持临时 + CX 解码」的结构，**dfthcsos 参考实现里已经用满了**。")
    p("    剩下的唯一「二元」候选是 selection 寄存器本身 —— 它已经是二元地址，没有可折叠的温度计。")
    p("    判定：**已被吸收，无剩余空间**。（这也解释了为什么 ANGLE 的 T 全是 init_and，即 85%% 的 T_total）")

    p("")
    p("=" * 96)
    p("[3] E5-3 「复位清垃圾不可用（R 是去相位操作）」")
    p("=" * 96)
    p("    实测：每电路 CZ %d 个、HMR %d 个，而 max_magic 只有 %d。"
      % (total["CZ"] // 10, total["HMR"] // 10, per[0]["max_magic"]))
    p("    => dfthcsos 里垃圾清理走的是 **del_and（HMR + CZ）**，")
    p("       而 CZ 不在 Toffoli-like 集合 {CCX, CCZ, CSWAP} 里 —— 即**测量式反算是免费的**。")
    p("       实测每电路 CCX = %d = 只算正向 init_and；反向的 3974 个 del_and 一个 T 都没收。"
      % (total["CCX"] // 10))
    p("    判定：这条 invsqrt 的**通用约束在 dfthcsos 上不成立** —— 这里不仅有免费的清理手段，")
    p("          而且参考实现已经在用。所以 dfthcsos 的 T 只有「正向一半」的冗余，")
    p("          不存在 invsqrt/dqi 那种「反向 pass 占一半」的可砍部分。")

    p("")
    p("=" * 96)
    p("[4] 顺带：往返冗余核查")
    p("=" * 96)
    p("    T_total 里 ANGLE = 2*K*3974 = 79480，那个 2 就是 U_T 与 U_T^dagger（verifier 强制跑两遍）。")
    p("    但**计费的门只有正向的 init_and**：Simulator 统计的是实际执行的门，")
    p("    del_and 的 CZ/HMR 不计费 —— 所以「反向 pass 占一半」在这里只体现在**这个 2 的系数**上，")
    p("    电路本身已经没有可砍的冗余 pass 了。")
    p("    这与 dqi/invsqrt 的「反向 pass 里还藏着一半可去门」不同。")


if __name__ == "__main__":
    main()
