import random
import math

# ====================== 参数配置 ======================
Vread = 0.2                     # 读电压 (V)
R_LRS_mean = 9800               # LRS 平均电阻 (Ω)
R_LRS_std = 960                 # LRS 标准差 (Ω)
R_HRS_mean = 21.4e6             # HRS 平均电阻 (Ω)
R_HRS_std = 8.4e6               # HRS 标准差 (Ω)
R_on = 100                      # 晶体管导通电阻 (Ω)
I_leak = 1e-12                  # 关态漏电流 (A)
R_off = Vread / I_leak          # 关态等效电阻 (Ω)（仅用于参考）

# 根据阵列规模定义每段导线电阻 (Ω)
# 10x10 阵列: 62.3 Ω/段, 100x100 阵列: 20.2 Ω/段
R_seg_table = {
    10: 62.3,
    100: 20.2
}

def get_R_seg(N):
    """根据阵列规模 N 返回对应的每段导线电阻"""
    return R_seg_table[N]

random.seed(42)                 # 固定随机种子，结果可重复

def gaussian(mean, std):
    """Box-Muller 生成高斯分布随机数"""
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z

def get_cell_resistance(state='LRS'):
    """返回单个单元的忆阻器 + 导通电阻（典型值，非随机）"""
    if state == 'LRS':
        return R_LRS_mean + R_on
    else:   # HRS
        return R_HRS_mean + R_on

def analyze_read_disturb_and_irdrop(N, selected_state='LRS'):
    """
    读干扰 & IR 压降：选中单元为 LRS（最坏电流情况）
    假设选中单元位于最远角落 (N-1, N-1)
    """
    # 根据规模获取当前的每段电阻
    R_seg = get_R_seg(N)

    # 最远单元的 BL 段数 = 行号（从底部驱动器算起），SL 段数 = 列号（从左侧地算起）
    n_bl = N - 1   # 从底部驱动器到第 N-1 行经过 N-1 段
    n_sl = N - 1   # 从第 N-1 列到左侧地经过 N-1 段
    R_parasitic_total = (n_bl + n_sl) * R_seg   # 位线BL和源线SL总寄生电阻
    R_cell = get_cell_resistance(selected_state)  # 选中单元的电阻（忆阻器+晶体管）
    R_total = R_parasitic_total + R_cell           # 总串联电阻
    I_read = Vread / R_total                       # 实际电流
    V_cell = I_read * R_cell                       # 选中单元实际电压
    V_drop = I_read * R_parasitic_total            # 寄生电阻分压
    rel_error = (Vread - V_cell) / Vread           # 相对误差
    return {
        'R_parasitic_total': R_parasitic_total,
        'R_cell': R_cell,
        'I_read': I_read,
        'V_cell': V_cell,
        'V_drop': V_drop,
        'rel_error': rel_error
    }

def analyze_sneak_path(N, selected_state='HRS'):
    """
    潜行路径：最坏情况选中单元为 HRS（读电流最小）
    漏电流来源：与选中位线同列、但行未选中的单元，共 (N-1) 个。
    未选中位线所在列、以及字线/位线均未选中的单元漏电流可忽略。
    """
    # 根据规模获取当前的每段电阻
    R_seg = get_R_seg(N)

    # 选中单元读电流（考虑最远寄生电阻，最坏情况）
    n_bl = N - 1
    n_sl = N - 1
    R_parasitic_far = (n_bl + n_sl) * R_seg
    R_selected = get_cell_resistance(selected_state)   # HRS
    I_selected = Vread / (R_selected + R_parasitic_far) # 实际读电流

    # 总潜行漏电流：与选中位线同列的其他 N-1 个单元
    total_leakage = (N - 1) * I_leak
    # 分流比 = 漏电流总和 / 选中读电流
    sneak_ratio = total_leakage / I_selected if I_selected > 0 else float('inf')

    return {
        'I_selected': I_selected,
        'total_leakage': total_leakage,
        'sneak_ratio': sneak_ratio
    }

# ====================== 主仿真 ======================
print("=" * 70)
print("1T1R 忆阻器阵列读/写干扰、IR 压降及潜行路径仿真（修正模型）")
print(f"读电压: {Vread} V, 晶体管导通电阻: {R_on} Ω, 关态漏电流: {I_leak} A")
print("=" * 70)

for N in [10, 100]:
    # 获取当前规模对应的每段线电阻，用于打印信息
    R_seg_current = get_R_seg(N)
    print(f"\n>>> 阵列规模: {N}×{N} (每段导线电阻 = {R_seg_current} Ω)")

    # 1. 读干扰 & IR 压降（选中 LRS，最坏电流）
    res_rd = analyze_read_disturb_and_irdrop(N, selected_state='LRS')
    print("\n[读干扰 & IR 压降] (选中单元 LRS，位于最远角落)")
    print(f"  路径总寄生电阻: {res_rd['R_parasitic_total']:.2f} Ω")
    print(f"  单元总电阻 (R_mem + R_on): {res_rd['R_cell']:.2f} Ω")
    print(f"  读电流: {res_rd['I_read']:.3e} A")
    print(f"  单元实际电压: {res_rd['V_cell']:.4f} V")
    print(f"  寄生压降: {res_rd['V_drop']:.4f} V")
    print(f"  电压相对误差: {res_rd['rel_error'] * 100:.2f}%")

    # 2. 潜行路径（选中 HRS，漏电流分流最显著）
    res_sp = analyze_sneak_path(N, selected_state='HRS')
    print("\n[潜行路径] (选中单元 HRS，位于最远角落)")
    print(f"  选中单元读电流 (考虑远端寄生): {res_sp['I_selected']:.3e} A")
    print(f"  潜行漏电流来源: 与选中位线同列的 {N - 1} 个未选中单元")
    print(f"  总漏电流: {res_sp['total_leakage']:.3e} A")
    print(f"  漏电流与选中读电流之比: {res_sp['sneak_ratio']:.3e}")
    if res_sp['sneak_ratio'] < 0.01:
        print("  → 分流比 < 1%，潜行路径影响可忽略")
    elif res_sp['sneak_ratio'] < 0.05:
        print("  → 分流比 < 5%，影响较小")
    else:
        print("  → 分流比 ≥ 5%，需谨慎")

    # 3. 写干扰说明
    print("\n[写干扰]")
    print("  写电压通常 >1V。未选中单元晶体管关断，漏电流仅1pA，无法提供足够能量改变忆阻器状态。")
    print("  即使半选中单元（字线或位线之一选中），因晶体管关断，写干扰也可忽略。")

