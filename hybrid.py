import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------
# ✅ 1. 公共参数
# -------------------------------------
beta_E = 1.2
gamma = 0.7
lambda_p = 0.3
alpha = 0.01
actions_fine = np.linspace(0, 1, 101)  # 0 ~ 1

# -------------------------------------
# ✅ 2. 不同法系参数 (κ/η)
# （η 已 ×50）
# -------------------------------------
legal_systems = {
    "common_law": {"kappa": 0.791474314, "eta": 0.0096485 * 200},
    "civil_law": {"kappa": 0.655739046, "eta": 0.0062033 * 200}
}

results = []

# -------------------------------------
# ✅ 3. 不同法系循环
# -------------------------------------
for system, params in legal_systems.items():
    kappa = params["kappa"]
    eta = params["eta"]

    a_c, a_p = 0.5, 0.5  # 初始值

    for t in range(1, 501):

        # ✅ 真实采纳率
        R = a_c * a_p

        # === 📌 混合动态更新 ===

        # 1️⃣ 法院：Best Response
        def court_payoff(ac):
            return beta_E * ac - gamma * (ac**2) - kappa * abs(ac - a_p)
        best_ac = max(actions_fine, key=court_payoff)
        a_c += alpha * (best_ac - a_c)
        a_c = max(min(a_c, 1.0), 0.0)  # 强制限制

        # 2️⃣ 公众：Replicator Dynamics
        U_p = beta_E * R - eta * (R**3) - lambda_p * (a_p**2)
        U_p_baseline = 0.0  # 不支持时无收益无成本
        U_p_mean = a_p * U_p + (1 - a_p) * U_p_baseline
        dot_ap = a_p * (U_p - U_p_mean)
        a_p += alpha * dot_ap
        a_p = max(min(a_p, 1.0), 0.0)  # 强制限制

        # ✅ 记录结果
        results.append({
            'system': system,
            't': t,
            'a_c': round(a_c, 4),
            'a_p': round(a_p, 4),
            'R': round(R, 4),
            'T': round(1 - a_c**2, 4),
            'E': round(a_c, 4),
            'court_U': round(court_payoff(a_c), 4),
            'public_U': round(U_p, 4),
            'diff': round(abs(a_c - a_p), 4)
        })

# -------------------------------------
# ✅ 4. 保存结果
# -------------------------------------
df = pd.DataFrame(results)
df.to_csv("hybrid_dynamics_stable.csv", index=False)
print("✅ 已保存：hybrid_dynamics_stable.csv")

# -------------------------------------
# ✅ 5. 可视化对比
# -------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='t', y='R', hue='system')
plt.title('不同法系的实际采纳率 R 收敛对比')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='t', y='diff', hue='system')
plt.title('不同法系的分歧度 |a_c - a_p| 收敛对比')
plt.show()

# -------------------------------------
# ✅ 6. 长期均值
# -------------------------------------
print("\n=== 长期均值 (后 100 期) ===")
for system in df['system'].unique():
    last_part = df[(df['system'] == system) & (df['t'] > 400)]
    print(f"\n法系: {system}")
    print("法院采纳率均值 a_c:", round(last_part['a_c'].mean(), 4))
    print("公众意向均值 a_p:", round(last_part['a_p'].mean(), 4))
    print("实际采纳率均值 R:", round(last_part['R'].mean(), 4))
    print("分歧度均值 diff:", round(last_part['diff'].mean(), 4))
    print("法院效用均值 court_U:", round(last_part['court_U'].mean(), 4))
    print("公众效用均值 public_U:", round(last_part['public_U'].mean(), 4))
