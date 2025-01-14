import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf

# データの読み込み
data_path = './data/hirano_staircase_offsets.csv'
data = pd.read_csv(data_path)
data['log2_offset'] = np.log2(data['Offset'])
log2_offset = data['log2_offset'].values
correct = data['Correct'].astype(int).values

# モデルの定義とサンプリング
with pm.Model() as model:
    # パラメータの事前分布
    a = pm.Normal("a", mu=0, sigma=10)
    b = pm.Normal("b", mu=0, sigma=10)
    
    # 線形モデル
    z = a * log2_offset + b
    
    # 確率モデル（2AFC）
    two_afc_probit = 0.5 + 0.5 * (0.5 + 0.5 * pm.math.erf(z / pm.math.sqrt(2)))
    p = pm.Deterministic("p", two_afc_probit)
    
    # 観測モデル
    y_obs = pm.Bernoulli("y_obs", p=p, observed=correct)
    
    # サンプリング
    trace = pm.sample(2000, tune=1000, chains=3, target_accept=0.99, cores=1)
    
    # MAP推定
    map_estimate = pm.find_MAP()

# MAP推定値を取得
a_map = map_estimate["a"]
b_map = map_estimate["b"]

# プロット用の値
x_plot = np.linspace(log2_offset.min(), log2_offset.max(), 100)
z_map = a_map * x_plot + b_map
p_map = 0.5 + 0.5 * (0.5 + 0.5 * erf(z_map / np.sqrt(2)))

# 事後サンプルを取得
a_samples = trace.posterior["a"].stack(draws=("chain", "draw")).values
b_samples = trace.posterior["b"].stack(draws=("chain", "draw")).values

# プロット
plt.figure(figsize=(10, 6))

# 不確実性を反映した曲線の描画
for i in range(100):  # 100本の事後分布サンプル曲線
    a_sample = a_samples[i]
    b_sample = b_samples[i]
    z_sample = a_sample * x_plot + b_sample
    p_sample = 0.5 + 0.5 * (0.5 + 0.5 * erf(z_sample / np.sqrt(2)))
    plt.plot(x_plot, p_sample, color='blue', alpha=0.1)

# データポイント
plt.scatter(log2_offset, correct, label="Data", color="black", alpha=0.6)

# MAP推定曲線の描画
plt.plot(x_plot, p_map, color='red', label="MAP Fitted Curve", linewidth=2)

# ラベルとタイトル
plt.xlabel("log2(Offset)")
plt.ylabel("Probability")
plt.title("MAP Fitted Curve with Posterior Uncertainty")
plt.legend()
plt.grid(True)
plt.show()