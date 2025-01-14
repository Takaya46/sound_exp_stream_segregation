import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import minimize

# データの読み込み
data_path = './data/oya_staircase_offsets.csv'
data = pd.read_csv(data_path)
data['log2_offset'] = np.log2(data['Offset'])
log2_offset = data['log2_offset'].values
correct = data['Correct'].astype(int).values

# 負の対数尤度関数
def negative_log_likelihood(params):
    a, b = params
    z = a * log2_offset + b
    p = 0.5 + 0.5 * (0.5 + 0.5 * erf(z / np.sqrt(2)))
    epsilon = 1e-10  # 数値誤差回避
    p = np.clip(p, epsilon, 1 - epsilon)
    return -np.sum(correct * np.log(p) + (1 - correct) * np.log(1 - p))

# 最適化（MLE）
result = minimize(negative_log_likelihood, x0=[0, 0], method='Nelder-Mead')
a_mle, b_mle = result.x

# プロット用の値
x_plot = np.linspace(0, 6, 100)  # x軸を [0, 6] に統一
z_mle = a_mle * x_plot + b_mle
p_mle = 0.5 + 0.5 * (0.5 + 0.5 * erf(z_mle / np.sqrt(2)))

# プロット
plt.figure(figsize=(10, 6))

# データ点のジッター（x軸とy軸方向）
jittered_x = log2_offset + np.random.uniform(-0.05, 0.05, size=len(log2_offset))
jittered_y = correct + np.random.uniform(-0.03, 0.03, size=len(correct))

# ジッターを追加したデータ点の描画
plt.scatter(jittered_x, jittered_y, label="Data", color="black", alpha=0.6)

# MLEによるフィッティング曲線
plt.plot(x_plot, p_mle, color='blue', label="MLE Fitted Curve")

# 75% 閾値の計算と描画
threshold_log2_mle = -b_mle / a_mle
threshold_mle = 2 ** threshold_log2_mle
plt.axvline(threshold_log2_mle, color='green', linestyle='--', label=f"75% Threshold ({threshold_mle:.2f})")

# プロット設定
plt.xlabel("log2(Offset[ms])")
plt.ylabel("Probability")
plt.title("Psychometric Curve with MLE and 75% Threshold")
plt.legend()
plt.grid(True)
plt.xlim(0, 6)  # x軸を [0, 6] に固定
plt.ylim(-0.1, 1.1)  # y軸を [0, 1] に固定して見やすく
plt.show()