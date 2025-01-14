import os
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import norm
from scipy.special import erf

# データの読み込み関数
def load_experiment_data(data_dir, participant_id):
    # ファイルを探索（複数の参加者や実験条件がある場合に対応）
    files = [f for f in os.listdir(data_dir) if f.startswith(participant_id) and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No data files found for participant: {participant_id}")
    
    # 最新のファイルを取得（必要に応じて条件を変更可能）
    file_path = os.path.join(data_dir, files[-1])
    print(f"Loading data from: {file_path}")
    
    # データを読み込む
    data = pd.read_csv(file_path)
    data['log2_offset'] = np.log2(data['Offset'])
    return data

# データの読み込み
data_dir = './data'  # Flaskアプリの `/data` ディレクトリ
participant_id = 'hirano'  # 対象の参加者ID
data = load_experiment_data(data_dir, participant_id)

# データの読み込み
data['log2_offset'] = np.log2(data['Offset'])
processed_data_log2 = data[['log2_offset', 'Correct']].dropna()

# 必要な列を抽出
log2_offset = processed_data_log2['log2_offset'].values
correct = processed_data_log2['Correct'].astype(int).values

# PyMCモデルの定義
with pm.Model() as model:
    # 事前分布（パラメータ a, b の分布を指定）
    a = pm.Normal("a", mu=0, sigma=10)
    b = pm.Normal("b", mu=0, sigma=10)

    # 線形モデル
    z = a * log2_offset + b

    # 確率モデル
    two_afc_probit = 0.5 + 0.5 * (0.5+0.5*pm.math.erf(z / pm.math.sqrt(2)))
    p = pm.Deterministic("p", two_afc_probit)

    # 観測モデル(データ)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=correct)

    # サンプリング
    trace = pm.sample(5000, tune=1000, target_accept=0.95, cores=1, chains=3)

# サンプリング結果の解析
az.plot_trace(trace)
#plt.show()

# 75%閾値の計算
a_mean = trace.posterior["a"].mean().values
b_mean = trace.posterior["b"].mean().values
threshold_log2 = -b_mean / a_mean
threshold = 2 ** threshold_log2
print(f"75% Threshold: {threshold}")

# フィッティング結果をプロット
x_plot = np.linspace(log2_offset.min(), log2_offset.max(), 100)
z_plot = a_mean * x_plot + b_mean
p_plot = 0.5 + 0.5 * (0.5 + 0.5 * erf(z_plot / np.sqrt(2)))

plt.scatter(log2_offset, correct, label="Data", color="black")
plt.plot(x_plot, p_plot, label="Fitted Curve", color="red")
plt.axvline(threshold_log2, color="green", linestyle="--", label="75% Threshold")
plt.xlabel("log2(Offset)")
plt.ylabel("Probability")
plt.legend()
plt.title("2AFC Psychometric Curve Fitting")
plt.grid(True)
#plt.show()