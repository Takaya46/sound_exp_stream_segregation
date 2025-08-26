import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg')  # GUIを無効化

def perform_mle_analysis(data_file, output_dir):
    """
    Perform MLE fitting and save the resulting plot.
    
    Parameters:
        data_file (str): Path to the input CSV file with experimental data.
        output_dir (str): Directory to save the resulting plot.
    
    Returns:
        dict: Analysis results including threshold and path to the saved figure.
    """
    # データの読み込み
    if not os.path.exists(data_file):
        return {"error": f"Data file {data_file} does not exist."}

    data = pd.read_csv(data_file)
    if 'Offset' not in data or 'Correct' not in data:
        return {"error": "Required data columns not found in the file."}
    
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

    # 最適化
    result = minimize(negative_log_likelihood, x0=[0, 0], method='Nelder-Mead')
    a_mle, b_mle = result.x

    # プロット用データ
    x_plot = np.linspace(0, 6, 100)
    z_mle = a_mle * x_plot + b_mle
    p_mle = 0.5 + 0.5 * (0.5 + 0.5 * erf(z_mle / np.sqrt(2)))
    
    # 図の作成
    plt.figure(figsize=(7, 4))

    # データ点の描画（ジッター付き）
    jittered_x = log2_offset + np.random.uniform(-0.1, 0.1, size=len(log2_offset))
    jittered_y = correct + np.random.uniform(-0.05, 0.05, size=len(correct))
    plt.scatter(jittered_x, jittered_y, label="Data", color="black", alpha=0.6)

    # フィッティング曲線の描画
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
    plt.xlim(0, 6)
    plt.ylim(-0.1, 1.1)

    # 図を保存するパス。既存のファイルがある場合、新しい名前を付ける
    base_fig_path = os.path.join(output_dir, 'mle_fitted_curve.png')
    fig_path = base_fig_path
    count = 1
    while os.path.exists(fig_path):
        fig_path = os.path.join(output_dir, f"mle_fitted_curve_{count}.png")
        count += 1
    # 図の保存
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return {"threshold": threshold_mle, "log2_threshold": threshold_log2_mle, "fig_path": fig_path}