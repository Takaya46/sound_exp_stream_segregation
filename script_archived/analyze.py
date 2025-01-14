import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

FIGURE_FOLDER = 'static/figures'
os.makedirs(FIGURE_FOLDER, exist_ok=True)

# プロビット関数
def probit(x, alpha, beta):
    return norm.cdf(beta * (np.log2(x) - alpha))

# 75%閾値計算
def calculate_threshold(alpha, beta, threshold=0.75):
    log_theta = norm.ppf((threshold - 0.5) / 0.5, loc=-alpha/beta, scale=1/beta)
    return 2 ** log_theta

# データ分析関数
def process_results(filename):
    df = pd.read_csv(filename)
    df = df[['Correct', 'Offset']].dropna()
    df['corr'] = df['Correct'].astype(float)
    df['log_offset'] = np.log2(df['Offset'])

    # フィッティング
    x_data = df['Offset']
    y_data = df['corr']
    popt, _ = curve_fit(probit, x_data, y_data, p0=[2, 1])
    alpha, beta = popt

    # フィッティングカーブ生成
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    y_fit = probit(x_fit, alpha, beta)

    # プロット生成
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, c='grey', alpha=0.5, label='Data')
    plt.plot(x_fit, y_fit, 'b-', label='Fitted Curve')
    plt.xlabel("Offset")
    plt.ylabel("Proportion of correct responses")
    plt.title("Fitted Psychometric Curve")
    plt.xscale("log", base=2)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    # 画像を保存
    figure_path = os.path.join(FIGURE_FOLDER, 'psychometric_curve.png')
    plt.savefig(figure_path)
    plt.close()

    # 75%閾値計算
    threshold = calculate_threshold(alpha, beta)
    return threshold, figure_path