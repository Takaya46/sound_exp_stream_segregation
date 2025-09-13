# add_from_summary.py
import os
import time
from pathlib import Path
from typing import Union, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# ================= ヘルパー =================

def _require_columns(df: pd.DataFrame, required: set, name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} に必要列がありません: {missing}. 実際の列: {list(df.columns)}")

def _to_numeric_inplace(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _interp_pred(grid_g: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    """pred_grid（あるgroupの行）から sub_value→y_pred を線形補間（範囲外は端でクリップ）"""
    gg = grid_g.dropna(subset=["sub_value","y_pred"]).sort_values("sub_value")
    if gg.empty:
        return np.full_like(x, np.nan, dtype=float)
    xp = gg["sub_value"].to_numpy()
    fp = gg["y_pred"].to_numpy()
    x_clip = np.clip(x, xp.min(), xp.max())
    return np.interp(x_clip, xp, fp)

# ================= コア：図を作って評価も返す =================

def _render_and_score(points_master: pd.DataFrame,
                      pred_grid: pd.DataFrame,
                      new_points_df: pd.DataFrame,
                      summary_df: Optional[pd.DataFrame] = None,
                      *,
                      output_dir="static/plots",
                      fig_prefix="mt_additions",
                      figsize=(12,5),
                      point_color="gray",
                      new_point_color="red",
                      new_point_size=80):
    """既存点は灰、newは赤で上乗せ。再計算はしない。偏差値等も返す。"""

    # 列名統一
    points_master = points_master.rename(columns={c: c.lower() for c in points_master.columns})
    pred_grid     = pred_grid.rename(columns={c: c.lower() for c in pred_grid.columns})
    new_points_df = new_points_df.rename(columns={c: c.lower() for c in new_points_df.columns})

    _require_columns(pred_grid,     {"group","sub_value","y_pred","y_ci_low","y_ci_high"}, "pred_grid")
    _require_columns(points_master, {"group","sub_value","y_obs"},                         "points")
    _require_columns(new_points_df, {"participant","group","sub_value","y_obs"},           "new_points")

    _to_numeric_inplace(pred_grid,     ["sub_value","y_pred","y_ci_low","y_ci_high"])
    _to_numeric_inplace(points_master, ["sub_value","y_obs"])
    _to_numeric_inplace(new_points_df, ["sub_value","y_obs"])

    participant = str(new_points_df["participant"].iloc[0])

    group_order = ["0オクターブ","1オクターブ","2オクターブ","3オクターブ"]
    color_map   = {
        "0オクターブ": "#1b9e77",
        "1オクターブ": "#d95f02",
        "2オクターブ": "#7570b3",
        "3オクターブ": "#e7298a",
    }
    
    # 表示用の英語ラベル変換マップ
    display_labels = {
        "0オクターブ": "0 octave",
        "1オクターブ": "1 octave", 
        "2オクターブ": "2 octave",
        "3オクターブ": "3 octave"
    }

    # 参加者のデータがあるグループのみに絞る
    available_groups = new_points_df["group"].dropna().unique().tolist()
    # group_orderの順序を保持しつつ、available_groupsにあるものだけをフィルタ
    filtered_group_order = [g for g in group_order if g in available_groups]
    
    if not filtered_group_order:
        # データがない場合は空の図を返す
        fig, ax = plt.subplots(1, 1, figsize=(3, 5))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fig_path = Path(output_dir) / f"{fig_prefix}_{participant}_{stamp}_no_data.png"
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
        return {
            "participant": participant,
            "fig_path": str(fig_path),
            "metrics_by_group": [],
            "overall": {"hensachi_mean": np.nan, "percentile_mean": np.nan, "groups_count": 0}
        }

    # ---- 図 ----
    # 各パネルの縦横比を固定（元の4パネル時：12x5 → 1パネルあたり3x5）
    panel_width = 3
    panel_height = 5
    total_width = panel_width * len(filtered_group_order)
    adjusted_figsize = (total_width, panel_height)
    
    fig, axes = plt.subplots(1, len(filtered_group_order), figsize=adjusted_figsize, sharey=True)
    # 1つのグループしかない場合、axesをリストに変換
    if len(filtered_group_order) == 1:
        axes = [axes]
        
    raw_ticks  = np.array([2,4,8,16,32,64,128,256], dtype=float)
    log2_ticks = np.log2(raw_ticks)

    for ax, g in zip(axes, filtered_group_order):
        gd = pred_grid.loc[pred_grid["group"] == g, ["sub_value","y_pred","y_ci_low","y_ci_high"]].dropna().copy().sort_values("sub_value")
        c = color_map[g]
        if not gd.empty:
            ax.fill_between(gd["sub_value"].to_numpy(), gd["y_ci_low"].to_numpy(), gd["y_ci_high"].to_numpy(),
                            alpha=0.20, lw=0, facecolor=c)
            ax.plot(gd["sub_value"].to_numpy(), gd["y_pred"].to_numpy(), lw=1.5, color=c, label=g)

        pd_all = points_master.loc[points_master["group"] == g, ["sub_value","y_obs"]].dropna().copy()
        if not pd_all.empty:
            ax.scatter(pd_all["sub_value"].to_numpy(), pd_all["y_obs"].to_numpy(),
                       alpha=0.55, s=12, color=point_color, zorder=2)

        pd_new = new_points_df.loc[new_points_df["group"] == g, ["sub_value","y_obs"]].dropna().copy()
        if not pd_new.empty:
            ax.scatter(pd_new["sub_value"].to_numpy(), pd_new["y_obs"].to_numpy(),
                       s=new_point_size, color=new_point_color, edgecolors="white", linewidths=0.6, zorder=3)

        ax.set_title(display_labels.get(g, g))  # 英語ラベルで表示
        ax.set_xlabel("Gold-MSI MT scores")
        ax.set_yticks(log2_ticks)
        ax.set_yticklabels([str(int(v)) for v in raw_ticks])

    axes[0].set_ylabel("75 % Threshold")
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fig_path = Path(output_dir) / f"{fig_prefix}_{participant}_{stamp}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    # ---- 成績（偏差値/順位） ----
    metrics = []
    for g in filtered_group_order:
        pd_new = new_points_df.loc[new_points_df["group"] == g, ["sub_value","y_obs"]].dropna().copy()
        if pd_new.empty:
            continue
        x_new = pd_new["sub_value"].to_numpy()
        y_new = pd_new["y_obs"].to_numpy()

        gd = pred_grid.loc[pred_grid["group"] == g, ["sub_value","y_pred"]].dropna().copy()
        if gd.empty:
            for yi in y_new:
                metrics.append({
                    "group": g, "y_obs": float(yi), "y_pred": np.nan,
                    "residual": np.nan, "score": np.nan, "z": np.nan,
                    "hensachi": np.nan, "percentile": np.nan, "rank": None,
                    "n": int(points_master.loc[points_master["group"] == g].shape[0])
                })
            continue

        y_pred_new = _interp_pred(gd, x_new)

        pd_all = points_master.loc[points_master["group"] == g, ["sub_value","y_obs"]].dropna().copy()
        if pd_all.empty:
            mu = sigma = None
            threshold_values = None
        else:
            # 45人の閾値データ(y_obs)から直接平均・標準偏差を計算
            threshold_values = pd_all["y_obs"].to_numpy()
            mu = float(np.nanmean(threshold_values)) if threshold_values.size else None
            sigma = float(np.nanstd(threshold_values, ddof=1)) if threshold_values.size > 1 else None

        # このgroupに対応するsub_valueを取得
        group_sub_value = None
        if summary_df is not None:
            # 元のsummary.csvからsub_valueを取得
            participant_data = summary_df[summary_df["group"] == g]
            if not participant_data.empty and "sub_value" in participant_data.columns:
                group_sub_value = participant_data["sub_value"].iloc[0]
        
        for yi, yp in zip(y_new, y_pred_new):
            # 被験者の閾値(yi)を直接使って偏差値を計算
            threshold_value = float(yi)
            n_all = int(pd_all.shape[0])
            
            if threshold_values is None or n_all == 0:
                z = hensachi = percentile = rank = None
            else:
                if sigma is None or sigma == 0:
                    z = 0.0
                else:
                    # 閾値は小さいほど良いので、符号を反転
                    z = -(threshold_value - mu) / sigma
                hensachi = 50 + 10 * z
                # パーセンタイルと順位計算（小さい値ほど良い）
                better_count = np.sum(threshold_values > threshold_value)
                percentile = float(100.0 * better_count / n_all)
                rank = int(np.sum(threshold_values < threshold_value)) + 1  # 1が最優秀

            metrics.append({
                "group": g,
                "y_obs": float(yi),
                "y_pred": float(yp),
                "threshold_value": threshold_value,
                "threshold_mean": float(mu) if mu is not None else np.nan,
                "threshold_std": float(sigma) if sigma is not None else np.nan,
                "z": float(z) if z is not None else np.nan,
                "hensachi": float(hensachi) if hensachi is not None else np.nan,
                "percentile": float(percentile) if percentile is not None else np.nan,
                "rank": rank,
                "n": n_all,
                "sub_value": group_sub_value
            })

    dfm = pd.DataFrame(metrics)
    overall = {"hensachi_mean": float(dfm["hensachi"].mean(skipna=True)) if not dfm.empty else np.nan,
               "percentile_mean": float(dfm["percentile"].mean(skipna=True)) if not dfm.empty else np.nan,
               "groups_count": int(dfm.shape[0]) if not dfm.empty else 0}

    return {
        "participant": participant,
        "fig_path": str(fig_path),
        "metrics_by_group": metrics,
        "overall": overall
    }

# ================= 入口：summary.csv から新規点を作る =================

def add_from_summary(
    participant_id: str,
    *,
    base_dir: str = "static/thrMt_fig_data",  # thrMt_fig_data は固定で使う想定
    summary_csv_path: Optional[str] = None,   # 個別のsummary.csvパス
    subscore: str = "MT_sum",
    sound_type: Optional[str] = None,         # 例: "pure_tone"。指定しなければ全件から最新を使う
    output_dir: str = "static/plots",
    fig_prefix: str = "mt_additions",
    figsize=(12,5)
) -> dict:
    """
    個別のsummary.csv を読み込み、指定 participant の行を抽出・整形して
    "赤い追加点"として上乗せ図を保存し、偏差値/順位などを返す。
    再計算なし。thrMt_fig_data の pred_grid/points を土台に使う。
    """
    base = Path(base_dir)
    # 土台
    grid = pd.read_csv(base / f"{subscore}_pred_grid.csv")
    pts  = pd.read_csv(base / f"{subscore}_points.csv")

    # summary.csv → new points 抽出
    if summary_csv_path is not None:
        summ_path = Path(summary_csv_path)
    else:
        summ_path = base / "summary.csv"
    
    if not summ_path.exists():
        raise FileNotFoundError(f"{summ_path} が見つかりません。")

    summ = pd.read_csv(summ_path)
    # 列名を小文字に
    summ.columns = [c.lower() for c in summ.columns]

    # 必須列チェック
    _require_columns(
        summ,
        {"participant_id","experiment_date","sound_type",
         "frequency_condition","frequency_label",
         "threshold_ms","log2_threshold","level","sub_value"},
        "summary.csv"
    )

    # 対象参加者 & サウンドタイプで絞る
    sdf = summ[summ["participant_id"] == participant_id].copy()
    if sound_type is not None:
        sdf = sdf[sdf["sound_type"] == sound_type]
    if sdf.empty:
        raise ValueError(f"summary.csv に participant_id={participant_id} の行がありません（sound_type={sound_type} も考慮）。")

    # 日付でソートし最新を使う（groupごとに直近1つ）
    sdf["experiment_date"] = pd.to_datetime(sdf["experiment_date"], errors="coerce")
    sdf = sdf.sort_values(["frequency_condition","experiment_date"]).dropna(subset=["experiment_date"])

    # Frequency を土台の日本語ラベルへ正規化
    # thrMt_fig_data 側は ["0オクターブ","1オクターブ","2オクターブ","3オクターブ"] を使っている前提
    cond2label = {
        "g_base":     "0オクターブ",
        "g_1octave":  "1オクターブ",
        "g_2octave":  "2オクターブ",
        "g_3octave":  "3オクターブ",
    }
    sdf["group"] = sdf["frequency_condition"].map(cond2label)

    # 必要列の整形：y_obs は log2_threshold（無ければ threshold_ms から計算）
    y_obs = sdf["log2_threshold"].copy()
    if y_obs.isna().any():
        y_obs = np.where(sdf["threshold_ms"].notna(), np.log2(sdf["threshold_ms"].astype(float)), np.nan)

    new_points = (
        sdf.assign(
            participant = participant_id,
            y_obs = y_obs
        )
        .dropna(subset=["group","sub_value","y_obs"])
        .sort_values(["group","experiment_date"])
        .drop_duplicates(subset=["group"], keep="last")  # groupごと直近1つに集約
        [["participant","group","sub_value","y_obs"]]
        .reset_index(drop=True)
    )

    # コア処理へ（図＋評価）
    result = _render_and_score(
        points_master = pts,
        pred_grid     = grid,
        new_points_df = new_points,
        summary_df    = sdf,
        output_dir    = output_dir,
        fig_prefix    = fig_prefix,
        figsize       = figsize
    )

    # 追加で、summary.csv側のメタ（level など）をくっつけたい場合はここで
    # groupごと直近の level を拾っておく
    level_map = (
        sdf.sort_values(["group","experiment_date"])
           .drop_duplicates(subset=["group"], keep="last")[["group","level"]]
           .set_index("group")["level"].to_dict()
    )
    result["levels_by_group"] = level_map
    return result
