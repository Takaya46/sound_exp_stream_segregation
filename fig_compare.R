library(data.table)
library(tidyverse)
library(psyphy)

# ★ 日付と被験者名の設定（ここで変更できます）
current_dir <- "/Users/takaya/Codes/Nishida-lab/Music_illusion/web_pest2_combine_condition"
date <- "2025-02-18"
sub_name <- "oya_sister"

# ★ ディレクトリのパスを変数で設定
g_base_dir    <- file.path(current_dir, "data", date, sub_name, "g_base")
g_1octave_dir <- file.path(current_dir, "data", date, sub_name, "g_1octave")
g_3octave_dir <- file.path(current_dir, "data", date, sub_name, "g_3octave")
output_dir    <- file.path(current_dir, "static/fig", date, sub_name, "combined")

# 出力ディレクトリが存在しない場合は作成
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# g_baseディレクトリ内のCSVファイルを取得
g_base_files <- list.files(g_base_dir, pattern = "\\.csv$", full.names = TRUE)

# ファイルが見つからない場合のエラーハンドリング
if (length(g_base_files) == 0) {
  stop("g_baseディレクトリにCSVファイルが見つかりません。")
}

# ★ ファイル名からシリアル番号部分（例: "", "_1", "_2"）を抽出する関数
extract_suffix <- function(file_name, condition) {
  # 例: file_name が "takaya_g_base_results.csv" または "takaya_g_base_results_1.csv" の場合
  pattern <- paste0("^", sub_name, "_", condition, "_results(.*)\\.csv$")
  suffix <- sub(pattern, "\\1", file_name)
  if (suffix == file_name) {  # マッチしなかった場合は空文字にする
    return("")
  } else {
    return(suffix)
  }
}

# ★ プロットタイトルを生成する関数（必要に応じてシリアル番号を表示）
get_title <- function(file_name) {
  # g_baseファイル名からsuffixを抽出
  suffix <- extract_suffix(basename(file_name), "g_base")
  if (suffix != "") {
    display_suffix <- gsub("^_", "", suffix)  # 先頭の"_"を削除して表示
    return(paste0("被験者: ", sub_name, " (ファイル", display_suffix, ")"))
  } else {
    return(paste0("被験者: ", sub_name))
  }
}

# g_baseディレクトリ内のすべてのファイルに対して処理
for (g_base_file in g_base_files) {
  
  # g_baseのファイル名からsuffixを取得
  file_base_name <- basename(g_base_file)
  suffix <- extract_suffix(file_base_name, "g_base")
  
  # 各条件のファイル名を生成
  g_1octave_file <- file.path(g_1octave_dir, paste0(sub_name, "_g_1octave_results", suffix, ".csv"))
  g_3octave_file <- file.path(g_3octave_dir, paste0(sub_name, "_g_3octave_results", suffix, ".csv"))
  
  # g_baseデータの読み込みと前処理（ResponseTimeは除外）
  g_base_data <- fread(g_base_file)
  g_base_data <- g_base_data[, c("Response", "Correct", "Offset", "CorrectResponse")]
  colnames(g_base_data) <- c("resp", "corr", "offset", "interval")
  
  # 行数が5未満ならスキップ
  if(nrow(g_base_data) < 5){
    print(paste("欠損データファイル（g_base）:", file_base_name, "行数が5未満のため、処理をスキップします。"))
    next
  }
  
  g_base_data <- mutate(g_base_data, log_offset = log2(offset))
  g_base_data$corr <- as.numeric(g_base_data$corr)
  g_base_data <- na.omit(g_base_data)
  
  # g_1octaveデータの読み込みと前処理
  if (!file.exists(g_1octave_file)) {
    print(paste("対応するg_1octaveファイルが", g_1octave_file, "に見つかりません。スキップします。"))
    next
  }
  g_1octave_data <- fread(g_1octave_file)
  g_1octave_data <- g_1octave_data[, c("Response", "Correct", "Offset", "CorrectResponse")]
  colnames(g_1octave_data) <- c("resp", "corr", "offset", "interval")
  
  if(nrow(g_1octave_data) < 5){
    print(paste("欠損データファイル（g_1octave）:", basename(g_1octave_file), "行数が5未満のため、処理をスキップします。"))
    next
  }
  
  g_1octave_data <- mutate(g_1octave_data, log_offset = log2(offset))
  g_1octave_data$corr <- as.numeric(g_1octave_data$corr)
  g_1octave_data <- na.omit(g_1octave_data)
  
  # g_3octaveデータの読み込みと前処理
  if (!file.exists(g_3octave_file)) {
    print(paste("対応するg_3octaveファイルが", g_3octave_file, "に見つかりません。スキップします。"))
    next
  }
  g_3octave_data <- fread(g_3octave_file)
  g_3octave_data <- g_3octave_data[, c("Response", "Correct", "Offset", "CorrectResponse")]
  colnames(g_3octave_data) <- c("resp", "corr", "offset", "interval")
  
  if(nrow(g_3octave_data) < 5){
    print(paste("欠損データファイル（g_3octave）:", basename(g_3octave_file), "行数が5未満のため、処理をスキップします。"))
    next
  }
  
  g_3octave_data <- mutate(g_3octave_data, log_offset = log2(offset))
  g_3octave_data$corr <- as.numeric(g_3octave_data$corr)
  g_3octave_data <- na.omit(g_3octave_data)
  
  # GLMフィッティング
  g_base_fit    <- glm(corr ~ log_offset, data = g_base_data, family = binomial(mafc.probit(2)), control = glm.control(maxit = 100, epsilon = 1e-8))
  g_1octave_fit <- glm(corr ~ log_offset, data = g_1octave_data, family = binomial(mafc.probit(2)), control = glm.control(maxit = 100, epsilon = 1e-8))
  g_3octave_fit <- glm(corr ~ log_offset, data = g_3octave_data, family = binomial(mafc.probit(2)), control = glm.control(maxit = 100, epsilon = 1e-8))
  
  # 75%閾値の計算関数
  calc_threshold <- function(model) {
    m <- -coef(model)[1] / coef(model)[2]
    std <- 1 / coef(model)[2]
    prob <- 0.75
    log_theta <- qnorm((prob - 0.5) / 0.5, m, std)
    theta <- 2 ** log_theta
    return(theta)
  }
  
  base_threshold    <- calc_threshold(g_base_fit)
  octave1_threshold <- calc_threshold(g_1octave_fit)
  octave3_threshold <- calc_threshold(g_3octave_fit)
  
  # 各条件の閾値をlog2変換
  base_log   <- log2(base_threshold)
  octave1_log <- log2(octave1_threshold)
  octave3_log <- log2(octave3_threshold)
  
  # baseと1octaveの差を正規化係数として使用
  norm_factor <- octave1_log - base_log
  
  # 正規化した上昇率を計算
  #ratio_base_3octave   <- (octave3_log - base_log) / norm_factor
  ratio_1octave_3octave <- (octave3_log - octave1_log) / norm_factor
  
  # フィッティングカーブ作成
  xseq <- seq(0.6, 100, len = 1000)
  g_base_curve    <- data.frame(xseq = xseq, yseq = predict(g_base_fit, data.frame(log_offset = log2(xseq)), type = "response"))
  g_1octave_curve <- data.frame(xseq = xseq, yseq = predict(g_1octave_fit, data.frame(log_offset = log2(xseq)), type = "response"))
  g_3octave_curve <- data.frame(xseq = xseq, yseq = predict(g_3octave_fit, data.frame(log_offset = log2(xseq)), type = "response"))
  
  # プロットタイトルの取得
  title <- get_title(file_base_name)
  
  # プロット作成
  g <- ggplot() + 
    geom_jitter(data = g_base_data, aes(x = offset, y = corr, color = "g_base"), alpha = 0.6, width = 0.05, height = 0.05) +
    geom_line(data = g_base_curve, aes(x = xseq, y = yseq, color = "g_base"), size = 1, linetype = "solid") +
    geom_jitter(data = g_1octave_data, aes(x = offset, y = corr, color = "g_1octave"), alpha = 0.6, width = 0.05, height = 0.05) +
    geom_line(data = g_1octave_curve, aes(x = xseq, y = yseq, color = "g_1octave"), size = 1, linetype = "solid") +
    geom_jitter(data = g_3octave_data, aes(x = offset, y = corr, color = "g_3octave"), alpha = 0.6, width = 0.05, height = 0.05) +
    geom_line(data = g_3octave_curve, aes(x = xseq, y = yseq, color = "g_3octave"), size = 1, linetype = "solid") +
    scale_x_continuous(trans = "log2", breaks = c(1, 2, 4, 8, 16, 32, 64)) +
    ylim(0, 1) +
    theme_classic() +
    xlab("ターゲット刺激のB音のズレ幅[ms]") + 
    ylab("正答率") +
    #ggtitle(title) +
    geom_vline(xintercept = base_threshold, color = "blue", linetype = "dashed", size = 0.8) +
    geom_vline(xintercept = octave1_threshold, color = "orange", linetype = "dashed", size = 0.8) +
    geom_vline(xintercept = octave3_threshold, color = "darkgreen", linetype = "dashed", size = 0.8) +
    scale_color_manual(
      name = "周波数条件",
      values = c("g_base" = "blue", "g_1octave" = "orange", "g_3octave" = "darkgreen"),
      labels = c("g_base" = "同じ音　　　", "g_1octave" = "1オクターブ", "g_3octave" = "3オクターブ")
    ) +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5)
    ) +
    # threshold値をそれぞれの色に合わせてプロット上に表示する
    annotate("text", x = 0.6, y = 0.4, 
             label = paste0("0octave: ", round(base_threshold, 2)), 
             color = "blue", hjust = 0, size = 5) +
    annotate("text", x = 0.6, y = 0.3, 
             label = paste0("1octave: ", round(octave1_threshold, 2)), 
             color = "orange", hjust = 0, size = 5) +
    annotate("text", x = 0.6, y = 0.2, 
             label = paste0("3octave: ", round(octave3_threshold, 2)), 
             color = "darkgreen", hjust = 0, size = 5) +
    # 新たな統計量（正規化した上昇率）の表示
    annotate("text", x = 0.6, y = 1.0, 
             label = paste0("0→1octave: 1"),
             color = "black", hjust = 0, size = 4) +
    annotate("text", x = 0.6, y = 0.9, 
             label = paste0("1→3octave: ", round(ratio_1octave_3octave, 2)), 
             color = "black", hjust = 0, size = 4) 

  
  # プロットの保存
  output_file <- file.path(output_dir, paste0("combined_", file_base_name, "_pastel", ".png"))
  ggsave(output_file, plot = g, width = 7, height = 3)
  
  print(paste("プロットを保存しました:", file_base_name))
}
