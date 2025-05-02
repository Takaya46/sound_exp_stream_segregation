from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import datetime
import random
import csv
from analysis_MLE_v2 import perform_mle_analysis

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# フォルダ設定
STIMULI_ROOT = os.path.join('static', 'stimuli', 'successive_0.52*4')
DATA_FOLDER = 'data'
FIG_FOLDER = 'fig'
os.makedirs(DATA_FOLDER, exist_ok=True)

# Offsetリストの設定
OFFSET_LIST = [f"{(2 ** (i / 2)):.2f}" for i in range(13)]
OFFSET_LIST.reverse()
FREQUENCY_CONDITIONS_ORDERED = ['g_base', 'as_semitone', 'g_1octave', 'g_2octave', 'g_3octave']
FREQUENCY_LABELS = {
    "g_base": "条件１（同じ音）",
    "as_semitone": "条件２（半音）",
    "g_1octave": "条件３（１オクターブ）",
    "g_2octave": "条件４（２オクターブ）",
    "g_3octave": "条件５（３オクターブ）"
}

# 各周波数条件の実験パラメータ初期化関数 (freq_cond_param を初期化する関数に変更)
def initialize_condition_session(frequency_dir):
    return {
        'offset_index': 0,  # OFFSET_LIST の初期インデックス
        'current_direction': 'down', # 'down', 'up', or None
        'next_direction': None,
        'same_direction_count': 0,
        'step_size': 4,
        'double_flag': False,
        'reversals_double_flag': True,
        'max_step_size': 4, # step_sizeが大きくなりすぎないようにlimitを設ける
        'correct_count': 0,
        'reversals': 0,
        'freq_cond_trial_count': 0, # 周波数条件内でのトライアル数をカウント
    }


# データ保存用のファイル、figのディレクトリを設定
def set_data_file_path(freq_list):
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    session['today'] = today
    participant_id = session['participant_id']
    session['data_file_index'] = 0
    mail_address = session['mail_address']

    for freq in freq_list:
        # データファイルの作成
        data_dir = os.path.join(DATA_FOLDER, today, participant_id, freq)
        os.makedirs(data_dir, exist_ok=True)
        # ファイル名の決定
        base_filename = f"{participant_id}_{freq}_results"
        data_file_path = os.path.join(data_dir, f"{base_filename}.csv")
        # 既存のファイルがある場合、新しい名前をつける
        file_index = 1
        while os.path.exists(data_file_path):
            data_file_path = os.path.join(data_dir, f"{base_filename}_{file_index}.csv")
            session['data_file_index'] = file_index #重複がある場合のインデックス
            file_index += 1
        # 新しいデータファイルを作成
        with open(data_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trial', 'CorrectResponse', 'Response', 'Correct', 'Offset', 'NextStepSize', 'Reversals', 'NextDirection', 'SameDirectionCount'])
        print(f"データファイル作成: {data_file_path}")  # デバッグ用
        # figディレクトリの作成
        fig_dir = os.path.join('static', FIG_FOLDER, today, participant_id, freq)
        os.makedirs(fig_dir, exist_ok=True)

        if mail_address:
            data_mail_path = os.path.join(DATA_FOLDER, today, participant_id, f"{mail_address}.txt")
            with open(data_mail_path, 'w') as f:
                f.write(f'mail: {mail_address}\n')

# ステップサイズの決定をする関数
def decide_step_size():
    freq = session['current_block_freq'] # 現在の周波数条件
    step_size = session['freq_cond_param'][freq]['step_size'] # 現在の条件の step_size

    # 反転が発生した場合
    if  session['freq_cond_param'][freq]['current_direction'] != session['freq_cond_param'][freq]['next_direction']:
        session['freq_cond_param'][freq]['reversals'] += 1  # 反転回数をカウント
        session['freq_cond_param'][freq]['reversals_double_flag'] = session['freq_cond_param'][freq]['double_flag']  # 反転の直前のフラグ記録
        session['freq_cond_param'][freq]['double_flag'] = False  # フラグリセット
        step_size = max(step_size // 2, 1)  # step_sizeを半分に、最小は1
        session['freq_cond_param'][freq]['same_direction_count'] = 0  # 同じ方向のカウントをリセット
        print(f"反転が発生！ Step Sizeを半分に: {step_size}")

    # 反転しなかった場合
    else:
        session['freq_cond_param'][freq]['same_direction_count'] += 1
        if session['freq_cond_param'][freq]['same_direction_count'] >= 3:  # 同じ方向に4回以上連続
            step_size = min(step_size * 2, session['freq_cond_param'][freq]['max_step_size'])# ステップサイズを2倍に、最大はmax_step_size
            session['freq_cond_param'][freq]['double_flag'] = True
            print(f"4回以上同じ方向 Step Size倍: {step_size}")
        elif session['freq_cond_param'][freq]['same_direction_count'] == 2:  # 同じ方向に3回連続
            if session['freq_cond_param'][freq]['reversals_double_flag']:  # 直近の反転前のステップが倍であればそのまま
                step_size = step_size
            else:
                step_size = min(step_size * 2, session['freq_cond_param'][freq]['max_step_size'])# ステップサイズを2倍に、最大はmax_step_size
                session['freq_cond_param'][freq]['double_flag'] = True
                print(f"3回連続同じ方向 Step Size倍: {step_size}")
        else:  # 同じ方向に2回連続
            step_size = step_size
            print(f"2回連続同じ方向 Step Sizeそのまま: {step_size}")
    
    # ステップサイズの最大値を適用
    step_size = min(step_size, session['freq_cond_param'][freq]['max_step_size'])

    # セッション変数を更新
    session['freq_cond_param'][freq]['step_size'] = step_size
    print(f"Step Size: {step_size}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start_experiment():
    session.clear()

    # JSONのデバッグ
    data = request.get_json()
    print("Received JSON data:", data)
    # if not data:
        # return jsonify({'error': 'No data received'}), 400
    # キーの確認
    participant_id = data.get('participant_id')
    frequency_dirs = data.get('frequency_dirs')
    trials_per_cond = data.get('trials_per_cond', 20)
    mail_address = data.get('mail_address')
    # if not participant_id or not frequency_dirs:
    #     return jsonify({'error': 'Missing required data'}), 400

    session['participant_id'] = participant_id
    session['frequency_dirs'] = frequency_dirs # frequency_dir のリスト
    session['mail_address'] = mail_address

    selected_frequency_dirs = [fre_dir for fre_dir in frequency_dirs] #例['g_base', 'as_semitone', 'g_1octave', 'g_2octave', 'g_3octave'] 

    # データ保存用のファイル、figのpathを設定
    set_data_file_path(selected_frequency_dirs)

    # 周波数条件ごとにトライアルのsession変数を設定し初期化
    session['freq_cond_param'] = {}
    for freq in selected_frequency_dirs:
        session['freq_cond_param'][freq] = initialize_condition_session(freq)
    
    # 条件をブロックに分割し順序を決定**
    frequency_dirs_in_order = []
    trials_per_block = int(trials_per_cond / 2) # 各条件を2つのブロックに分ける。ブロックごとのトライアル数
    # 前半: 難易度順
    phase1_conditions = [cond for cond in FREQUENCY_CONDITIONS_ORDERED if cond in selected_frequency_dirs]
    for block in phase1_conditions:
        frequency_dirs_in_order.append({'frequency_dir': block, 'trials': trials_per_block})
    # 後半: 難易度逆順
    phase2_conditions = [cond for cond in reversed(FREQUENCY_CONDITIONS_ORDERED) if cond in selected_frequency_dirs]
    for block in phase2_conditions:
        frequency_dirs_in_order.append({'frequency_dir': block, 'trials': trials_per_block})
    # セッションに保存
    session['frequency_dirs_in_order'] = frequency_dirs_in_order

    # 1番初めのブロックのデータをセッションに保存
    session['block_index'] = 0 # frequency_dirs_in_orderのインデックス.現在どのブロックかを示す
    session['total_blocks'] = len(session['frequency_dirs_in_order'])  # ブロック総数を保存
    session['current_block_data'] = frequency_dirs_in_order[0] # 最初の条件(block_index=0)で, {'frequency_dir': 'g_base', 'trials': 10}の形式
    session['current_block_freq'] = session['current_block_data']['frequency_dir'] #'g_base'の形式
    session['num_block_trials'] = trials_per_block # 1ブロックのトライアル数(int)
    session['block_trial_count'] = 0 # ブロック内のトライアル数をカウント

    # sessionを更新してjsonを返す
    session.modified = True
    print("Session data:", dict(session))
    return jsonify({'status': 'success'})

@app.route('/practice')
def practice():
    return render_template('practice.html', frequency_labels=FREQUENCY_LABELS)

@app.route('/experiment')
def experiment():
    return render_template('experiment.html', frequency_labels=FREQUENCY_LABELS)


@app.route('/next_trial', methods=['POST', 'GET'])
def next_trial():
    freq = session['current_block_freq'] # 現在の周波数条件
    offset_index = session['freq_cond_param'][freq]['offset_index'] # 現在のoffset_index

    # トライアルデータ生成
    if random.choice([True, False]):
        trial = {
            'Tone1': '0.00',
            'Tone2': '0.00',
            'Tone3': OFFSET_LIST[offset_index],
            'CorrectResponse': '3'  # 3番目がターゲット
        }
    else:
        trial = {
            'Tone1': OFFSET_LIST[offset_index],
            'Tone2': '0.00',
            'Tone3': '0.00',
            'CorrectResponse': '1'  # 1番目がターゲット
        }

    # トライアルデータをここだけで使うセッション変数(current_trial_data)に保存
    session['current_trial_data'] = trial

    # トライアル数をカウント
    session['block_trial_count'] += 1

    # デバッグ出力
    print(f"Condition: {freq}, Trial: {session['freq_cond_param'][freq]['freq_cond_trial_count']}/{session['num_block_trials']}, Offset Index: {offset_index}, Trial Data: {trial}")

    return jsonify({
        "status": 'trial_data',
        "tones": [trial['Tone1'], trial['Tone2'], trial['Tone3']],
        "correct_response": trial['CorrectResponse'],
        "frequency_dir": freq,
        "trial_number": session['freq_cond_param'][freq]['freq_cond_trial_count'],
        "offset_index_display": OFFSET_LIST.index(OFFSET_LIST[offset_index]) # offset_index は session['freq_cond_param'][freq] から取得
    })

@app.route('/submit_response', methods=['POST'])
def submit_response():
    trial = session['current_trial_data']# セッションからトライアルデータを取得
    # クライアントからの応答を取得
    response = request.json['response']

    freq = session['current_block_freq'] # 現在の周波数条件
    correct_response = session['current_trial_data']['CorrectResponse'] #'1'か'3'のどちらか正解の方
    correct = (response == correct_response)  # 正誤判定
    
    current_offset_index= session['freq_cond_param'][freq]['offset_index'] # 現在のoffset
    current_offset = OFFSET_LIST[current_offset_index]  # 現在のオフセットを記録する

    # 正解した場合の処理
    if correct:
        session['freq_cond_param'][freq]['correct_count'] += 1
        print(f"Correct!, current correct count: {session['freq_cond_param'][freq]['correct_count']}")
        next_offset_index = current_offset_index#必要
        if session['freq_cond_param'][freq]['correct_count'] == 2: # 2回目の正解でoffset_indexを増やす
            print("Correct twice!")
            session['freq_cond_param'][freq]['correct_count'] = 0 # 正解カウントリセット
            session['freq_cond_param'][freq]['next_direction'] = 'down'
            decide_step_size()
            next_offset_index = min(current_offset_index + session['freq_cond_param'][freq]['step_size'], len(OFFSET_LIST) - 1)
            session['freq_cond_param'][freq]['offset_index'] = next_offset_index # 更新
            print(f"Offset Index increased to {next_offset_index}")
            # 方向の更新
            session['freq_cond_param'][freq]['current_direction'] = 'down'
    # 間違えた場合の処理
    else:
        session['freq_cond_param'][freq]['correct_count'] = 0 # 正解カウントリセット
        session['freq_cond_param'][freq]['next_direction'] = 'up'
        decide_step_size()
        next_offset_index = max(current_offset_index - session['freq_cond_param'][freq]['step_size'], 0) 
        session['freq_cond_param'][freq]['offset_index'] = next_offset_index # 更新
        print(f"Incorrect. Offset Index decreased to {next_offset_index}")
        # 方向の更新
        session['freq_cond_param'][freq]['current_direction'] = 'up'

    # session変数の更新
    session['freq_cond_param'][freq]['freq_cond_trial_count'] += 1 # freq_cond_trial_count を増やす
    session.update()  # セッションを更新

    # データファイルのパス
    today = session['today']
    participant_id = session['participant_id']
    base_filename = f"{participant_id}_{freq}_results"
    data_file_path = os.path.join(DATA_FOLDER, today, participant_id, freq, f"{base_filename}.csv")
    # 既存のファイルがある場合のファイルパス
    file_index = session['data_file_index']
    if file_index > 0:
        data_file_path = os.path.join(DATA_FOLDER, today, participant_id, freq, f"{base_filename}_{file_index}.csv")
    # データ保存
    try: 
        with open(data_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                session['freq_cond_param'][freq]['freq_cond_trial_count'], correct_response, response, correct,
                current_offset,  # 保存するのは現在のトライアルのオフセット
                session['freq_cond_param'][freq]['step_size'], session['freq_cond_param'][freq]['reversals'], session['freq_cond_param'][freq]['next_direction'], session['freq_cond_param'][freq]['same_direction_count']
            ])
    except Exception as e:
        print("データ保存中にエラー:", e)
        return jsonify({"error": "データの保存中にエラーが発生しました"}), 500

    # デバッグ用出力
    print(f"Trial {session['freq_cond_param'][freq]['freq_cond_trial_count']} saved: Response = {response}, Correct = {correct}, Offset = {current_offset}")

    # フィードバックを返す
    current_offset = OFFSET_LIST[current_offset_index]
    next_offset = OFFSET_LIST[next_offset_index]

    # **レベルアップ or レベルダウンの判定**
    feedback_message = "Stay"
    if float(next_offset) < float(current_offset):
        feedback_message = "Level UP🔥"
    elif float(next_offset) > float(current_offset):
        feedback_message = "Level DOWN💧"

    return jsonify({
        "message": "応答受領",
        "correct": correct,
        "current_offset": current_offset,
        "next_offset": next_offset,
        "completed": session['block_trial_count'] >= session['num_block_trials'], # ここでブロック間の休憩に行くか判定
        "feedback": feedback_message
    })

@app.route('/next_block', methods=['GET'])
def next_block():
    # 次の条件へ移行
    session['block_trial_count'] = 0
    session['block_index'] += 1
    # 次の条件がない場合は終了
    if session['block_index'] >= session['total_blocks']:
        return redirect(url_for('complete'))
    # 次の条件のデータをセッションに保存
    session['current_block_data'] = session['frequency_dirs_in_order'][session['block_index']]
    session['current_block_freq'] = session['frequency_dirs_in_order'][session['block_index']]["frequency_dir"]
    # 休憩ページにリダイレクト
    return redirect(url_for('break_page'))


@app.route('/break_page')
def break_page():
    return render_template('break_page.html', frequency_labels=FREQUENCY_LABELS)


@app.route('/complete')
def complete():
    # セッションから選択した周波数条件リストを取得
    frequency_dirs = session.get('frequency_dirs', [])
    if not frequency_dirs:
        return "Error: No frequency directories found in session.", 400

    # MLE分析結果を格納するリスト
    results_list = []
    # MLE分析を各周波数条件ごとに実行
    today = session.get('today')
    participant_id = session.get('participant_id')
    for freq in frequency_dirs:
        # **データファイルのパスを統一**
        base_filename = f"{participant_id}_{freq}_results"
        file_index = session['data_file_index']

        if file_index > 0:
            data_file_path = os.path.join(DATA_FOLDER, today, participant_id, freq, f"{base_filename}_{file_index}.csv")
        else:
            data_file_path = os.path.join(DATA_FOLDER, today, participant_id, freq, f"{base_filename}.csv")
        # **データファイルが存在しない場合はスキップ**
        if not os.path.exists(data_file_path):
            print(f"Warning: Data file {data_file_path} not found. Skipping...")
            continue

        fig_dir = os.path.join('static', FIG_FOLDER, today, participant_id, freq)
        os.makedirs(fig_dir, exist_ok=True)  # フォルダがない場合は作成
        # **MLE分析の実行**
        results = perform_mle_analysis(data_file_path, fig_dir)
        # エラーがあればスキップ
        if "error" in results:
            print(f"Error in MLE analysis for {freq} ({data_file_path}): {results['error']}")
            continue
        # `fig_path` と `threshold` の存在チェック
        fig_path = results.get("fig_path")
        threshold = results.get("threshold")
        if not fig_path or threshold is None:
            print(f"Error: Missing results for {freq} ({data_file_path})")
            continue

        # 結果をリストに追加
        results_list.append({
            "frequency_label": FREQUENCY_LABELS.get(freq, freq),  # ラベルがない場合はそのまま表示
            "fig_path": fig_path.replace('static/', ''),  # `static/` を取り除く
            "threshold": f"{threshold:.2f}",
            "file_name": os.path.basename(data_file_path)  # どのデータファイルの結果かを表示
        })

    # **MLE分析が1つも成功しなかった場合**
    if not results_list:
        return "Error: No valid MLE analysis results.", 500
    # **`complete.html` にリストを渡す**
    return render_template('complete.html', results_list=results_list)


@app.route('/debug_session')
def debug_session():
    return jsonify(dict(session))


# 開発用のFlaskサーバーを起動
# gunicornを使う場合は関係ない
if __name__ == '__main__':
    app.run(debug=True)

