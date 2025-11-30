from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import datetime
import random
import csv
import pandas as pd
from analysis_MLE_v2 import perform_mle_analysis
from add_from_summary import add_from_summary

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


# 音声ファイルのディレクトリとプレフィックスを取得
def get_audio_settings(sound_type):
    if sound_type == 'piano':
        return 'piano_successive_0.52*4', 'piano_gallops_'
    else:  # pure_tone
        return 'successive_0.52*4', 'gallops_'

# サマリーファイルに実験結果を保存
def save_summary_file(results_list):
    today = session.get('today')
    participant_id = session.get('participant_id')
    sound_type = session.get('sound_type', 'pure_tone')
    
    # サマリーファイルのパスを設定
    summary_dir = os.path.join(DATA_FOLDER, today, participant_id)
    os.makedirs(summary_dir, exist_ok=True)
    
    base_filename = f"{participant_id}_summary"
    summary_file_path = os.path.join(summary_dir, f"{base_filename}.csv")
    
    # 既存のファイルがある場合、名前を変更
    file_index = session.get('data_file_index', 0)
    if file_index > 0:
        summary_file_path = os.path.join(summary_dir, f"{base_filename}_{file_index}.csv")
    
    # サマリーファイルを作成
    with open(summary_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # ヘッダー行を書き込み（sub_valueも含める）
        writer.writerow([
            'participant_id', 'experiment_date', 'sound_type',
            'frequency_condition', 'frequency_label', 
            'threshold_ms', 'log2_threshold', 'level', 'sub_value'
        ])
        
        # 各結果を書き込み（sub_valueは未回答時は空白）
        for result in results_list:
            writer.writerow([
                participant_id,
                today,
                sound_type,
                result['freq_key'],
                result['frequency_label'],
                result['threshold'],
                result['log2_threshold'],
                result['level'],
                ''  # sub_valueは後でアンケート回答時に更新
            ])
    
    print(f"Summary file saved: {summary_file_path}")

# アンケート詳細結果を別途CSVに保存
def save_questionnaire_details(questionnaire_data):
    """アンケートの詳細回答を別途CSVファイルに保存"""
    today = session.get('today')
    participant_id = session.get('participant_id')
    
    if not today or not participant_id:
        print(f"Error: Missing session data for questionnaire details - today: {today}, participant_id: {participant_id}")
        return
    
    # アンケート詳細用のディレクトリとファイルパス
    details_dir = os.path.join(DATA_FOLDER, today, participant_id)
    os.makedirs(details_dir, exist_ok=True)
    
    details_file_path = os.path.join(details_dir, f"{participant_id}_questionnaire_details.csv")
    
    # CSVヘッダー
    headers = ['participant_id', 'timestamp', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'total_score']
    
    # 現在の時刻とデータを準備
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_score = sum(int(questionnaire_data[f'q{i}']) for i in range(1, 8))
    
    # データ行
    data_row = [
        participant_id,
        current_time,
        questionnaire_data.get('q1', ''),
        questionnaire_data.get('q2', ''),
        questionnaire_data.get('q3', ''),
        questionnaire_data.get('q4', ''),
        questionnaire_data.get('q5', ''),
        questionnaire_data.get('q6', ''),
        questionnaire_data.get('q7', ''),
        total_score
    ]
    
    try:
        # ファイルが存在しない場合はヘッダー付きで新規作成
        file_exists = os.path.exists(details_file_path)
        
        with open(details_file_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # ヘッダーを書き込み（新規ファイルの場合のみ）
            if not file_exists:
                writer.writerow(headers)
            
            # データ行を追加
            writer.writerow(data_row)
        
        print(f"Questionnaire details saved: {details_file_path}")
        print(f"Individual answers: {[questionnaire_data.get(f'q{i}') for i in range(1, 8)]}")
        
    except Exception as e:
        print(f"Error saving questionnaire details: {e}")

# アンケート結果でサマリーファイルを更新
# グローバルsummary.csvは使用しない方針のため、関連処理は削除

def update_summary_with_questionnaire(questionnaire_data):
    today = session.get('today')
    participant_id = session.get('participant_id')
    
    # セッション情報の確認
    if not today or not participant_id:
        print(f"Error: Missing session data - today: {today}, participant_id: {participant_id}")
        # 最新のディレクトリを探して取得
        today = today or datetime.datetime.now().strftime('%Y-%m-%d')
        
        # 最近のサマリーファイルを検索
        import glob
        summary_pattern = os.path.join(DATA_FOLDER, '*', '*', '*_summary*.csv')
        summary_files = glob.glob(summary_pattern)
        if summary_files:
            # 最新のファイルを取得
            latest_file = max(summary_files, key=os.path.getmtime)
            print(f"Found latest summary file: {latest_file}")
            summary_file_path = latest_file
        else:
            print("No summary files found")
            return
    else:
        # 既存のサマリーファイルのパスを取得
        summary_dir = os.path.join(DATA_FOLDER, today, participant_id)
        base_filename = f"{participant_id}_summary"
        summary_file_path = os.path.join(summary_dir, f"{base_filename}.csv")
        
        file_index = session.get('data_file_index', 0)
        if file_index > 0:
            summary_file_path = os.path.join(summary_dir, f"{base_filename}_{file_index}.csv")
    
    # 既存のサマリーファイルを読み込み
    if os.path.exists(summary_file_path):
        try:
            # アンケート結果の合計点数を計算
            total_score = sum(int(questionnaire_data[f'q{i}']) for i in range(1, 8))
            print(f"Questionnaire answers: {questionnaire_data}")
            print(f"Questionnaire total score: {total_score}")
            
            # CSVファイルを読み込み
            rows = []
            with open(summary_file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader)  # ヘッダー行を読み込み
                rows.append(header)
                
                # データ行を読み込み、sub_value列を更新
                for row in reader:
                    if len(row) >= 9:  # sub_value列が存在することを確認
                        row[8] = str(total_score)  # sub_value列を更新（0ベースで8番目）
                    rows.append(row)
            
            # ファイルを上書き保存
            with open(summary_file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
            
            print(f"Summary file updated with questionnaire results: {summary_file_path}")
            print(f"Updated sub_value with total score: {total_score}")
            
        except Exception as e:
            print(f"Error updating summary file with questionnaire: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Summary file not found: {summary_file_path}")

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
        # 既存のファイルがある場合、名前をつける
        file_index = 1
        while os.path.exists(data_file_path):
            data_file_path = os.path.join(data_dir, f"{base_filename}_{file_index}.csv")
            session['data_file_index'] = file_index #重複がある場合のインデックス
            file_index += 1
        # 新しいデータファイルを作成
        with open(data_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trial', 'CorrectResponse', 'Response', 'Correct', 'Offset', 'NextStepSize', 'Reversals', 'NextDirection', 'SameDirectionCount'])
        # figディレクトリの作成
        fig_dir = os.path.join('static', FIG_FOLDER, today, participant_id, freq)
        os.makedirs(fig_dir, exist_ok=True)

        if mail_address and mail_address.strip():
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

    data = request.get_json()
    print("Received JSON data:", data)
    
    participant_id = data.get('participant_id')
    sound_type = data.get('sound_type', 'pure_tone')  # デフォルトは純音
    frequency_dirs = data.get('frequency_dirs')
    trials_per_cond = data.get('trials_per_cond', 20)
    mail_address = data.get('mail_address')

    session['participant_id'] = participant_id
    session['sound_type'] = sound_type
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

@app.route('/demo')
def demo():
    # Simple demo page (English) independent of session state
    return render_template('demo.html')

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
    file_index = session.get('data_file_index', 0)
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


    # フィードバックを返す
    current_offset = OFFSET_LIST[current_offset_index]
    next_offset = OFFSET_LIST[next_offset_index]

    # **フィードバックメッセージの決定**
    feedback_message = ""
    level_feedback = ""
    
    # レベル変化があった場合のフィードバック
    if float(next_offset) < float(current_offset):
        level_feedback = "Level UP🔥"
    elif float(next_offset) > float(current_offset):
        level_feedback = "Level DOWN💧"
    # レベル変化なしの場合は何も表示しない（level_feedbackは空文字のまま）
    
    # 正解/不正解のフィードバック
    if correct:
        feedback_message = "正解"
    else:
        feedback_message = "ざんねん"

    return jsonify({
        "message": "応答受領",
        "correct": correct,
        "current_offset": current_offset,
        "next_offset": next_offset,
        "completed": session['block_trial_count'] >= session['num_block_trials'], # ここでブロック間の休憩に行くか判定
        "feedback": feedback_message,
        "level_feedback": level_feedback
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
    # 既にMLE結果がセッションにあれば、それを使って再計算を抑制
    cached_results = session.get('mle_results_cached')
    cached_today = session.get('today')
    cached_participant = session.get('participant_id')
    if cached_results and cached_today and cached_participant:
        results_list = cached_results
    else:
        results_list = []

    # セッションから選択した周波数条件リストを取得
    frequency_dirs = session.get('frequency_dirs', [])
    if not frequency_dirs and not results_list:
        # セッションが失われている場合、最新のデータディレクトリを検索
        import glob
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # 最新の参加者ディレクトリを検索
        participant_dirs = glob.glob(os.path.join(DATA_FOLDER, today, '*'))
        if not participant_dirs:
            return "Error: No experiment data found for today.", 400
        
        # 最新の参加者ディレクトリを取得
        latest_participant_dir = max(participant_dirs, key=os.path.getmtime)
        participant_id = os.path.basename(latest_participant_dir)
        
        # その参加者の周波数条件を検索
        freq_dirs = [d for d in os.listdir(latest_participant_dir) 
                     if os.path.isdir(os.path.join(latest_participant_dir, d)) and d != '__pycache__']
        
        if not freq_dirs:
            return "Error: No frequency data found.", 400
        
        # セッションに設定
        session['frequency_dirs'] = freq_dirs
        session['today'] = today
        session['participant_id'] = participant_id
        frequency_dirs = freq_dirs

    # 以降で使う日付/参加者IDを必ず確保
    today = session.get('today')
    participant_id = session.get('participant_id')

    # まだキャッシュがない場合のみMLE分析を実行
    if not results_list:
        # MLE分析を各周波数条件ごとに実行
        today = session.get('today')
        participant_id = session.get('participant_id')
        for freq in frequency_dirs:
            # **データファイルのパスを統一**
            base_filename = f"{participant_id}_{freq}_results"
            file_index = session.get('data_file_index', 0)

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
            log2_threshold = results.get("log2_threshold")
            if not fig_path or threshold is None or log2_threshold is None:
                print(f"Error: Missing results for {freq} ({data_file_path})")
                continue

            # レベル判定
            def get_level(freq_key, log2_thresh):
                if freq_key in ['g_base', 'as_semitone']:
                    if log2_thresh <= 2:
                        return "天才！🌟"
                    elif log2_thresh <= 2.5:
                        return "エキスパート🎯"
                    elif log2_thresh <= 3:
                        return "スキルド💪"
                    else:
                        return "ファイター⚡"
                elif freq_key == 'g_1octave':
                    if log2_thresh <= 2.5:
                        return "天才！🌟"
                    elif log2_thresh <= 3.0:
                        return "エキスパート🎯"
                    elif log2_thresh <= 3.5:
                        return "スキルド💪"
                    else:
                        return "ファイター⚡"
                elif freq_key in ['g_2octave', 'g_3octave']:
                    if log2_thresh <= 3:
                        return "天才！🌟"
                    elif log2_thresh <= 4:
                        return "エキスパート🎯"
                    elif log2_thresh <= 5:
                        return "スキルド💪"
                    else:
                        return "ファイター⚡"
                else:
                    return "判定不能"

            level = get_level(freq, log2_threshold)
            
            # 結果をリストに追加
            results_list.append({
                "frequency_label": FREQUENCY_LABELS.get(freq, freq),
                "freq_key": freq,  # レベル判定用のキー
                "fig_path": fig_path.replace('static/', ''),
                "threshold": f"{threshold:.2f}",
                "log2_threshold": f"{log2_threshold:.2f}",
                "level": level,
                "file_name": os.path.basename(data_file_path)
            })

        # MLE結果をキャッシュ（アンケート後のcompleteでも再計算しない）
        session['mle_results_cached'] = results_list
        session.modified = True

    # **MLE分析が1つも成功しなかった場合**
    if not results_list:
        return "Error: No valid MLE analysis results.", 500
    
    # **サマリーファイルが存在しない場合のみ保存**
    today = session.get('today')
    participant_id = session.get('participant_id')
    file_index = session.get('data_file_index', 0)
    
    if today and participant_id:
        base_filename = f"{participant_id}_summary"
        summary_file_path = os.path.join(DATA_FOLDER, today, participant_id, f"{base_filename}.csv")
        if file_index > 0:
            summary_file_path = os.path.join(DATA_FOLDER, today, participant_id, f"{base_filename}_{file_index}.csv")
        
        if not os.path.exists(summary_file_path):
            save_summary_file(results_list)
    
    # アンケート結果の統一チェック（キャッシュ利用）。アンケート送信後のデータがある場合のみ作成。
    has_questionnaire = False
    questionnaire_score = 0
    survey_figure_path = None
    survey_results = None

    # アンケート完了判定：
    # 1) セッションに questionnaire_answers がある
    # 2) または summary.csv に sub_value が埋まっている
    questionnaire_done = bool(session.get('questionnaire_answers'))
    individual_summary_path = None
    if not questionnaire_done and results_list:
        file_index = session.get('data_file_index', 0)
        if file_index > 0:
            individual_summary_path = os.path.join(DATA_FOLDER, today, participant_id, f"{participant_id}_summary_{file_index}.csv")
        else:
            individual_summary_path = os.path.join(DATA_FOLDER, today, participant_id, f"{participant_id}_summary.csv")
        if os.path.exists(individual_summary_path):
            try:
                sdf = pd.read_csv(individual_summary_path)
                if 'sub_value' in sdf.columns and sdf['sub_value'].astype(str).str.strip().ne('').any():
                    questionnaire_done = True
            except Exception:
                pass

    if questionnaire_done:
        survey_cached = session.get('survey_cached')
        if survey_cached and survey_cached.get('today') == today and survey_cached.get('participant_id') == participant_id:
            # セッションに比較図があれば再計算しない
            has_questionnaire = survey_cached.get('has_questionnaire', False)
            questionnaire_score = survey_cached.get('questionnaire_score', 0)
            survey_figure_path = survey_cached.get('survey_figure_path')
            survey_results = None  # 図は再利用。詳細メトリクスは再計算しない
        elif results_list:
            # 個別のsummary.csvパスを構築（未取得なら改めて作る）
            if individual_summary_path is None:
                file_index = session.get('data_file_index', 0)
                if file_index > 0:
                    individual_summary_path = os.path.join(DATA_FOLDER, today, participant_id, f"{participant_id}_summary_{file_index}.csv")
                else:
                    individual_summary_path = os.path.join(DATA_FOLDER, today, participant_id, f"{participant_id}_summary.csv")

            # アンケート結果がある場合のみ add_from_summary 実行
            if os.path.exists(individual_summary_path):
                try:
                    # 各被験者のディレクトリに図を保存
                    participant_dir = os.path.join('static', 'fig', today, participant_id)
                    os.makedirs(participant_dir, exist_ok=True)

                    sound_type = session.get('sound_type', 'pure_tone')
                    print(f"Attempting unified survey analysis for participant: {participant_id}, sound_type: {sound_type}")
                    print(f"Using individual summary file: {individual_summary_path}")

                    survey_results = add_from_summary(
                        participant_id,
                        summary_csv_path=individual_summary_path,
                        sound_type=sound_type,
                        output_dir=participant_dir
                    )

                    # アンケート結果の有無を統一判定
                    if survey_results and survey_results.get('metrics_by_group'):
                        first_metric = survey_results['metrics_by_group'][0] if survey_results['metrics_by_group'] else {}
                        if 'sub_value' in first_metric and first_metric['sub_value'] is not None:
                            questionnaire_score = int(first_metric['sub_value'])
                            has_questionnaire = True
                            survey_figure_path = survey_results.get('fig_path', '').replace('static/', '')
                            print(f"Survey analysis completed: {survey_figure_path}")
                            print(f"Questionnaire score: {questionnaire_score}")

                    # 比較図が得られた場合のみキャッシュ
                    session['survey_cached'] = {
                        'today': today,
                        'participant_id': participant_id,
                        'has_questionnaire': has_questionnaire,
                        'questionnaire_score': questionnaire_score,
                        'survey_figure_path': survey_figure_path
                    }
                    session.modified = True

                except FileNotFoundError as e:
                    print(f"FileNotFoundError in survey analysis: {e}")
                except ValueError as e:
                    print(f"ValueError in survey analysis: {e}")
                    survey_results = None
                except Exception as e:
                    print(f"Unexpected error generating survey analysis: {e}")
                    import traceback
                    traceback.print_exc()

    # **`complete.html` にリストを渡す**
    return render_template('complete.html', 
                         results_list=results_list,
                         has_questionnaire=has_questionnaire,
                         questionnaire_score=questionnaire_score,
                         survey_figure_path=survey_figure_path,
                         survey_results=survey_results)


@app.route('/questionnaire')
def questionnaire():
    # アンケートページを表示
    return render_template('questionnaire.html')

@app.route('/submit_questionnaire', methods=['POST'])
def submit_questionnaire():
    # アンケート回答を処理
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'error': 'No data received'}), 400
    
    # セッションにアンケート結果を保存
    session['questionnaire_answers'] = data
    
    # アンケート詳細結果を別途CSVに保存
    save_questionnaire_details(data)
    
    # サマリーファイルを更新（アンケート結果を含む）
    update_summary_with_questionnaire(data)
    
    # 比較図キャッシュを無効化（次回completeで最新を生成）
    if 'survey_cached' in session:
        session.pop('survey_cached')
        session.modified = True
    
    return jsonify({'success': True})

@app.route('/debug_session')
def debug_session():
    return jsonify(dict(session))


# 開発用のFlaskサーバーを起動
# gunicornを使う場合は関係ない
if __name__ == '__main__':
    app.run(debug=True, port=5001)
