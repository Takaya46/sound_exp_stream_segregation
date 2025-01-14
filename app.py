from flask import Flask, render_template, request, jsonify, session
import os
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

# 実験初期化
def initialize_experiment(frequency_dir):
    session['frequency_dir'] = frequency_dir
    session['offset_index'] = 0
    session['step_size'] = 4
    session['max_step_size'] = 4
    session['correct_count'] = 0
    session['trial_count'] = 0
    session['reversals'] = 0
    session['current_direction'] = 'down'
    session['next_direction'] = None
    session['same_direction_count'] = 0
    session['step_size'] = 4
    session['double_flag'] = False
    session['reversals_double_flag'] = False

def decide_step_size():
    step_size = session['step_size']
    offset_index = session['offset_index']
    max_offset_index = len(OFFSET_LIST) - 1
    min_offset_index = 0

    # 反転が発生した場合
    if  session['current_direction'] != session['next_direction']:
        session['reversals'] += 1  # 反転回数をカウント
        session['reversals_double_flag'] = session['double_flag']  # 反転の直前のフラグ記録
        session['double_flag'] = False  # フラグリセット
        step_size = max(step_size // 2, 1)  # step_sizeを半分に、最小は1
        session['same_direction_count'] = 0  # 同じ方向のカウントをリセット
        print(f"反転が発生！ Step Sizeを半分に: {step_size}")

    # 反転しなかった場合
    else:
        session['same_direction_count'] += 1

        if session['same_direction_count'] >= 3:  # 同じ方向に4回以上連続
            step_size = min(step_size * 2, session['max_step_size'])# ステップサイズを2倍に、最大はmax_step_size
            session['double_flag'] = True
            print(f"4回以上同じ方向 Step Size倍: {step_size}")

        elif session['same_direction_count'] == 2:  # 同じ方向に3回連続
            if session['reversals_double_flag']:  # 直近の反転前のステップが倍であればそのまま
                step_size = step_size
            else:
                step_size = min(step_size * 2, session['max_step_size'])# ステップサイズを2倍に、最大はmax_step_size
                session['double_flag'] = True
                print(f"3回連続同じ方向 Step Size倍: {step_size}")

        else:  # 同じ方向に2回連続
            step_size = step_size
            print(f"2回連続同じ方向 Step Sizeそのまま: {step_size}")
    # ステップサイズの最大値を適用
    step_size = min(step_size, session['max_step_size'])

    # セッションを更新
    session.update({
        'step_size': step_size,
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_experiment():
    data = request.json
    participant_id = data.get('participant_id', 'unknown')
    frequency_dir = data.get('frequency_dir')
    max_trials = data.get('max_trials', 50)
    
    # 実験初期化
    initialize_experiment(frequency_dir)

    # セッションにデータを保存
    session['participant_id'] = participant_id
    session['max_trials'] = max_trials

    # データファイルの作成
    data_dir = os.path.join(DATA_FOLDER, participant_id, frequency_dir)
    os.makedirs(data_dir, exist_ok=True)
    session['data_file'] = os.path.join(data_dir, f"{participant_id}_{frequency_dir}_results.csv")

    # figディレクトリの作成
    fig_dir = os.path.join('static', FIG_FOLDER, participant_id, frequency_dir)
    os.makedirs(fig_dir, exist_ok=True)
    session['fig_dir'] = fig_dir

    with open(session['data_file'], 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trial', 'CorrectResponse', 'Response', 'Correct', 'Offset', 'NextStepSize', 'Reversals', 'NextDirection', 'SameDirectionCount'])

    return jsonify({"message": "実験を開始しました", "status": "success"})

@app.route('/next_trial', methods=['POST'])
def next_trial():
    offset_index = session['offset_index']
    frequency_dir = session['frequency_dir']

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

    # トライアルデータをセッションに保存
    session['current_trial'] = trial

    # デバッグ出力
    print("Next Trial Generated:", trial)

    return jsonify({
        "tones": [trial['Tone1'], trial['Tone2'], trial['Tone3']],
        "correct_response": trial['CorrectResponse']
    })

@app.route('/submit_response', methods=['POST'])
def submit_response():
    trial = session.get('current_trial')  # セッションからトライアルデータを取得
    if not trial:
        print("Error: No current trial data in session")
        return jsonify({"error": "No trial data found"}), 400

    # クライアントからの応答を取得
    response = request.json.get('response')
    correct = (response == trial['CorrectResponse'])  # キーを 'CorrectResponse' に合わせる

    offset_index = session['offset_index']  # 現在のオフセットインデックス
    current_offset = OFFSET_LIST[offset_index]  # 現在のオフセットを記録する

    # 正誤に応じた処理
    if correct:
        session['correct_count'] += 1
        if session['correct_count'] == 2:
            session['correct_count'] = 0
            session['next_direction'] = 'down'
            decide_step_size()
            offset_index = min(offset_index + session['step_size'], len(OFFSET_LIST) - 1)
            # 方向の更新
            session['current_direction'] = session['next_direction']
    else:
        session['correct_count'] = 0
        session['next_direction'] = 'up'
        decide_step_size()
        offset_index = max(offset_index - session['step_size'], 0)
        # 方向の更新
        session['current_direction'] = session['next_direction']

    # セッションの更新（次のトライアル用のオフセットにする）
    session['offset_index'] = offset_index
    session['trial_count'] += 1

    # CSVへのデータ保存（現在のオフセットを記録）
    try:
        with open(session['data_file'], 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                session['trial_count'], trial['CorrectResponse'], response, correct,
                current_offset,  # 保存するのは現在のトライアルのオフセット
                session['step_size'], session['reversals'], session['next_direction'], session['same_direction_count']
            ])
    except Exception as e:
        print("データ保存中にエラー:", e)
        return jsonify({"error": "データの保存中にエラーが発生しました"}), 500

    # デバッグ用出力
    print(f"Trial {session['trial_count']} saved: Response = {response}, Correct = {correct}, Offset = {current_offset}")

    return jsonify({
        "message": "応答受領",
        "correct": correct,
        "next_offset": OFFSET_LIST[offset_index],
        "completed": session['trial_count'] >= session['max_trials']
    })

@app.route('/experiment')
def experiment():
    return render_template('experiment.html')

@app.route('/complete')
def complete():
    # セッションからデータファイルを取得
    data_file = session.get('data_file')
    fig_dir = session.get('fig_dir')
    if not data_file or not fig_dir:
        return "Error: Data or figure directory not found."

    # MLE分析の実行
    results = perform_mle_analysis(data_file, fig_dir)
    if "error" in results:
        return results["error"]

    # 結果をテンプレートに渡す
    return render_template('complete.html', fig_path=results["fig_path"], threshold=f"{results['threshold']:.2f}")

if __name__ == '__main__':
    app.run(debug=True)