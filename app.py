from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from Logger import PostgresLogger
from BLEU_calcs import BLEUScorer
from dotenv import load_dotenv
import os

load_dotenv()
logger = PostgresLogger(host=os.getenv('DB_HOST'), db=os.getenv('DB_NAME'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'))

upload_folder = os.path.join(os.getcwd(), 'temp')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

allowed_ext = {'txt', 'doc'}
app = Flask(__name__)
app.config["upload_folder"] = upload_folder


@app.route('/')
def home():
    return render_template('template.html')


@app.route('/process', methods=['POST'])
def process_input():
    temp_files = []
    texts_ref = []
    texts_cand = []

    try:
        text_ref = request.form.get('textRef')
        file_ref = request.files.getlist('fileRef')
        for file in file_ref:
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['upload_folder'], filename)
                file.save(file_path)
                temp_files.append(file_path)
                texts_ref.append(file_path)
        if text_ref:
            texts_ref.append(text_ref)

        text_cand = request.form.get('textCand')
        file_cand = request.files.getlist('fileCand')
        for file in file_cand:
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['upload_folder'], filename)
                file.save(file_path)
                temp_files.append(file_path)
                texts_cand.append(file_path)
        if text_cand:
            texts_cand.append(text_cand)

        scorer = BLEUScorer(texts_ref, texts_cand, logger)
        bleu_scores = scorer.calculate_bleu_score()

    finally:
        # Удаление всех временных файлов после обработки
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    def format_key(key):
        return (key[:30] + '...') if len(key) > 30 else key

    output_result = "\n".join(f"{format_key(key)}: {value}" for key, value in bleu_scores.items())
    return output_result


@app.route('/info')
def info():
    return render_template("description.html")


if __name__ == '__main__':
    app.run(debug=True)
