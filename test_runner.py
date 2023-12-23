from Logger import PostgresLogger
from BLEU_calcs import BLEUScorer
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    logger = PostgresLogger(host=os.getenv('DB_HOST'), db=os.getenv('DB_NAME'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'))
    reference_files = ["data/ref_1_MTmedicine.txt"]  # Список файлов с эталонными переводами
    candidate_files = ["data/hyp_1_1_MTmedicineyandex.txt", "data/hyp_1_2_MTmedicinedeep.txt"]
    scorer = BLEUScorer(reference_files, candidate_files, logger)
    bleu_scores = scorer.calculate_bleu_score()

    for translation_id, score in bleu_scores.items():
        print(f"Статистика для '{translation_id}': {score}")


if __name__ == '__main__':
    main()
