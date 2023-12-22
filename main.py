from collections import Counter
from nltk.util import ngrams
from nltk import word_tokenize
import math
import logging
import time


class BLEUScorer:
    def __init__(self, reference_files, logger):
        self.logger = logger
        self.logger.log_start()
        self.references = [file_loader(file) for file in reference_files]
        self.logger.log_data_sources(reference_files)

    def calculate_bleu_score(self, candidates):
        """Вычисляет BLEU-балл для кандидата по сравнению с несколькими эталонными переводами."""
        try:
            bleu_scores = {}
            for i, candidate in enumerate(candidates):
                candidate_scores, precisions, brevity_penalty, ratio, candidate_len, reference_len = \
                    [self.calculate_individual_bleu(candidate, [ref]) for ref in self.references][0]
                unigram, bigram, trigram, quadram = [x * 100 for x in precisions]
                bleu_scores[f'Перевод {i + 1}'] = (
                    f'bleu: {candidate_scores * 100:.2f}, {unigram:.1f}/{bigram:.1f}/{trigram:.1f}/{quadram:.1f}, '
                    f'BP: {brevity_penalty:.3f}, ratio: {ratio:.3f}, hyp_len = {candidate_len}, ref_len: {reference_len} ')
            self.logger.log_results(bleu_scores)
            return bleu_scores
        except Exception as e:
            self.logger.log_error(f"Ошибка при расчете BLEU-балла: {str(e)}")

    @staticmethod
    def calculate_individual_bleu(candidate, references):
        """Вычисляет BLEU-балл для одного кандидата по отношению к нескольким эталонным переводам."""
        # Обработка кандидата (машинного перевода)
        candidate_tokens = word_tokenize(candidate)

        # Расчет точности для каждой n-граммы
        precisions = []
        for n in range(1, 5):
            candidate_ngrams = list(ngrams(candidate_tokens, n))
            ref_counts = []
            for ref in references:
                ref_ngrams = list(ngrams(word_tokenize(ref), n))
                ref_count_dict = Counter(ref_ngrams)
                ref_counts.append(ref_count_dict)

            overlap = 0
            for ngram in candidate_ngrams:
                max_overlap = max(ref_count[ngram] for ref_count in ref_counts)
                overlap += min(1, max_overlap)

            if len(candidate_ngrams) == 0:
                precisions.append(0)
            else:
                precisions.append(overlap / len(candidate_ngrams))

        # Расчет геометрического среднего
        geometric_mean = math.exp(sum(map(math.log, precisions)) / 4)

        # Штраф за краткость
        len_candidate = len(candidate_tokens)
        len_references = [len(word_tokenize(ref)) for ref in references]
        closest_ref_len = min(len_references, key=lambda ref_len: (abs(ref_len - len_candidate), ref_len))
        brevity_penalty = math.exp(1 - closest_ref_len / len_candidate) if len_candidate < closest_ref_len else 1

        return brevity_penalty * geometric_mean, precisions, brevity_penalty, closest_ref_len / len_candidate, len_candidate, closest_ref_len


class Logger:
    def __init__(self, log_file):
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%dT%H:%M:%S%z', encoding="utf-8")
        self.start_time = time.time()

    @staticmethod
    def log_start():
        logging.info("Программа запущена")

    def log_end(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        logging.info(f"Программа завершена. Время выполнения: {elapsed_time:.2f} секунд.")

    @staticmethod
    def log_data_sources(file_names):
        for file_name in file_names:
            logging.info(f"Источник данных: {file_name}")

    @staticmethod
    def log_results(results):
        logging.info(f"Результаты работы программы: {results}")

    # todo: Добавить метод, сохраняющий комментарий пользователя, при его отсутствии находить обобщающее слово для текста

    @staticmethod
    def log_error(error_message):
        logging.error(f"Ошибка: {error_message}")


def file_loader(file_path):
    # todo: Возможно добавиться ручной ввод, поэтому пока не ясно стоит ли вносить в класс
    """Загружает эталонный перевод из файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        logger.log_error(f"Ошибка при загрузке файла {file_path}: {str(e)}")
        return None


def main():
    logger = Logger("program_log.log")  # todo: Перевести логгер из записи в файл в запись в СУБД
    reference_files = ["data/ref_1_MTmedicine.txt"]  # Список файлов с эталонными переводами
    candidate_files = ["data/hyp_1_1_MTmedicineyandex.txt", "data/hyp_1_2_MTmedicinedeep.txt"]
    scorer = BLEUScorer(reference_files, logger)
    machine_translation = [file_loader(file) for file in candidate_files]

    try:
        bleu_scores = scorer.calculate_bleu_score(machine_translation)
    finally:
        logger.log_end()

    for translation_id, score in bleu_scores.items():
        print(f"Статистика для '{translation_id}': {score}")


if __name__ == '__main__':
    main()
