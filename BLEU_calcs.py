import os.path
from collections import Counter
from nltk.util import ngrams
from nltk import word_tokenize
import math
from datetime import datetime


class BLEUScorer:
    def __init__(self, reference_texts, candidate_texts, logger):
        self.logger = logger
        self.start_time = datetime.now()
        self.ref_sources = ",".join([os.path.basename(text) for text in reference_texts]) if isinstance(reference_texts, list) else os.path.basename(reference_texts)
        self.machine_translation = ",".join([os.path.basename(text) for text in candidate_texts]) if isinstance(candidate_texts, list) else os.path.basename(candidate_texts)
        self.candidates = [self.process_input(text) for text in candidate_texts]
        self.references = [self.process_input(text) for text in reference_texts]

    def calculate_bleu_score(self):
        """Вычисляет BLEU-балл для кандидата по сравнению с несколькими эталонными переводами."""
        try:
            bleu_scores = {}
            for i, candidate in enumerate(self.candidates):
                candidate_scores, precisions, brevity_penalty, ratio, candidate_len, reference_len = \
                    [self.calculate_individual_bleu(candidate, [ref]) for ref in self.references][0]
                unigram, bigram, trigram, quadram = [x * 100 for x in precisions]
                bleu_scores[f"{self.machine_translation.split(',')[i]}"] = (
                    f'BLEU = {candidate_scores * 100:.2f}, {unigram:.1f}/{bigram:.1f}/{trigram:.1f}/{quadram:.1f} '
                    f'(BP = {brevity_penalty:.3f}, ratio = {ratio:.3f}, hyp_len = {candidate_len}, ref_len = {reference_len})')
            end_time = datetime.now()
            results = str(bleu_scores)
            error_details = None
            self.logger.log(self.start_time, end_time, self.ref_sources, self.machine_translation, results, error_details)
            return bleu_scores
        except Exception as e:
            end_time = datetime.now()
            error_details, results = f"Ошибка при расчете BLEU-балла: {str(e)}", "Ошибка при выполнении расчёта"
            self.logger.log(self.start_time, end_time, self.ref_sources, self.machine_translation, results, error_details)
            raise e

    @staticmethod
    def process_input(input_data):
        """Загружает эталонный перевод из файла."""
        if os.path.isfile(input_data):
            try:
                with open(input_data, 'r', encoding='utf-8') as file:
                    return file.read().strip()
            except Exception as e:
                raise e
        else:
            return input_data

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
        if 0.0 in precisions:
            geometric_mean = 0.0
        else:
            geometric_mean = math.exp(sum(map(math.log, precisions)) / 4)

        # Штраф за краткость
        len_candidate = len(candidate_tokens)
        len_references = [len(word_tokenize(ref)) for ref in references]
        closest_ref_len = min(len_references, key=lambda ref_len: (abs(ref_len - len_candidate), ref_len))
        brevity_penalty = math.exp(1 - closest_ref_len / len_candidate) if len_candidate < closest_ref_len else 1

        return brevity_penalty * geometric_mean, precisions, brevity_penalty, closest_ref_len / len_candidate, len_candidate, closest_ref_len