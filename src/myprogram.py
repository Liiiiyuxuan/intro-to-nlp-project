#!/usr/bin/env python
import csv
import json
import math
import os
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path


class MyModel:
    """
    Hybrid next-character model:
    - Character n-gram backbone
    - Lightweight neural reranker trained with negative sampling
    """

    MODEL_FILE = 'model.checkpoint'
    MODEL_VERSION = 8
    USE_NEURAL_RERANK_AT_INFERENCE = False

    def __init__(self, max_order=8, emb_dim=24, ctx_window=12):
        self.max_order = max_order
        self.global_counts = Counter()
        self.order_counts = {k: defaultdict(Counter) for k in range(1, self.max_order + 1)}

        # Neural reranker params
        self.emb_dim = emb_dim
        self.ctx_window = ctx_window
        self.char_emb = {}   # char -> [float]
        self.char_bias = {}  # char -> float
        self._rng = random.Random(0)

    @classmethod
    def _repo_root(cls):
        return Path(__file__).resolve().parent.parent

    @classmethod
    def load_training_data(cls):
        labeled = []
        unlabeled = []
        data_dir = cls._repo_root() / 'data'
        if not data_dir.is_dir():
            return {'labeled': labeled, 'unlabeled': unlabeled}

        for input_path in sorted(data_dir.glob('**/input*.txt')):
            with open(input_path, 'rt', encoding='utf-8') as f_in:
                for context in f_in:
                    unlabeled.append(context.rstrip('\n'))

        for answer_path in sorted(data_dir.glob('**/answer*.txt')):
            input_name = answer_path.name.replace('answer', 'input', 1)
            input_path = answer_path.with_name(input_name)
            if not input_path.is_file():
                continue

            with open(input_path, 'rt', encoding='utf-8') as f_in, open(answer_path, 'rt', encoding='utf-8') as f_ans:
                for context, nxt in zip(f_in, f_ans):
                    context = context.rstrip('\n')
                    nxt = nxt.rstrip('\n')
                    if nxt:
                        labeled.append((context, nxt[0]))

        return {'labeled': labeled, 'unlabeled': unlabeled}

    @classmethod
    def load_test_data(cls, fname):
        ids = []
        data = []

        if fname.lower().endswith('.csv'):
            with open(fname, 'rt', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return ids, data

                context_key = None
                for field in reader.fieldnames:
                    if field and field.upper() != 'ID':
                        context_key = field
                        break

                if context_key is None:
                    raise ValueError('CSV test data must include a non-ID text/context column')

                for i, row in enumerate(reader):
                    row_id = row.get('ID')
                    if row_id in (None, ''):
                        row_id = str(i)
                    ids.append(str(row_id))
                    data.append((row.get(context_key) or '').rstrip('\n'))
        else:
            with open(fname, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    ids.append(str(i))
                    data.append(line.rstrip('\n'))

        return ids, data

    @classmethod
    def write_pred(cls, ids, preds, fname):
        if fname.lower().endswith('.csv'):
            with open(fname, 'wt', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'prediction'])
                for row_id, pred in zip(ids, preds):
                    writer.writerow([row_id, pred])
            return

        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write(f'{p}\n')

    def _ensure_char(self, ch):
        if ch not in self.char_emb:
            self.char_emb[ch] = [self._rng.uniform(-0.05, 0.05) for _ in range(self.emb_dim)]
            self.char_bias[ch] = 0.0

    def _update_counts(self, context, nxt, weight=1.0):
        if not nxt:
            return

        self.global_counts[nxt] += weight
        max_k = min(self.max_order, len(context))
        for k in range(1, max_k + 1):
            suffix = context[-k:]
            self.order_counts[k][suffix][nxt] += weight

    def _pretrain_on_contexts(self, contexts, weight=0.2, max_chars=256):
        for text in contexts:
            if not text:
                continue
            truncated = text[:max_chars]
            for ch in truncated:
                self.global_counts[ch] += weight
            for i in range(1, len(truncated)):
                prev = truncated[i - 1]
                nxt = truncated[i]
                self.order_counts[1][prev][nxt] += weight

    def _ctx_vector(self, context):
        tail = context[-self.ctx_window:]
        if not tail:
            return [0.0] * self.emb_dim

        vec = [0.0] * self.emb_dim
        used = 0
        for ch in tail:
            emb = self.char_emb.get(ch)
            if emb is None:
                continue
            used += 1
            for i in range(self.emb_dim):
                vec[i] += emb[i]

        if used == 0:
            return [0.0] * self.emb_dim

        scale = 1.0 / used
        for i in range(self.emb_dim):
            vec[i] *= scale
        return vec

    def _dot(self, a, b):
        return sum(x * y for x, y in zip(a, b))

    def _sigmoid(self, x):
        if x < -40:
            return 0.0
        if x > 40:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def _train_neural_reranker(self, labeled, epochs=2, neg_samples=5, lr=0.03):
        # Build vocabulary from observed outputs and context chars.
        vocab = set(self.global_counts.keys())
        for context, nxt in labeled:
            vocab.add(nxt)
            for ch in context[-self.ctx_window:]:
                vocab.add(ch)

        for ch in vocab:
            self._ensure_char(ch)

        vocab_list = list(vocab)
        if len(vocab_list) <= 1 or not labeled:
            return

        for _ in range(epochs):
            self._rng.shuffle(labeled)
            for context, true_ch in labeled:
                self._ensure_char(true_ch)
                ctx = self._ctx_vector(context)

                # Positive update
                true_emb = self.char_emb[true_ch]
                true_score = self._dot(ctx, true_emb) + self.char_bias[true_ch]
                p = self._sigmoid(true_score)
                g = (1.0 - p)  # d log sigmoid(x)

                for i in range(self.emb_dim):
                    true_emb[i] += lr * g * ctx[i]
                self.char_bias[true_ch] += lr * g

                # Negative samples
                for _n in range(neg_samples):
                    neg = self._rng.choice(vocab_list)
                    if neg == true_ch:
                        continue
                    neg_emb = self.char_emb[neg]
                    neg_score = self._dot(ctx, neg_emb) + self.char_bias[neg]
                    pn = self._sigmoid(neg_score)
                    gn = -pn  # d log sigmoid(-x)

                    for i in range(self.emb_dim):
                        neg_emb[i] += lr * gn * ctx[i]
                    self.char_bias[neg] += lr * gn

    def run_train(self, train_data, work_dir):
        labeled = train_data.get('labeled', [])
        unlabeled = train_data.get('unlabeled', [])

        self._pretrain_on_contexts(unlabeled, weight=0.2)
        for context, nxt in labeled:
            self._update_counts(context, nxt, weight=1.0)

        if not self.global_counts:
            for ch in ' etaoinshrdlucmfwypvbgkqjxz,.!?':
                self.global_counts[ch] += 1.0

        # Neural fine-tuning stage
        self._train_neural_reranker(labeled)

    def _ngram_scores(self, context):
        scores = Counter()
        max_k = min(self.max_order, len(context))

        for k in range(max_k, 0, -1):
            suffix = context[-k:]
            counts = self.order_counts[k].get(suffix)
            if not counts:
                continue

            total = sum(counts.values())
            if total <= 0:
                continue

            order_weight = float(k ** 1.5)
            for ch, cnt in counts.items():
                scores[ch] += order_weight * (cnt / total)

        global_total = sum(self.global_counts.values())
        if global_total > 0:
            for ch, cnt in self.global_counts.items():
                scores[ch] += 0.15 * (cnt / global_total)

        return scores

    def _top_guesses(self, context, n=3):
        ngram_scores = self._ngram_scores(context)

        # Fast path required by course runtime constraints.
        if not self.USE_NEURAL_RERANK_AT_INFERENCE:
            ranked = [ch for ch, _ in ngram_scores.most_common(16)]
            if len(ranked) < n:
                for ch, _ in self.global_counts.most_common(32):
                    if ch not in ranked:
                        ranked.append(ch)
                    if len(ranked) >= n:
                        break
            while len(ranked) < n:
                ranked.append(' ')
            return ''.join(ranked[:n])

        # Optional neural rerank path.
        candidates = [ch for ch, _ in ngram_scores.most_common(32)]
        for ch, _ in self.global_counts.most_common(32):
            if ch not in candidates:
                candidates.append(ch)

        if not candidates:
            return ' ' * n

        ctx = self._ctx_vector(context)
        combined = []
        for ch in candidates:
            emb = self.char_emb.get(ch)
            neural = 0.0
            if emb is not None:
                neural = self._dot(ctx, emb) + self.char_bias.get(ch, 0.0)
            score = ngram_scores.get(ch, 0.0) + 0.35 * neural
            combined.append((score, ch))

        combined.sort(reverse=True)
        ranked = [ch for _, ch in combined]

        while len(ranked) < n:
            ranked.append(' ')

        return ''.join(ranked[:n])

    def run_pred(self, data):
        return [self._top_guesses(inp) for inp in data]

    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        payload = {
            'model_version': self.MODEL_VERSION,
            'max_order': self.max_order,
            'global_counts': dict(self.global_counts),
            'order_counts': {
                str(k): {suffix: dict(cnts) for suffix, cnts in suffix_map.items()}
                for k, suffix_map in self.order_counts.items()
            },
            'emb_dim': self.emb_dim,
            'ctx_window': self.ctx_window,
            'char_emb': self.char_emb,
            'char_bias': self.char_bias,
        }
        with open(os.path.join(work_dir, self.MODEL_FILE), 'wt', encoding='utf-8') as f:
            json.dump(payload, f)

    @classmethod
    def _train_and_save_default(cls, work_dir):
        model = cls()
        model.run_train(cls.load_training_data(), work_dir)
        model.save(work_dir)
        return model

    @classmethod
    def load(cls, work_dir):
        model_path = os.path.join(work_dir, cls.MODEL_FILE)
        if not os.path.isfile(model_path):
            return cls._train_and_save_default(work_dir)

        try:
            with open(model_path, 'rt', encoding='utf-8') as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            return cls._train_and_save_default(work_dir)

        if payload.get('model_version') != cls.MODEL_VERSION:
            return cls._train_and_save_default(work_dir)

        model = cls(
            max_order=payload.get('max_order', 8),
            emb_dim=payload.get('emb_dim', 24),
            ctx_window=payload.get('ctx_window', 12),
        )
        model.global_counts.update(payload.get('global_counts', {}))

        raw_order_counts = payload.get('order_counts', {})
        for key, suffix_map in raw_order_counts.items():
            k = int(key)
            for suffix, cnts in suffix_map.items():
                model.order_counts[k][suffix].update(cnts)

        for ch, vec in payload.get('char_emb', {}).items():
            model.char_emb[ch] = [float(v) for v in vec]
        for ch, b in payload.get('char_bias', {}).items():
            model.char_bias[ch] = float(b)

        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Making working directory {args.work_dir}')
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print(
            f"Pretraining on {len(train_data.get('unlabeled', []))} contexts "
            f"and fine-tuning on {len(train_data.get('labeled', []))} labeled samples"
        )
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print(f'Loading test data from {args.test_data}')
        test_ids, test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print(f'Writing predictions to {args.test_output}')
        assert len(pred) == len(test_data), f'Expected {len(test_data)} predictions but got {len(pred)}'
        model.write_pred(test_ids, pred, args.test_output)
    else:
        raise NotImplementedError(f'Unknown mode {args.mode}')
