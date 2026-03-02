#!/usr/bin/env python
import csv
import json
import math
import os
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path


class MyModel2:
    """
    Stronger hybrid next-character model.

    Components:
    - Interpolated character n-gram language model (orders 1..max_order)
    - Neural reranker trained with negative sampling on labeled next-char pairs
    - Script-consistency prior to improve multilingual robustness
    """

    MODEL_FILE = 'model2.checkpoint'
    MODEL_VERSION = 1

    def __init__(self, max_order=7, emb_dim=32, ctx_window=20, seed=0):
        self.max_order = max_order
        self.emb_dim = emb_dim
        self.ctx_window = ctx_window
        self._rng = random.Random(seed)

        self.global_counts = Counter()
        self.order_counts = {k: defaultdict(Counter) for k in range(1, self.max_order + 1)}

        self.char_emb = {}
        self.char_bias = {}
        self.adagrad_emb = {}
        self.adagrad_bias = defaultdict(float)

        self.script_counts = Counter()

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
            with open(input_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    unlabeled.append(line.rstrip('\n'))

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
        ids, data = [], []
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
                    raise ValueError('CSV test data must include ID and one context/text column')

                for i, row in enumerate(reader):
                    row_id = row.get('ID') or str(i)
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
                for row_id, p in zip(ids, preds):
                    writer.writerow([row_id, p])
            return

        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write(f'{p}\n')

    def _script(self, ch):
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            return 'han'
        if 0x3040 <= o <= 0x30FF:
            return 'ja'
        if 0xAC00 <= o <= 0xD7AF:
            return 'ko'
        if 0x0400 <= o <= 0x04FF:
            return 'cyr'
        if 0x0600 <= o <= 0x06FF:
            return 'arab'
        if 0x0900 <= o <= 0x097F:
            return 'dev'
        if 65 <= o <= 122 or ch.isascii():
            return 'latn'
        return 'other'

    def _ensure_char(self, ch):
        if ch in self.char_emb:
            return
        self.char_emb[ch] = [self._rng.uniform(-0.08, 0.08) for _ in range(self.emb_dim)]
        self.char_bias[ch] = 0.0
        self.adagrad_emb[ch] = [1e-6] * self.emb_dim
        self.adagrad_bias[ch] = 1e-6

    def _update_ngram_counts(self, context, nxt, weight=1.0):
        if not nxt:
            return
        self.global_counts[nxt] += weight
        self.script_counts[self._script(nxt)] += weight

        max_k = min(self.max_order, len(context))
        for k in range(1, max_k + 1):
            suffix = context[-k:]
            self.order_counts[k][suffix][nxt] += weight

    def _pretrain_from_unlabeled(self, unlabeled, weight=0.12, cap_chars=300):
        for text in unlabeled:
            if not text:
                continue
            t = text[:cap_chars]
            for ch in t:
                self.global_counts[ch] += weight
                self.script_counts[self._script(ch)] += weight
            for i in range(1, len(t)):
                self.order_counts[1][t[i - 1]][t[i]] += weight

    def _ctx_vec(self, context):
        tail = context[-self.ctx_window:]
        if not tail:
            return [0.0] * self.emb_dim

        vec = [0.0] * self.emb_dim
        norm = 0.0
        # recency weighting
        start_idx = max(0, len(tail) - self.ctx_window)
        for i, ch in enumerate(tail[start_idx:]):
            emb = self.char_emb.get(ch)
            if emb is None:
                continue
            w = 1.0 + (i / max(1, len(tail) - 1))
            norm += w
            for d in range(self.emb_dim):
                vec[d] += w * emb[d]

        if norm <= 0.0:
            return [0.0] * self.emb_dim
        inv = 1.0 / norm
        for d in range(self.emb_dim):
            vec[d] *= inv
        return vec

    def _dot(self, a, b):
        return sum(x * y for x, y in zip(a, b))

    def _sigmoid(self, x):
        if x < -40:
            return 0.0
        if x > 40:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def _adagrad_step(self, ch, grad_vec, grad_bias, lr):
        emb = self.char_emb[ch]
        acc = self.adagrad_emb[ch]
        for i, g in enumerate(grad_vec):
            acc[i] += g * g
            emb[i] += lr * g / math.sqrt(acc[i])

        self.adagrad_bias[ch] += grad_bias * grad_bias
        self.char_bias[ch] += lr * grad_bias / math.sqrt(self.adagrad_bias[ch])

    def _train_neural(self, labeled, epochs=3, neg_samples=8, lr=0.06, max_examples=160000):
        vocab = set(self.global_counts.keys())
        for context, nxt in labeled[:max_examples]:
            vocab.add(nxt)
            for ch in context[-self.ctx_window:]:
                vocab.add(ch)

        for ch in vocab:
            self._ensure_char(ch)
        vocab_list = list(vocab)
        if len(vocab_list) < 2:
            return

        train_data = labeled[:max_examples]
        for _ in range(epochs):
            self._rng.shuffle(train_data)
            for context, pos in train_data:
                ctx = self._ctx_vec(context)

                # positive sample
                pos_emb = self.char_emb[pos]
                score = self._dot(ctx, pos_emb) + self.char_bias[pos]
                p = self._sigmoid(score)
                g = (1.0 - p)
                grad = [g * v for v in ctx]
                self._adagrad_step(pos, grad, g, lr)

                # negatives
                for _n in range(neg_samples):
                    neg = self._rng.choice(vocab_list)
                    if neg == pos:
                        continue
                    neg_emb = self.char_emb[neg]
                    ns = self._dot(ctx, neg_emb) + self.char_bias[neg]
                    pn = self._sigmoid(ns)
                    gn = -pn
                    gradn = [gn * v for v in ctx]
                    self._adagrad_step(neg, gradn, gn, lr)

    def run_train(self, train_data, work_dir):
        labeled = train_data.get('labeled', [])
        unlabeled = train_data.get('unlabeled', [])

        self._pretrain_from_unlabeled(unlabeled)
        for context, nxt in labeled:
            self._update_ngram_counts(context, nxt, weight=1.0)

        if not self.global_counts:
            for ch in ' etaoinshrdlucmfwypvbgkqjxz,.!?':
                self.global_counts[ch] += 1.0
                self.script_counts[self._script(ch)] += 1.0

        self._train_neural(labeled)

    def _ngram_scores(self, context):
        scores = Counter()
        max_k = min(self.max_order, len(context))

        # Interpolated order weights (favor long context)
        order_weights = {}
        denom = 0.0
        for k in range(1, max_k + 1):
            w = float(k * k)
            order_weights[k] = w
            denom += w
        if denom == 0.0:
            denom = 1.0

        for k in range(max_k, 0, -1):
            suffix = context[-k:]
            counts = self.order_counts[k].get(suffix)
            if not counts:
                continue
            total = float(sum(counts.values()))
            if total <= 0:
                continue
            alpha = order_weights[k] / denom
            for ch, cnt in counts.items():
                scores[ch] += alpha * (cnt / total)

        global_total = float(sum(self.global_counts.values()))
        if global_total > 0:
            for ch, cnt in self.global_counts.items():
                scores[ch] += 0.08 * (cnt / global_total)

        return scores

    def _dominant_script(self, context):
        if not context:
            return None
        c = Counter(self._script(ch) for ch in context[-self.ctx_window:])
        return c.most_common(1)[0][0] if c else None

    def _top_guesses(self, context, n=3):
        ngram_scores = self._ngram_scores(context)

        candidates = [ch for ch, _ in ngram_scores.most_common(48)]
        for ch, _ in self.global_counts.most_common(32):
            if ch not in candidates:
                candidates.append(ch)

        if not candidates:
            return ' ' * n

        dom_script = self._dominant_script(context)
        total_script = float(sum(self.script_counts.values()))
        ctx = self._ctx_vec(context)

        combined = []
        for ch in candidates:
            # Neural score
            emb = self.char_emb.get(ch)
            neural = 0.0
            if emb is not None:
                neural = self._dot(ctx, emb) + self.char_bias.get(ch, 0.0)

            # Script prior bonus/penalty
            s_bonus = 0.0
            if dom_script is not None:
                if self._script(ch) == dom_script:
                    s_bonus += 0.06
                elif dom_script != 'other':
                    s_bonus -= 0.02

            # Frequency smoothing by script prevalence
            if total_script > 0:
                s_bonus += 0.02 * (self.script_counts.get(self._script(ch), 0.0) / total_script)

            score = ngram_scores.get(ch, 0.0) + 0.28 * neural + s_bonus
            combined.append((score, ch))

        combined.sort(reverse=True)
        ranked = []
        for _, ch in combined:
            if ch not in ranked:
                ranked.append(ch)
            if len(ranked) >= n:
                break

        while len(ranked) < n:
            ranked.append(' ')

        return ''.join(ranked[:n])

    def run_pred(self, data):
        return [self._top_guesses(ctx, n=3) for ctx in data]

    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        payload = {
            'model_version': self.MODEL_VERSION,
            'max_order': self.max_order,
            'emb_dim': self.emb_dim,
            'ctx_window': self.ctx_window,
            'global_counts': dict(self.global_counts),
            'order_counts': {
                str(k): {suffix: dict(cnts) for suffix, cnts in mp.items()}
                for k, mp in self.order_counts.items()
            },
            'script_counts': dict(self.script_counts),
            'char_emb': self.char_emb,
            'char_bias': self.char_bias,
            'adagrad_emb': self.adagrad_emb,
            'adagrad_bias': dict(self.adagrad_bias),
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
            max_order=payload.get('max_order', 7),
            emb_dim=payload.get('emb_dim', 32),
            ctx_window=payload.get('ctx_window', 20),
        )

        model.global_counts.update(payload.get('global_counts', {}))
        model.script_counts.update(payload.get('script_counts', {}))

        for k, mp in payload.get('order_counts', {}).items():
            kk = int(k)
            for suffix, cnts in mp.items():
                model.order_counts[kk][suffix].update(cnts)

        for ch, vec in payload.get('char_emb', {}).items():
            model.char_emb[ch] = [float(v) for v in vec]
        for ch, b in payload.get('char_bias', {}).items():
            model.char_bias[ch] = float(b)

        for ch, vec in payload.get('adagrad_emb', {}).items():
            model.adagrad_emb[ch] = [float(v) for v in vec]
        for ch, b in payload.get('adagrad_bias', {}).items():
            model.adagrad_bias[ch] = float(b)

        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', default='work', help='working directory')
    parser.add_argument('--test_data', default='example/input.txt', help='path to test data')
    parser.add_argument('--test_output', default='pred.txt', help='path to predictions')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Making working directory {args.work_dir}')
            os.makedirs(args.work_dir)
        print('Instantiating MyModel2')
        model = MyModel2()
        print('Loading training data')
        train_data = MyModel2.load_training_data()
        print(
            f"Training with {len(train_data.get('unlabeled', []))} unlabeled contexts and "
            f"{len(train_data.get('labeled', []))} labeled pairs"
        )
        model.run_train(train_data, args.work_dir)
        print('Saving model checkpoint')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model checkpoint')
        model = MyModel2.load(args.work_dir)
        print(f'Loading test data: {args.test_data}')
        test_ids, test_data = MyModel2.load_test_data(args.test_data)
        print('Generating predictions')
        preds = model.run_pred(test_data)
        assert len(preds) == len(test_data), f'Expected {len(test_data)} preds but got {len(preds)}'
        print(f'Writing predictions to {args.test_output}')
        MyModel2.write_pred(test_ids, preds, args.test_output)
    else:
        raise NotImplementedError(args.mode)
