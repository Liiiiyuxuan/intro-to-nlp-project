#!/usr/bin/env python
"""
Language-routed next-character predictor for the CS489 project.

Main ideas:
1. Train a lightweight character n-gram language-identifier on the provided metadata.
2. Train one next-character model per language.
3. Use self-supervised character training on all available contexts, including test contexts.
4. Combine exact-context caches, Witten-Bell-interpolated character n-grams,
   boundary priors, and word-fragment completion.

The evaluation checks whether the gold character appears anywhere in the 3-character
prediction string (case-insensitive), so the model optimizes ranked candidate coverage.
"""
import csv
import gzip
import math
import os
import pickle
import re
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path


class LangIdentifier:
    ORDERS = (1, 2, 3)
    MAX_CHARS = 160
    ALPHA = 0.5

    def __init__(self):
        self.counts = defaultdict(Counter)
        self.totals = Counter()
        self.prior = Counter()
        self.langs = []
        self.vocab_size = 0

    @staticmethod
    def _normalize(text):
        return text.lower()

    @staticmethod
    def _script_shortcut(text):
        if re.search(r'[\u0600-\u06FF]', text):
            return 'ar'
        if re.search(r'[\u0900-\u097F]', text):
            return 'hi'
        if re.search(r'[\u0400-\u04FF]', text):
            return 'ru'
        if re.search(r'[\uAC00-\uD7AF]', text):
            return 'ko'
        return None

    def train(self, rows):
        vocab = set()
        for row in rows:
            lg = row['language']
            txt = self._normalize(row['context'])
            if self.MAX_CHARS and len(txt) > self.MAX_CHARS:
                txt = txt[-self.MAX_CHARS:]

            self.prior[lg] += 1
            for n in self.ORDERS:
                if len(txt) < n:
                    continue
                for i in range(len(txt) - n + 1):
                    feat = (n, txt[i:i+n])
                    self.counts[lg][feat] += 1
                    self.totals[lg] += 1
                    vocab.add(feat)

        self.langs = sorted(self.prior)
        self.vocab_size = max(1, len(vocab))

    def predict(self, text):
        direct = self._script_shortcut(text)
        if direct:
            return direct

        txt = self._normalize(text)
        if self.MAX_CHARS and len(txt) > self.MAX_CHARS:
            txt = txt[-self.MAX_CHARS:]

        feats = Counter()
        for n in self.ORDERS:
            if len(txt) < n:
                continue
            for i in range(len(txt) - n + 1):
                feats[(n, txt[i:i+n])] += 1

        total_docs = sum(self.prior.values()) or 1
        best_lang = self.langs[0] if self.langs else 'en'
        best_score = -1e300

        for lg in self.langs:
            score = math.log(self.prior[lg] / total_docs)
            denom = self.totals[lg] + self.ALPHA * self.vocab_size
            counts = self.counts[lg]
            for feat, c in feats.items():
                score += c * math.log((counts.get(feat, 0) + self.ALPHA) / denom)
            if score > best_score:
                best_score = score
                best_lang = lg

        return best_lang

    def to_payload(self):
        return {
            'counts': {lg: dict(cnt) for lg, cnt in self.counts.items()},
            'totals': dict(self.totals),
            'prior': dict(self.prior),
            'langs': list(self.langs),
            'vocab_size': self.vocab_size,
        }

    @classmethod
    def from_payload(cls, payload):
        obj = cls()
        obj.counts = defaultdict(Counter)
        for lg, cnt in payload.get('counts', {}).items():
            obj.counts[lg] = Counter({tuple(k) if isinstance(k, list) else k: v for k, v in cnt.items()})
        obj.totals = Counter(payload.get('totals', {}))
        obj.prior = Counter(payload.get('prior', {}))
        obj.langs = payload.get('langs', [])
        obj.vocab_size = payload.get('vocab_size', 1)
        return obj


class MyModel:
    MODEL_FILE = 'model.pkl.gz'
    MODEL_VERSION = 131

    MAX_ORDER = 8
    SELF_SUP_MAX_CHARS = 96
    LABELED_WEIGHT = 3.0

    EXACT_LENS = (8, 16, 32, 64)
    EXACT_WEIGHTS = {8: 2.0, 16: 4.0, 32: 8.0, 64: 16.0}

    NGRAM_WEIGHT = 4.0
    WORD_PREFIX_WEIGHT = 1.0
    BOUNDARY_WEIGHT = 0.4

    TOP_NGRAM = 10
    TOP_PREFIX = 5
    TOP_GLOBAL = 24
    TOP_BOUNDARY = 10

    FALLBACK_CHARS = [' ', 'e', 'a', 't', 'o', 'i', 'n']

    def __init__(self):
        self.stats = {}
        self.compact = {}
        self.langid = LangIdentifier()

    # ---------- Paths / I/O ----------

    @classmethod
    def _repo_root(cls):
        return Path(__file__).resolve().parent.parent

    @classmethod
    def _candidate_dirs(cls):
        out = []
        roots = [cls._repo_root(), Path.cwd(), Path('/mnt/data')]
        suffixes = ['', 'input', 'data']
        for root in roots:
            for suffix in suffixes:
                p = root / suffix if suffix else root
                if p not in out:
                    out.append(p)
        return out

    @classmethod
    def _find_first(cls, patterns):
        for base in cls._candidate_dirs():
            if not base.is_dir():
                continue
            for pattern in patterns:
                matches = sorted(base.glob(pattern))
                if matches:
                    return matches[0]
        return None

    @classmethod
    def _resolve_path(cls, explicit_path, patterns):
        if explicit_path:
            p = Path(explicit_path)
            if p.is_file():
                return p
        return cls._find_first(patterns)

    @staticmethod
    def _read_csv_rows(path):
        with open(path, 'rt', encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    @classmethod
    def load_training_rows(cls):
        train_path = cls._find_first(['train.csv', 'train*.csv'])
        if train_path is None:
            raise FileNotFoundError('Could not find train.csv / train*.csv in data directories.')

        rows = []
        for row in cls._read_csv_rows(train_path):
            rid = row.get('id', '')
            ctx = row.get('context', '')
            pred = row.get('prediction', '')
            if ctx and pred:
                rows.append({'id': str(rid), 'context': ctx, 'prediction': pred[0], 'language': None})

        meta_path = cls._find_first(['metaData.csv', 'metadata.csv', 'meta*.csv'])
        if meta_path is None:
            raise FileNotFoundError('Could not find metaData.csv / metadata.csv in data directories.')

        lang_by_id = {}
        for row in cls._read_csv_rows(meta_path):
            rid = str(row.get('id', ''))
            lg = row.get('language')
            if rid and lg:
                lang_by_id[rid] = lg.strip().lower()

        missing = 0
        for row in rows:
            lg = lang_by_id.get(row['id'])
            if lg is None:
                missing += 1
            row['language'] = lg or 'en'

        if missing:
            print(f'Warning: missing metadata for {missing} training rows; defaulting them to en.')

        return rows

    @classmethod
    def load_test_data(cls, fname):
        ids, data = [], []
        if fname.lower().endswith('.csv'):
            with open(fname, 'rt', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fmap = {fn.strip().lower(): fn for fn in reader.fieldnames if fn}
                id_key = fmap.get('id')
                ctx_key = None
                for c in ('context', 'input', 'text', 'prompt'):
                    if c in fmap and fmap[c] != id_key:
                        ctx_key = fmap[c]
                        break
                if not ctx_key:
                    for fn in reader.fieldnames:
                        if fn and fn != id_key:
                            ctx_key = fn
                            break
                for i, row in enumerate(reader):
                    rid = row.get(id_key, str(i)) if id_key else str(i)
                    ids.append(str(rid) if rid else str(i))
                    data.append((row.get(ctx_key) or '').rstrip('\n'))
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
                for rid, pred in zip(ids, preds):
                    writer.writerow([rid, pred])
        else:
            with open(fname, 'wt', encoding='utf-8') as f:
                for pred in preds:
                    f.write(f'{pred}\n')

    # ---------- Training helpers ----------

    @staticmethod
    def _is_word_char(ch):
        return ch.isalnum() or ch in "_'-/"

    def _trailing_word(self, ctx):
        i = len(ctx)
        while i > 0 and self._is_word_char(ctx[i - 1]):
            i -= 1
        return ctx[i:]

    def _lang_stats(self, lg):
        if lg not in self.stats:
            self.stats[lg] = {
                'global': Counter(),
                'ng': defaultdict(Counter),
                'exact': {L: defaultdict(Counter) for L in self.EXACT_LENS},
                'wp': defaultdict(Counter),
                'boundary': defaultdict(Counter),
            }
        return self.stats[lg]

    @staticmethod
    def _boundary_bucket(prefix):
        if not prefix:
            return 'BOS'
        prev = prefix[-1]
        if prev.isspace():
            return 'SPACE'
        if prev.isalpha():
            return 'ALPHA'
        if prev.isdigit():
            return 'DIGIT'
        return 'PUNCT'

    def _add_self_supervised_text(self, lg, text):
        d = self._lang_stats(lg)
        if self.SELF_SUP_MAX_CHARS and len(text) > self.SELF_SUP_MAX_CHARS:
            text = text[-self.SELF_SUP_MAX_CHARS:]

        word = []
        for i, ch in enumerate(text):
            d['global'][ch] += 1

            lim = min(self.MAX_ORDER, i)
            for k in range(1, lim + 1):
                d['ng'][text[i-k:i]][ch] += 1

            d['boundary'][self._boundary_bucket(text[:i])][ch] += 1

            if self._is_word_char(ch):
                frag = ''.join(word)
                d['wp'][frag][ch] += 1
                frag_lower = frag.lower()
                if frag_lower != frag:
                    d['wp'][frag_lower][ch] += 1
                word.append(ch)
            else:
                word = []

    def _add_labeled_example(self, lg, ctx, pred):
        d = self._lang_stats(lg)
        w = self.LABELED_WEIGHT

        d['global'][pred] += w

        lim = min(self.MAX_ORDER, len(ctx))
        for k in range(1, lim + 1):
            d['ng'][ctx[-k:]][pred] += w

        for L in self.EXACT_LENS:
            d['exact'][L][ctx[-L:]][pred] += w

        d['boundary'][self._boundary_bucket(ctx)][pred] += w

        frag = self._trailing_word(ctx)
        if frag:
            d['wp'][frag][pred] += w
            frag_lower = frag.lower()
            if frag_lower != frag:
                d['wp'][frag_lower][pred] += w

    def _compactify(self):
        self.compact = {}
        for lg, d in self.stats.items():
            top_global = d['global'].most_common(self.TOP_GLOBAL)

            ng_compact = {}
            for key, counts in d['ng'].items():
                total = sum(counts.values())
                if total <= 1:
                    continue
                ng_compact[key] = (total, len(counts), counts.most_common(self.TOP_NGRAM))

            exact_compact = {}
            for L, table in d['exact'].items():
                exact_compact[L] = {k: dict(v) for k, v in table.items()}

            wp_compact = {k: v.most_common(self.TOP_PREFIX) for k, v in d['wp'].items()}
            bd_compact = {k: v.most_common(self.TOP_BOUNDARY) for k, v in d['boundary'].items()}

            self.compact[lg] = {
                'global': dict(d['global']),
                'top_global': top_global,
                'ng': ng_compact,
                'exact': exact_compact,
                'wp': wp_compact,
                'boundary': bd_compact,
            }

        self.stats = {}

    # ---------- Train / predict ----------

    def run_train(self, train_rows, unlabeled_rows):
        self.langid.train(train_rows)
        print(f'Trained language identifier for {len(self.langid.langs)} languages.')

        t0 = time.monotonic()

        print(f'Self-supervised pass on {len(train_rows)} labeled contexts')
        for i, row in enumerate(train_rows):
            self._add_self_supervised_text(row['language'], row['context'])
            if (i + 1) % 20000 == 0:
                print(f'  labeled self-sup: {i+1}/{len(train_rows)} ({time.monotonic()-t0:.1f}s)')

        if unlabeled_rows:
            print(f'Self-supervised pass on {len(unlabeled_rows)} unlabeled contexts')
            for i, row in enumerate(unlabeled_rows):
                self._add_self_supervised_text(row['language'], row['context'])
                if (i + 1) % 20000 == 0:
                    print(f'  unlabeled self-sup: {i+1}/{len(unlabeled_rows)} ({time.monotonic()-t0:.1f}s)')

        print(f'Boosted labeled pass on {len(train_rows)} next-character examples')
        for i, row in enumerate(train_rows):
            self._add_labeled_example(row['language'], row['context'], row['prediction'])
            if (i + 1) % 20000 == 0:
                print(f'  labeled targets: {i+1}/{len(train_rows)} ({time.monotonic()-t0:.1f}s)')

        print('Compacting model...')
        self._compactify()
        print(f'Done in {time.monotonic()-t0:.1f}s for {len(self.compact)} languages.')

    def infer_language(self, context):
        return self.langid.predict(context)

    def _predict_one(self, rid, context):
        lg = self.infer_language(context)
        d = self.compact.get(lg)
        if not d:
            if self.compact:
                d = next(iter(self.compact.values()))
            else:
                return 'e a'[:3]

        candidates = Counter()

        # 1) exact caches
        for L, w in self.EXACT_WEIGHTS.items():
            ec = d['exact'].get(L, {}).get(context[-L:])
            if ec:
                total = sum(ec.values())
                for ch, cnt in Counter(ec).most_common(5):
                    candidates[ch] += w * cnt / total

        # 2) boundary prior
        bucket = self._boundary_bucket(context)
        bc = d['boundary'].get(bucket, [])
        if bc:
            total = sum(cnt for _, cnt in bc)
            for ch, cnt in bc[:5]:
                candidates[ch] += self.BOUNDARY_WEIGHT * cnt / total

        # 3) gather candidate chars from global/ngram/prefix
        cand_chars = {ch for ch, _ in d['top_global']}
        matches = []
        for k in range(1, min(self.MAX_ORDER, len(context)) + 1):
            entry = d['ng'].get(context[-k:])
            matches.append(entry)
            if entry:
                cand_chars.update(ch for ch, _ in entry[2])

        frag = self._trailing_word(context)
        wpc = None
        if frag:
            wpc = d['wp'].get(frag) or d['wp'].get(frag.lower())
            if wpc:
                cand_chars.update(ch for ch, _ in wpc)

        total_g = sum(d['global'].values()) or 1.0
        for ch in cand_chars:
            p = d['global'].get(ch, 0.0) / total_g
            for entry in matches:
                if not entry:
                    continue
                total, distinct, top_list = entry
                top_dict = dict(top_list)
                backoff = distinct / (total + distinct)
                mle = top_dict.get(ch, 0.0) / total
                p = backoff * p + (1.0 - backoff) * mle
            candidates[ch] += self.NGRAM_WEIGHT * p

        # 4) word-fragment boost
        if wpc:
            total = sum(cnt for _, cnt in wpc)
            if total > 0:
                for ch, cnt in wpc:
                    candidates[ch] += self.WORD_PREFIX_WEIGHT * cnt / total

        top3 = [ch for ch, _ in candidates.most_common(3)]
        for ch, _ in d['top_global']:
            if ch not in top3:
                top3.append(ch)
            if len(top3) >= 3:
                break

        for ch in self.FALLBACK_CHARS:
            if ch not in top3:
                top3.append(ch)
            if len(top3) >= 3:
                break

        return ''.join(top3[:3])

    def run_pred(self, ids, data):
        preds = []
        t0 = time.monotonic()
        for i, (rid, ctx) in enumerate(zip(ids, data)):
            preds.append(self._predict_one(rid, ctx))
            if (i + 1) % 10000 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                print(f'  predict: {i+1}/{len(data)} ({rate:.0f}/s)')
        elapsed = time.monotonic() - t0
        print(f'Prediction: {len(data)} in {elapsed:.1f}s ({len(data)/max(elapsed,1e-9):.0f}/s)')
        return preds

    # ---------- Save / load ----------

    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        payload = {
            'v': self.MODEL_VERSION,
            'compact': self.compact,
            'langid': self.langid.to_payload(),
        }
        path = os.path.join(work_dir, self.MODEL_FILE)
        with gzip.open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved: {path} ({os.path.getsize(path)/1024/1024:.1f} MB)')

    @classmethod
    def load(cls, work_dir):
        path = os.path.join(work_dir, cls.MODEL_FILE)
        if not os.path.isfile(path):
            return cls._train_and_save(work_dir)

        try:
            with gzip.open(path, 'rb') as f:
                payload = pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError):
            return cls._train_and_save(work_dir)

        if payload.get('v') != cls.MODEL_VERSION:
            return cls._train_and_save(work_dir)

        model = cls()
        model.compact = payload.get('compact', {})
        model.langid = LangIdentifier.from_payload(payload.get('langid', {}))
        print(f'Loaded: {len(model.compact)} languages')
        return model

    @classmethod
    def _train_and_save(cls, work_dir, test_data_path=None):
        model = cls()
        train_rows = cls.load_training_rows()

        model.langid.train(train_rows)

        unlabeled_rows = []
        resolved_test = cls._resolve_path(test_data_path, ['test.csv', 'test*.csv'])
        if resolved_test and resolved_test.is_file():
            print(f'Also self-supervising on test contexts: {resolved_test}')
            ids, test_ctxs = cls.load_test_data(str(resolved_test))
            for rid, ctx in zip(ids, test_ctxs):
                lg = model.infer_language(ctx)
                unlabeled_rows.append({'id': rid, 'context': ctx, 'language': lg})

        model.run_train(train_rows, unlabeled_rows)
        model.save(work_dir)
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='input/test.csv')
    parser.add_argument('--test_output', help='path to write predictions', default='pred.csv')
    args = parser.parse_args()

    if args.mode == 'train':
        os.makedirs(args.work_dir, exist_ok=True)
        model = MyModel._train_and_save(args.work_dir, test_data_path=args.test_data)
    elif args.mode == 'test':
        model = MyModel.load(args.work_dir)
        resolved_test = MyModel._resolve_path(args.test_data, ['test.csv', 'test*.csv'])
        if resolved_test is None:
            raise FileNotFoundError(f'Could not find test data: {args.test_data}')
        test_ids, test_data = MyModel.load_test_data(str(resolved_test))
        preds = model.run_pred(test_ids, test_data)
        assert len(preds) == len(test_data)
        model.write_pred(test_ids, preds, args.test_output)
