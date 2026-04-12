
#!/usr/bin/env python
import csv
import json
import math
import os
import unicodedata
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path


class MyModel:
    """
    Multilingual next-character model with a much faster cached reranker.

    Design goals:
    - use unlabeled data as dense self-supervised next-char LM training
    - use a smoothed character n-gram backbone
    - add script-aware priors for multilingual data
    - add word-fragment completion where morphology/token boundaries help
    - add a lightweight neural average-embedding LM as an auxiliary scorer
    - keep a lightweight reranker, but train it on cached features so it stays fast
    """

    MODEL_FILE = 'model.checkpoint'
    MODEL_VERSION = 61

    BOS = '\u0002'  # training-only start marker
    DISALLOWED_OUTPUT_CHARS = {' ', BOS}

    DEFAULT_CHARS = list(
        "etaoinshrdlucmfwypvbgkqjxz"
        "ETAOINSHRDLUCMFWYPVBGKQJXZ"
        "0123456789"
        ".,!?;:'\"-_/()[]{}@#$%&*+=<>\\|`~"
    )

    SCRIPT_NAMES = [
        'LATIN', 'CYRILLIC', 'ARABIC', 'DEVANAGARI', 'CJK',
        'HIRAGANA', 'KATAKANA', 'HANGUL', 'GREEK', 'DIGIT',
        'COMMON', 'OTHER',
    ]

    FEATURE_NAMES = [
        'bias',
        'base_score',
        'ngram_prob',
        'exact_prob',
        'word_prob',
        'boundary_prob',
        'script_prob',
        'neural_prob',
        'global_prob',
        'same_as_last',
        'same_script_as_last',
        'is_common_candidate',
        'repeat_char',
        'repeat_bigram',
    ]

    def __init__(
        self,
        max_order=10,
        seq_weight=1.0,
        labeled_weight=4.0,
        emb_dim=16,
        ctx_window=16,
        rerank_candidates=16,
        max_neural_unlabeled=120000,
        max_chars_per_line=512,
        reranker_epochs=2,
        reranker_train_cap=12000,
        reranker_val_cap=2500,
    ):
        self.max_order = max_order
        self.seq_weight = seq_weight
        self.labeled_weight = labeled_weight
        self.emb_dim = emb_dim
        self.ctx_window = ctx_window
        self.rerank_candidates = rerank_candidates
        self.max_neural_unlabeled = max_neural_unlabeled
        self.max_chars_per_line = max_chars_per_line
        self.reranker_epochs = reranker_epochs
        self.reranker_train_cap = reranker_train_cap
        self.reranker_val_cap = reranker_val_cap

        # Count-based LM
        self.global_counts = Counter()
        self.global_probs = {}
        self.order_counts = {k: defaultdict(Counter) for k in range(1, self.max_order + 1)}
        self.exact_context_counts = defaultdict(Counter)

        # Token / fragment models
        self.word_prefix_counts = defaultdict(Counter)
        self.word_full_counts = Counter()
        self.word_start_counts = Counter()
        self.after_word_boundary_counts = Counter()
        self.token_bigram_start_counts = defaultdict(Counter)

        # Script-aware models
        self.script_transition_counts = defaultdict(Counter)   # last informative script -> next script
        self.script_char_counts = defaultdict(Counter)         # script -> chars in that script
        self.script_global_counts = Counter()

        # Neural average-embedding LM
        self.input_emb = {}
        self.output_emb = {}
        self.output_bias = {}
        self.char_seen_count = Counter()

        # Learned reranker
        self.reranker_weights = [0.0] * len(self.FEATURE_NAMES)
        self.selected_reranker_epochs = max(1, self.reranker_epochs)

        self.alphabet = set(self.DEFAULT_CHARS)

    def _log(self, message):
        print(message, flush=True)

    def _progress_points(self, total, target_updates=20):
        if total <= 0:
            return 1
        return max(1, total // max(1, target_updates))

    def _maybe_log_progress(self, label, done, total, step, start_time):
        if done == 1 or done == total or done % step == 0:
            elapsed = time.monotonic() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (total - done) / rate if rate > 0 else 0.0
            pct = 100.0 * done / max(1, total)
            self._log(
                f"  {label}: {done}/{total} ({pct:.1f}%) | "
                f"elapsed {elapsed:.1f}s | eta {remaining:.1f}s"
            )

    # ---------- I/O ----------

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

                field_map = {
                    field.strip().lower(): field
                    for field in reader.fieldnames
                    if field and field.strip()
                }
                id_key = field_map.get('id')

                context_key = None
                for candidate in ('context', 'input', 'text', 'prompt', 'sequence', 'prefix', 'source'):
                    actual = field_map.get(candidate)
                    if actual is not None and actual != id_key:
                        context_key = actual
                        break

                if context_key is None:
                    for field in reader.fieldnames:
                        if not field:
                            continue
                        if id_key is not None and field == id_key:
                            continue
                        context_key = field
                        break

                if context_key is None:
                    raise ValueError('CSV test data must include a non-id context column')

                for i, row in enumerate(reader):
                    row_id = row.get(id_key) if id_key is not None else None
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
                    if len(pred) != 3 or ' ' in pred:
                        raise ValueError(f'Invalid prediction for id={row_id!r}: {pred!r}')
                    writer.writerow([row_id, pred])
            return

        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                if len(p) != 3 or ' ' in p:
                    raise ValueError(f'Invalid prediction: {p!r}')
                f.write(f'{p}\n')

    # ---------- Unicode / token helpers ----------

    @staticmethod
    def _is_word_char(ch):
        return ch.isalnum() or ch in "_'-/"

    @staticmethod
    def _safe_name(ch):
        try:
            return unicodedata.name(ch)
        except ValueError:
            return ''

    @classmethod
    def _char_script(cls, ch):
        if ch == cls.BOS:
            return 'COMMON'
        if ch == ' ':
            return 'COMMON'
        if ch.isdigit():
            return 'DIGIT'

        o = ord(ch)
        if 0x3040 <= o <= 0x309F:
            return 'HIRAGANA'
        if 0x30A0 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9D:
            return 'KATAKANA'
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
            return 'CJK'
        if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            return 'HANGUL'
        if 0x0400 <= o <= 0x052F or 0x2DE0 <= o <= 0x2DFF or 0xA640 <= o <= 0xA69F:
            return 'CYRILLIC'
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
            return 'ARABIC'
        if 0x0900 <= o <= 0x097F or 0xA8E0 <= o <= 0xA8FF:
            return 'DEVANAGARI'
        if 0x0370 <= o <= 0x03FF:
            return 'GREEK'
        if ch.isascii() and ch.isalpha():
            return 'LATIN'
        if 0x00C0 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF:
            return 'LATIN'

        cat = unicodedata.category(ch)
        if cat.startswith('P') or cat.startswith('S') or cat in ('Zs', 'Cc', 'Cf'):
            return 'COMMON'

        name = cls._safe_name(ch)
        if 'LATIN' in name:
            return 'LATIN'
        if 'CYRILLIC' in name:
            return 'CYRILLIC'
        if 'ARABIC' in name:
            return 'ARABIC'
        if 'DEVANAGARI' in name:
            return 'DEVANAGARI'
        if 'HIRAGANA' in name:
            return 'HIRAGANA'
        if 'KATAKANA' in name:
            return 'KATAKANA'
        if 'HANGUL' in name:
            return 'HANGUL'
        if 'GREEK' in name:
            return 'GREEK'
        if 'CJK' in name or 'IDEOGRAPH' in name:
            return 'CJK'
        return 'OTHER'

    def _last_informative_script(self, text):
        for ch in reversed(text):
            script = self._char_script(ch)
            if script not in ('COMMON', 'DIGIT'):
                return script
        return None

    def _trailing_fragment(self, text):
        i = len(text)
        while i > 0 and self._is_word_char(text[i - 1]):
            i -= 1
        return text[i:]

    def _previous_completed_word(self, text):
        i = len(text) - 1
        while i >= 0 and not self._is_word_char(text[i]):
            i -= 1
        if i < 0:
            return ''
        j = i
        while j >= 0 and self._is_word_char(text[j]):
            j -= 1
        return text[j + 1:i + 1]

    # ---------- Core counts ----------

    def _update_counts(self, context, nxt, weight=1.0):
        if not nxt:
            return

        if nxt not in self.DISALLOWED_OUTPUT_CHARS:
            self.alphabet.add(nxt)
        self.global_counts[nxt] += weight

        max_k = min(self.max_order, len(context))
        for k in range(1, max_k + 1):
            suffix = context[-k:]
            self.order_counts[k][suffix][nxt] += weight

    def _train_on_sequence(self, text, weight=1.0):
        if not text:
            return

        truncated = text[:self.max_chars_per_line]
        padded = self.BOS * self.max_order + truncated
        for i, nxt in enumerate(truncated):
            context_end = self.max_order + i
            context = padded[context_end - self.max_order:context_end]
            self._update_counts(context, nxt, weight=weight)

    def _update_word_models(self, text, weight=1.0):
        if not text:
            return

        truncated = text[:self.max_chars_per_line]
        token = []
        prev_token = None

        for ch in truncated:
            if self._is_word_char(ch):
                token.append(ch)
                if len(token) == 1:
                    self.word_start_counts[ch] += weight
                    if prev_token is not None:
                        self.token_bigram_start_counts[prev_token][ch] += weight
            else:
                if token:
                    word = ''.join(token)
                    for i, next_ch in enumerate(word):
                        prefix = word[:i]
                        self.word_prefix_counts[prefix][next_ch] += weight
                    self.word_full_counts[word] += weight
                    self.after_word_boundary_counts[ch] += weight
                    prev_token = word
                    token = []
                else:
                    prev_token = None

        if token:
            word = ''.join(token)
            for i, next_ch in enumerate(word):
                prefix = word[:i]
                self.word_prefix_counts[prefix][next_ch] += weight
            self.word_full_counts[word] += weight

    def _update_script_models(self, text, weight=1.0):
        if not text:
            return
        truncated = text[:self.max_chars_per_line]
        last_informative_script = None
        for ch in truncated:
            script = self._char_script(ch)
            self.script_char_counts[script][ch] += weight
            self.script_global_counts[script] += weight
            if last_informative_script is not None:
                self.script_transition_counts[last_informative_script][script] += weight
            if script not in ('COMMON', 'DIGIT'):
                last_informative_script = script

    def _update_script_pair(self, context, nxt, weight=1.0):
        nxt_script = self._char_script(nxt)
        self.script_char_counts[nxt_script][nxt] += weight
        self.script_global_counts[nxt_script] += weight
        last_script = self._last_informative_script(context)
        if last_script is not None:
            self.script_transition_counts[last_script][nxt_script] += weight

    # ---------- Neural LM ----------

    def _ensure_neural_char(self, ch):
        if ch not in self.input_emb:
            base = float((sum(ord(c) for c in ch) % 997) + 1)
            vec_in = []
            vec_out = []
            for i in range(self.emb_dim):
                val = math.sin(base * (i + 1) * 0.137) * 0.05
                vec_in.append(val)
                vec_out.append(math.cos(base * (i + 1) * 0.173) * 0.05)
            self.input_emb[ch] = vec_in
            self.output_emb[ch] = vec_out
            self.output_bias[ch] = 0.0

    def _sigmoid(self, x):
        if x < -35.0:
            return 0.0
        if x > 35.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def _ctx_chars_and_weights(self, context):
        tail = context[-self.ctx_window:]
        if not tail:
            return []
        pairs = []
        # More recent chars matter more.
        total = 0.0
        for i, ch in enumerate(reversed(tail)):
            w = 1.0 / (1.0 + 0.35 * i)
            pairs.append((ch, w))
            total += w
        if total <= 0.0:
            return []
        return [(ch, w / total) for ch, w in pairs]

    def _ctx_vector(self, context):
        pairs = self._ctx_chars_and_weights(context)
        vec = [0.0] * self.emb_dim
        if not pairs:
            return vec
        for ch, w in pairs:
            emb = self.input_emb.get(ch)
            if emb is None:
                continue
            for i in range(self.emb_dim):
                vec[i] += w * emb[i]
        return vec

    def _dot(self, a, b):
        return sum(x * y for x, y in zip(a, b))

    def _neural_score(self, context, ch):
        out = self.output_emb.get(ch)
        if out is None:
            return 0.0
        ctx = self._ctx_vector(context)
        return self._sigmoid(self._dot(ctx, out) + self.output_bias.get(ch, 0.0))

    def _iter_neural_examples(self, unlabeled, labeled):
        # All labeled examples are always used.
        for context, nxt in labeled:
            yield context, nxt, 1.5

        total_positions = 0
        for text in unlabeled:
            total_positions += max(0, min(len(text), self.max_chars_per_line))
        stride = max(1, total_positions // max(1, self.max_neural_unlabeled))

        for text in unlabeled:
            truncated = text[:self.max_chars_per_line]
            for i, nxt in enumerate(truncated):
                if i % stride != 0:
                    continue
                yield truncated[:i], nxt, 1.0

    def _count_neural_examples(self, unlabeled, labeled):
        total_positions = 0
        for text in unlabeled:
            total_positions += max(0, min(len(text), self.max_chars_per_line))
        stride = max(1, total_positions // max(1, self.max_neural_unlabeled))
        sampled_unlabeled = 0
        for text in unlabeled:
            truncated_len = min(len(text), self.max_chars_per_line)
            if truncated_len <= 0:
                continue
            sampled_unlabeled += (truncated_len - 1) // stride + 1
        return len(labeled) + sampled_unlabeled

    def _train_neural_lm(self, unlabeled, labeled, epochs=2, neg_samples=6, lr=0.05, progress_label='neural LM'):
        vocab = set(self.global_counts.keys())
        for context, nxt in labeled:
            vocab.add(nxt)
            for ch in context[-self.ctx_window:]:
                vocab.add(ch)
        for ch in list(vocab):
            self._ensure_neural_char(ch)
        vocab_list = [ch for ch in vocab if ch not in self.DISALLOWED_OUTPUT_CHARS]
        if len(vocab_list) <= 1:
            return

        example_total = self._count_neural_examples(unlabeled, labeled)
        step = self._progress_points(example_total)
        for epoch in range(epochs):
            self._log(f"Starting {progress_label} epoch {epoch + 1}/{epochs}")
            epoch_start = time.monotonic()
            processed = 0
            for context, true_ch, ex_weight in self._iter_neural_examples(unlabeled, labeled):
                processed += 1
                self._maybe_log_progress(
                    f"{progress_label} epoch {epoch + 1}/{epochs}",
                    processed,
                    example_total,
                    step,
                    epoch_start,
                )
                self._ensure_neural_char(true_ch)
                pairs = self._ctx_chars_and_weights(context)
                if not pairs:
                    continue

                ctx = [0.0] * self.emb_dim
                for ch, w in pairs:
                    self._ensure_neural_char(ch)
                    emb = self.input_emb[ch]
                    for i in range(self.emb_dim):
                        ctx[i] += w * emb[i]

                # Positive update
                out = self.output_emb[true_ch]
                score = self._dot(ctx, out) + self.output_bias[true_ch]
                g = ex_weight * (1.0 - self._sigmoid(score))
                out_old = out[:]
                for i in range(self.emb_dim):
                    out[i] += lr * g * ctx[i]
                self.output_bias[true_ch] += lr * g
                for ch, w in pairs:
                    inp = self.input_emb[ch]
                    for i in range(self.emb_dim):
                        inp[i] += lr * g * w * out_old[i]

                # Deterministic negative samples
                base = (sum(ord(c) for c in context[-8:]) + ord(true_ch)) % max(1, len(vocab_list))
                for n in range(neg_samples):
                    neg = vocab_list[(base + 13 * n + 7) % len(vocab_list)]
                    if neg == true_ch:
                        continue
                    out = self.output_emb[neg]
                    score = self._dot(ctx, out) + self.output_bias[neg]
                    g = -ex_weight * self._sigmoid(score)
                    out_old = out[:]
                    for i in range(self.emb_dim):
                        out[i] += lr * g * ctx[i]
                    self.output_bias[neg] += lr * g
                    for ch, w in pairs:
                        inp = self.input_emb[ch]
                        for i in range(self.emb_dim):
                            inp[i] += lr * g * w * out_old[i]

    # ---------- Finalization ----------

    def _allowed_output_chars(self):
        chars = [ch for ch in sorted(self.alphabet) if ch not in self.DISALLOWED_OUTPUT_CHARS]
        if chars:
            return chars
        fallback = [ch for ch in self.DEFAULT_CHARS if ch not in self.DISALLOWED_OUTPUT_CHARS]
        return fallback if fallback else ['e', 'a', 'i']

    def _finalize(self):
        if not self.global_counts:
            for ch in self.DEFAULT_CHARS:
                self.global_counts[ch] += 1.0
                if ch not in self.DISALLOWED_OUTPUT_CHARS:
                    self.alphabet.add(ch)

        allowed = self._allowed_output_chars()
        total = sum(self.global_counts.values())
        alpha = 0.25
        z = total + alpha * len(allowed)
        self.global_probs = {
            ch: (self.global_counts.get(ch, 0.0) + alpha) / z
            for ch in allowed
        }
        # Make sure neural vectors exist for output chars.
        for ch in allowed:
            self._ensure_neural_char(ch)

        # Reasonable initial reranker.
        self.reranker_weights = [0.0] * len(self.FEATURE_NAMES)
        self.reranker_weights[self.FEATURE_NAMES.index('bias')] = 0.0
        self.reranker_weights[self.FEATURE_NAMES.index('base_score')] = 1.0

    # ---------- Probability components ----------

    def _ngram_distribution(self, context):
        if not self.global_probs:
            self._finalize()

        dist = dict(self.global_probs)
        padded_context = (self.BOS * self.max_order + context)[-self.max_order:]
        max_k = min(self.max_order, len(padded_context))

        for k in range(1, max_k + 1):
            suffix = padded_context[-k:]
            counts = self.order_counts[k].get(suffix)
            if not counts:
                continue

            total = sum(counts.values())
            if total <= 0.0:
                continue

            distinct = len(counts)
            backoff_mass = distinct / (total + distinct)
            mle_mass = 1.0 - backoff_mass
            inv_total = 1.0 / total

            new_dist = {}
            for ch, prev_p in dist.items():
                mle = counts.get(ch, 0.0) * inv_total
                new_dist[ch] = backoff_mass * prev_p + mle_mass * mle
            dist = new_dist
        return dist

    def _word_model_distribution(self, context):
        scores = Counter()
        fragment = self._trailing_fragment(context)

        if fragment:
            counts = self.word_prefix_counts.get(fragment)
            if counts:
                total = sum(counts.values())
                if total > 0.0:
                    for ch, cnt in counts.items():
                        scores[ch] += cnt / total
        else:
            total = sum(self.word_start_counts.values())
            if total > 0.0:
                for ch, cnt in self.word_start_counts.items():
                    scores[ch] += 0.30 * (cnt / total)

            prev_word = self._previous_completed_word(context)
            if prev_word:
                counts = self.token_bigram_start_counts.get(prev_word)
                if counts:
                    total = sum(counts.values())
                    if total > 0.0:
                        for ch, cnt in counts.items():
                            scores[ch] += 0.70 * (cnt / total)
        return scores

    def _boundary_distribution(self):
        total = sum(self.after_word_boundary_counts.values())
        if total <= 0.0:
            return {}
        return {ch: cnt / total for ch, cnt in self.after_word_boundary_counts.items()}

    def _script_distribution(self, context):
        allowed = self._allowed_output_chars()
        if not allowed:
            return {}

        last_script = self._last_informative_script(context)
        if last_script is None:
            total = sum(self.script_global_counts.values())
            if total <= 0.0:
                return {}
            script_prior = {
                script: self.script_global_counts.get(script, 0.0) / total
                for script in self.SCRIPT_NAMES
            }
        else:
            counts = self.script_transition_counts.get(last_script)
            if counts:
                total = sum(counts.values())
                alpha = 0.25
                z = total + alpha * len(self.SCRIPT_NAMES)
                script_prior = {
                    script: (counts.get(script, 0.0) + alpha) / z
                    for script in self.SCRIPT_NAMES
                }
            else:
                script_prior = {last_script: 1.0}

        dist = {}
        for ch in allowed:
            script = self._char_script(ch)
            char_counts = self.script_char_counts.get(script)
            if char_counts:
                total = sum(char_counts.values())
                alpha = 0.1
                z = total + alpha * max(1, len(char_counts))
                p_char = (char_counts.get(ch, 0.0) + alpha) / z
            else:
                p_char = self.global_probs.get(ch, 0.0)
            dist[ch] = script_prior.get(script, 0.0) * p_char
        return dist

    # ---------- Candidate generation and features ----------

    def _candidate_set(self, context, extra_char=None):
        allowed = set(self._allowed_output_chars())
        candidates = []

        def add_from_scores(score_map, limit):
            if not score_map:
                return
            for ch, _ in sorted(score_map.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
                if ch in allowed and ch not in candidates:
                    candidates.append(ch)

        ngram = self._ngram_distribution(context)
        word = self._word_model_distribution(context)
        script = self._script_distribution(context)
        boundary = self._boundary_distribution()
        exact = self.exact_context_counts.get(context, {})
        if exact:
            total = sum(exact.values())
            exact_probs = {ch: cnt / total for ch, cnt in exact.items()}
        else:
            exact_probs = {}

        add_from_scores(ngram, max(24, self.rerank_candidates // 2))
        add_from_scores(exact_probs, 12)
        add_from_scores(word, 12)
        add_from_scores(script, 12)

        fragment = self._trailing_fragment(context)
        if fragment and self.word_full_counts.get(fragment, 0.0) > 0.0:
            add_from_scores(boundary, 10)

        for ch, _ in self.global_counts.most_common(32):
            if ch in allowed and ch not in candidates:
                candidates.append(ch)
            if len(candidates) >= self.rerank_candidates:
                break

        if extra_char is not None and extra_char in allowed and extra_char not in candidates:
            candidates.append(extra_char)

        for ch in self._allowed_output_chars():
            if ch not in candidates:
                candidates.append(ch)
            if len(candidates) >= self.rerank_candidates:
                break

        return candidates[:self.rerank_candidates]

    def _component_scores(self, context, candidates):
        ngram = self._ngram_distribution(context)
        word = self._word_model_distribution(context)
        script = self._script_distribution(context)
        boundary = self._boundary_distribution()

        fragment = self._trailing_fragment(context)
        use_boundary = bool(fragment and self.word_full_counts.get(fragment, 0.0) > 0.0)

        cache = self.exact_context_counts.get(context, {})
        exact_probs = {}
        if cache:
            total = sum(cache.values())
            if total > 0.0:
                exact_probs = {ch: cnt / total for ch, cnt in cache.items()}

        scores = {}
        for ch in candidates:
            scores[ch] = {
                'ngram_prob': ngram.get(ch, 0.0),
                'exact_prob': exact_probs.get(ch, 0.0),
                'word_prob': word.get(ch, 0.0),
                'boundary_prob': boundary.get(ch, 0.0) if use_boundary else 0.0,
                'script_prob': script.get(ch, 0.0),
                'neural_prob': self._neural_score(context, ch),
                'global_prob': self.global_probs.get(ch, 0.0),
            }
        return scores

    def _feature_vector(self, context, ch, comp=None):
        if comp is None:
            comp = self._component_scores(context, [ch])[ch]

        last_char = context[-1] if context else ''
        last_script = self._last_informative_script(context)
        cand_script = self._char_script(ch)

        base_score = (
            4.0 * comp['ngram_prob'] +
            2.6 * comp['exact_prob'] +
            1.2 * comp['word_prob'] +
            0.35 * comp['boundary_prob'] +
            0.70 * comp['script_prob'] +
            0.90 * comp['neural_prob'] +
            0.25 * comp['global_prob']
        )

        return [
            1.0,
            base_score,
            comp['ngram_prob'],
            comp['exact_prob'],
            comp['word_prob'],
            comp['boundary_prob'],
            comp['script_prob'],
            comp['neural_prob'],
            comp['global_prob'],
            1.0 if ch == last_char and last_char else 0.0,
            1.0 if (last_script is not None and cand_script == last_script) else 0.0,
            1.0 if cand_script in ('COMMON', 'DIGIT') else 0.0,
            1.0 if len(context) >= 1 and ch == context[-1] else 0.0,
            1.0 if len(context) >= 2 and context[-2:] == ch * 2 else 0.0,
        ]

    def _score_with_weights(self, weights, feat):
        return sum(w * x for w, x in zip(weights, feat))

    def _rank_candidates(self, context, candidates, weights=None):
        if weights is None:
            weights = self.reranker_weights
        comp_scores = self._component_scores(context, candidates)
        scored = []
        for ch in candidates:
            feat = self._feature_vector(context, ch, comp_scores[ch])
            score = self._score_with_weights(weights, feat)
            # Small fixed penalties for degenerate repetition.
            if context:
                if len(context) >= 1 and ch == context[-1] and ch in '.,;:!?':
                    score -= 0.03
                if len(context) >= 2 and context[-1] == context[-2] == ch:
                    score -= 0.05
            scored.append((score, ch, feat))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored

    def _top_guesses(self, context, n=3):
        candidates = self._candidate_set(context)
        ranked = self._rank_candidates(context, candidates)
        preds = []
        for _, ch, _ in ranked:
            if ch in self.DISALLOWED_OUTPUT_CHARS:
                continue
            if ch not in preds:
                preds.append(ch)
            if len(preds) >= n:
                break
        while len(preds) < n:
            for ch in self._allowed_output_chars():
                if ch not in preds:
                    preds.append(ch)
                if len(preds) >= n:
                    break
        return ''.join(preds[:n])

    # ---------- Fast reranker training ----------

    @staticmethod
    def _stable_hash(text):
        h = 2166136261
        for ch in text:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def _split_labeled(self, labeled):
        if len(labeled) < 40:
            return labeled, []
        train = []
        val = []
        for context, nxt in labeled:
            bucket = self._stable_hash(context + '\u241f' + nxt) % 10
            if bucket == 0:
                val.append((context, nxt))
            else:
                train.append((context, nxt))
        if not train or not val:
            cut = max(1, len(labeled) // 10)
            return labeled[cut:], labeled[:cut]
        return train, val

    def _subsample_examples(self, examples, cap):
        if cap is None or cap <= 0 or len(examples) <= cap:
            return list(examples)
        ordered = sorted(
            examples,
            key=lambda item: (self._stable_hash(item[0] + '\u241f' + item[1]), len(item[0]))
        )
        stride = max(1, len(ordered) // cap)
        picked = ordered[::stride][:cap]
        if len(picked) < cap:
            seen = set(picked)
            for ex in ordered:
                if ex not in seen:
                    picked.append(ex)
                    seen.add(ex)
                if len(picked) >= cap:
                    break
        return picked[:cap]

    def _build_reranker_cache(self, examples, cap=None, progress_label='reranker cache'):
        sampled = self._subsample_examples(examples, cap)
        if not sampled:
            return []
        self._log(f"Building {progress_label} from {len(sampled)} example(s)")
        step = self._progress_points(len(sampled))
        start_time = time.monotonic()
        cached = []
        for idx, (context, gold) in enumerate(sampled, start=1):
            self._maybe_log_progress(progress_label, idx, len(sampled), step, start_time)
            candidates = self._candidate_set(context, extra_char=gold)[: self.rerank_candidates]
            comp_scores = self._component_scores(context, candidates)
            items = []
            seen = set()
            for ch in candidates:
                if ch in seen:
                    continue
                seen.add(ch)
                items.append((ch, self._feature_vector(context, ch, comp_scores[ch])))
            if gold not in seen and gold not in self.DISALLOWED_OUTPUT_CHARS:
                gold_comp = self._component_scores(context, [gold])[gold]
                items.append((gold, self._feature_vector(context, gold, gold_comp)))
            cached.append((gold, items))
        return cached

    def _rank_cached_items(self, items, weights):
        scored = []
        for ch, feat in items:
            scored.append((self._score_with_weights(weights, feat), ch, feat))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored

    def _mrr3_cached(self, cached_examples, weights):
        if not cached_examples:
            return 0.0
        total = 0.0
        for gold, items in cached_examples:
            ranked = self._rank_cached_items(items, weights)
            top3 = [ch for _, ch, _ in ranked[:3]]
            if gold in top3:
                total += 1.0 / (top3.index(gold) + 1.0)
        return total / len(cached_examples)

    def _train_reranker_cached(self, train_cache, val_cache, epochs, progress_label='fast reranker'):
        if not train_cache:
            return self.reranker_weights[:], 0.0, 0

        weights = self.reranker_weights[:]
        avg = [0.0] * len(weights)
        count = 0
        best_weights = weights[:]
        best_score = -1.0
        best_epoch = 1

        total_epochs = max(1, epochs)
        for epoch in range(total_epochs):
            self._log(f"Starting {progress_label} epoch {epoch + 1}/{total_epochs}")
            ordered = sorted(
                train_cache,
                key=lambda item: (-len(item[1]), self._stable_hash(item[0]))
            )
            step = self._progress_points(len(ordered))
            start_time = time.monotonic()
            for idx, (gold, items) in enumerate(ordered, start=1):
                self._maybe_log_progress(
                    f"{progress_label} epoch {epoch + 1}/{total_epochs}",
                    idx,
                    len(ordered),
                    step,
                    start_time,
                )
                ranked = self._rank_cached_items(items, weights)
                top3 = [ch for _, ch, _ in ranked[:3]]
                if gold not in top3:
                    gold_feat = None
                    pred_feat = ranked[0][2]
                    for _, ch, feat in ranked:
                        if ch == gold:
                            gold_feat = feat
                            break
                    if gold_feat is not None:
                        for i in range(len(weights)):
                            weights[i] += gold_feat[i] - pred_feat[i]
                for i in range(len(weights)):
                    avg[i] += weights[i]
                count += 1

            eval_weights = [x / max(1, count) for x in avg]
            score = self._mrr3_cached(val_cache if val_cache else train_cache[: min(1000, len(train_cache))], eval_weights)
            self._log(f"Completed {progress_label} epoch {epoch + 1}/{total_epochs}: MRR@3={score:.4f}")
            if score > best_score:
                best_score = score
                best_weights = eval_weights[:]
                best_epoch = epoch + 1

        return best_weights, best_score, best_epoch

    # ---------- Training ----------

    def _fit_components(self, unlabeled, labeled, progress_label='fit components'):
        self._log(f"Starting {progress_label}: count-based passes over unlabeled contexts")
        unlabeled_step = self._progress_points(len(unlabeled))
        unlabeled_start = time.monotonic()
        for idx, text in enumerate(unlabeled, start=1):
            self._train_on_sequence(text, weight=self.seq_weight)
            self._update_word_models(text, weight=self.seq_weight)
            self._update_script_models(text, weight=self.seq_weight)
            self._maybe_log_progress(f"{progress_label} unlabeled pass", idx, len(unlabeled), unlabeled_step, unlabeled_start)

        self._log(f"Starting {progress_label}: supervised pass over labeled pairs")
        labeled_step = self._progress_points(len(labeled))
        labeled_start = time.monotonic()
        for idx, (context, nxt) in enumerate(labeled, start=1):
            padded_context = (self.BOS * self.max_order + context)[-self.max_order:]
            self._update_counts(padded_context, nxt, weight=self.labeled_weight)
            self.exact_context_counts[context][nxt] += self.labeled_weight
            self._update_word_models(context + nxt, weight=1.0)
            self._update_script_pair(context, nxt, weight=self.labeled_weight)
            self._maybe_log_progress(f"{progress_label} labeled pass", idx, len(labeled), labeled_step, labeled_start)

        self._log(f"Starting {progress_label}: finalizing distributions")
        self._finalize()
        self._train_neural_lm(unlabeled, labeled, progress_label=f"{progress_label} neural LM")

    def run_train(self, train_data, work_dir):
        labeled = list(train_data.get('labeled', []))
        unlabeled = list(train_data.get('unlabeled', []))

        self._log('Fitting base model on all unlabeled and labeled data')
        self._fit_components(unlabeled, labeled, progress_label='base fit')

        self._log('Preparing lightweight held-out reranker split')
        train_split, val_split = self._split_labeled(labeled)
        self._log(f"Reranker train split: {len(train_split)} | validation split: {len(val_split)}")

        train_cache = self._build_reranker_cache(
            train_split,
            cap=self.reranker_train_cap,
            progress_label='reranker train cache',
        )
        val_cache = self._build_reranker_cache(
            val_split,
            cap=self.reranker_val_cap,
            progress_label='reranker validation cache',
        )

        self.reranker_weights, best_score, best_epoch = self._train_reranker_cached(
            train_cache,
            val_cache,
            self.reranker_epochs,
            progress_label='fast reranker',
        )
        self.selected_reranker_epochs = best_epoch
        self._log(
            f"Finished fast reranker: selected epoch {best_epoch} | "
            f"best held-out MRR@3={best_score:.4f} | "
            f"train cache={len(train_cache)} | val cache={len(val_cache)}"
        )

    # ---------- Prediction / persistence ----------

    def run_pred(self, data):
        return [self._top_guesses(inp) for inp in data]

    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        payload = {
            'model_version': self.MODEL_VERSION,
            'max_order': self.max_order,
            'seq_weight': self.seq_weight,
            'labeled_weight': self.labeled_weight,
            'emb_dim': self.emb_dim,
            'ctx_window': self.ctx_window,
            'rerank_candidates': self.rerank_candidates,
            'max_neural_unlabeled': self.max_neural_unlabeled,
            'max_chars_per_line': self.max_chars_per_line,
            'reranker_epochs': self.reranker_epochs,
            'reranker_train_cap': self.reranker_train_cap,
            'reranker_val_cap': self.reranker_val_cap,
            'selected_reranker_epochs': self.selected_reranker_epochs,
            'global_counts': dict(self.global_counts),
            'order_counts': {
                str(k): {suffix: dict(cnts) for suffix, cnts in suffix_map.items()}
                for k, suffix_map in self.order_counts.items()
            },
            'exact_context_counts': {
                ctx: dict(cnts) for ctx, cnts in self.exact_context_counts.items()
            },
            'word_prefix_counts': {
                prefix: dict(cnts) for prefix, cnts in self.word_prefix_counts.items()
            },
            'word_full_counts': dict(self.word_full_counts),
            'word_start_counts': dict(self.word_start_counts),
            'after_word_boundary_counts': dict(self.after_word_boundary_counts),
            'token_bigram_start_counts': {
                tok: dict(cnts) for tok, cnts in self.token_bigram_start_counts.items()
            },
            'script_transition_counts': {
                s: dict(cnts) for s, cnts in self.script_transition_counts.items()
            },
            'script_char_counts': {
                s: dict(cnts) for s, cnts in self.script_char_counts.items()
            },
            'script_global_counts': dict(self.script_global_counts),
            'input_emb': self.input_emb,
            'output_emb': self.output_emb,
            'output_bias': self.output_bias,
            'alphabet': sorted(self.alphabet),
            'reranker_weights': self.reranker_weights,
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
            max_order=payload.get('max_order', 10),
            seq_weight=payload.get('seq_weight', 1.0),
            labeled_weight=payload.get('labeled_weight', 4.0),
            emb_dim=payload.get('emb_dim', 16),
            ctx_window=payload.get('ctx_window', 16),
            rerank_candidates=payload.get('rerank_candidates', 64),
            max_neural_unlabeled=payload.get('max_neural_unlabeled', 120000),
            max_chars_per_line=payload.get('max_chars_per_line', 512),
            reranker_epochs=payload.get('reranker_epochs', 2),
            reranker_train_cap=payload.get('reranker_train_cap', 12000),
            reranker_val_cap=payload.get('reranker_val_cap', 2500),
        )
        model.selected_reranker_epochs = payload.get('selected_reranker_epochs', model.reranker_epochs)

        model.global_counts.update(payload.get('global_counts', {}))
        for key, suffix_map in payload.get('order_counts', {}).items():
            k = int(key)
            for suffix, cnts in suffix_map.items():
                model.order_counts[k][suffix].update(cnts)

        for ctx, cnts in payload.get('exact_context_counts', {}).items():
            model.exact_context_counts[ctx].update(cnts)
        for prefix, cnts in payload.get('word_prefix_counts', {}).items():
            model.word_prefix_counts[prefix].update(cnts)
        model.word_full_counts.update(payload.get('word_full_counts', {}))
        model.word_start_counts.update(payload.get('word_start_counts', {}))
        model.after_word_boundary_counts.update(payload.get('after_word_boundary_counts', {}))
        for tok, cnts in payload.get('token_bigram_start_counts', {}).items():
            model.token_bigram_start_counts[tok].update(cnts)

        for s, cnts in payload.get('script_transition_counts', {}).items():
            model.script_transition_counts[s].update(cnts)
        for s, cnts in payload.get('script_char_counts', {}).items():
            model.script_char_counts[s].update(cnts)
        model.script_global_counts.update(payload.get('script_global_counts', {}))

        model.input_emb = {
            ch: [float(v) for v in vec] for ch, vec in payload.get('input_emb', {}).items()
        }
        model.output_emb = {
            ch: [float(v) for v in vec] for ch, vec in payload.get('output_emb', {}).items()
        }
        model.output_bias = {
            ch: float(v) for ch, v in payload.get('output_bias', {}).items()
        }

        model.alphabet.update(payload.get('alphabet', []))
        model._finalize()
        saved_weights = payload.get('reranker_weights')
        if saved_weights and len(saved_weights) == len(model.FEATURE_NAMES):
            model.reranker_weights = [float(v) for v in saved_weights]
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='input/test.csv')
    parser.add_argument('--test_output', help='path to write test predictions', default='submission.csv')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Making working directory {args.work_dir}')
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print(
            f"Using {len(train_data.get('unlabeled', []))} unlabeled contexts and "
            f"{len(train_data.get('labeled', []))} labeled completion pairs"
        )
        model.run_train(train_data, args.work_dir)
        print(f'Best reranker epochs from held-out split: {model.selected_reranker_epochs}')
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
