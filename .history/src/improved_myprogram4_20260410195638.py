#!/usr/bin/env python
import csv
import json
import os
import time
import unicodedata
import math
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path


class MyModel:
    """
    High-performance multilingual next-character prediction model.

    Key improvements over v4:
    - Modified Kneser-Ney smoothing for n-gram model
    - Higher order n-grams (max_order=15) for better context matching
    - Context-adaptive ensemble: dynamically adjusts weights based on
      whether we're mid-word, at word boundary, after punctuation, etc.
    - Case prediction heuristics (after ". " -> uppercase, match case patterns)
    - Common English suffix/pattern priors
    - Improved word completion with longer prefix matching
    - Bigram word model for predicting word starts after known words
    - Efficient candidate generation and scoring
    """

    MODEL_FILE = 'model.checkpoint'
    MODEL_VERSION = 60

    BOS = '\u0002'
    DISALLOWED_OUTPUT_CHARS = {' ', BOS}

    DEFAULT_CHARS = list(
        "etaoinshrdlucmfwypvbgkqjxz"
        "ETAOINSHRDLUCMFWYPVBGKQJXZ"
        "0123456789"
        ".,!?;:'\"-_/()[]{}@#$%&*+=<>\\|`~"
    )

    # Common English suffixes for pattern boosting
    COMMON_SUFFIXES = [
        'ing', 'tion', 'ment', 'ness', 'able', 'ible', 'ous', 'ive',
        'ful', 'less', 'ence', 'ance', 'ally', 'erly', 'ated', 'ting',
        'ted', 'ers', 'est', 'ies', 'ize', 'ise', 'ory', 'ary',
        'the', 'and', 'ion', 'ent', 'ant', 'ate', 'ure', 'ity',
    ]

    # Common word starters after space
    COMMON_WORD_STARTS = {
        't': 0.15, 'a': 0.10, 's': 0.08, 'w': 0.07, 'i': 0.07,
        'c': 0.06, 'h': 0.06, 'b': 0.05, 'o': 0.05, 'f': 0.05,
        'd': 0.05, 'p': 0.05, 'm': 0.04, 'r': 0.04, 'n': 0.04,
        'l': 0.03, 'e': 0.03, 'g': 0.03, 'u': 0.02, 'v': 0.02,
        'T': 0.04, 'A': 0.03, 'S': 0.03, 'I': 0.04, 'W': 0.03,
        'C': 0.02, 'H': 0.02, 'B': 0.02, 'O': 0.02, 'F': 0.02,
    }

    def __init__(
        self,
        max_order=15,
        seq_weight=1.0,
        labeled_weight=5.0,
        max_chars_per_line=600,
        progress_updates=20,
    ):
        self.max_order = max_order
        self.seq_weight = seq_weight
        self.labeled_weight = labeled_weight
        self.max_chars_per_line = max_chars_per_line
        self.progress_updates = progress_updates

        # Core n-gram counts
        self.global_counts = Counter()
        self.global_probs = {}
        self.order_counts = {k: defaultdict(Counter) for k in range(1, self.max_order + 1)}
        # For Kneser-Ney: track continuation counts
        self.order_unique_follows = {k: defaultdict(int) for k in range(1, self.max_order + 1)}

        self.exact_context_counts = defaultdict(Counter)

        # Word-level models
        self.word_prefix_counts = defaultdict(Counter)
        self.word_full_counts = Counter()
        self.word_start_counts = Counter()
        self.after_word_boundary_counts = Counter()
        self.token_bigram_start_counts = defaultdict(Counter)

        # Case pattern model
        self.case_after_pattern = defaultdict(Counter)  # pattern -> {upper/lower/other: count}

        # Script-aware models
        self.script_transition_counts = defaultdict(Counter)
        self.script_char_counts = defaultdict(Counter)
        self.script_global_counts = Counter()

        # Character pair model (what chars follow what chars)
        self.char_pair_counts = defaultdict(Counter)

        self.alphabet = set(self.DEFAULT_CHARS)

    def _log(self, msg):
        print(msg, flush=True)

    def _progress_step(self, total):
        return max(1, total // max(1, self.progress_updates))

    def _maybe_log_progress(self, label, done, total, step, start_time):
        if done == 1 or done == total or done % step == 0:
            elapsed = time.monotonic() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else 0.0
            pct = 100.0 * done / max(1, total)
            self._log(
                f"  {label}: {done}/{total} ({pct:.1f}%) | elapsed {elapsed:.1f}s | eta {eta:.1f}s"
            )

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
            for pred in preds:
                if len(pred) != 3 or ' ' in pred:
                    raise ValueError(f'Invalid prediction: {pred!r}')
                f.write(f'{pred}\n')

    # ── Character utilities ──

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
        if ch == cls.BOS or ch == ' ':
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
        if cat.startswith('P') or cat.startswith('S'):
            return 'COMMON'
        return 'OTHER'

    def _last_informative_char(self, text):
        for ch in reversed(text):
            if ch != ' ' and ch != self.BOS:
                return ch
        return ''

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

    def _allowed_output_chars(self):
        chars = [ch for ch in sorted(self.alphabet) if ch not in self.DISALLOWED_OUTPUT_CHARS]
        if chars:
            return chars
        return [ch for ch in self.DEFAULT_CHARS if ch not in self.DISALLOWED_OUTPUT_CHARS] or ['e', 'a', 'i']

    # ── Case analysis ──

    @staticmethod
    def _case_pattern(context):
        """Determine the case pattern at end of context for prediction."""
        if not context:
            return 'start'

        # Check for sentence boundary patterns
        stripped = context.rstrip()
        if not stripped:
            return 'start'

        last_ch = stripped[-1]

        # After sentence-ending punctuation + space(s) -> likely uppercase
        if context.endswith(' ') or context.endswith('\n'):
            if last_ch in '.!?':
                return 'sentence_start'
            if last_ch == ':':
                return 'after_colon'
            if last_ch == ',':
                return 'after_comma'

        # Check if we're mid-word and can infer case
        trailing = ''
        for ch in reversed(context):
            if ch.isalpha():
                trailing = ch + trailing
            else:
                break

        if trailing:
            if trailing.isupper() and len(trailing) >= 2:
                return 'all_upper'
            if trailing[0].isupper() and trailing[1:].islower() and len(trailing) >= 2:
                return 'title_case'
            if trailing.islower():
                return 'lower'

        return 'other'

    def _case_boost(self, context, ch):
        """Return a multiplier for case-based boosting."""
        if not ch.isalpha():
            return 1.0

        pattern = self._case_pattern(context)

        if pattern == 'sentence_start' or pattern == 'start':
            # After sentence boundary, boost uppercase
            if ch.isupper():
                return 1.8
            return 0.6
        elif pattern == 'all_upper':
            # In an all-caps word, strongly boost uppercase
            if ch.isupper():
                return 2.5
            return 0.3
        elif pattern == 'title_case':
            # After title case start, boost lowercase (rest of word)
            if ch.islower():
                return 1.3
            return 0.8
        elif pattern == 'lower':
            # Mid lowercase word, boost lowercase
            if ch.islower():
                return 1.2
            return 0.7
        elif pattern == 'after_colon':
            # Could be either
            if ch.isupper():
                return 1.3
            return 0.9

        return 1.0

    # ── Training ──

    def _update_counts(self, context, nxt, weight=1.0):
        if not nxt:
            return
        self.alphabet.add(nxt)
        self.global_counts[nxt] += weight
        max_k = min(self.max_order, len(context))
        for k in range(1, max_k + 1):
            suffix = context[-k:]
            counts = self.order_counts[k][suffix]
            if nxt not in counts:
                self.order_unique_follows[k][suffix] += 1
            counts[nxt] += weight

    def _update_char_pairs(self, context, nxt, weight=1.0):
        if not nxt or not context:
            return
        last = context[-1]
        self.char_pair_counts[last][nxt] += weight
        if len(context) >= 2:
            bigram = context[-2:]
            self.char_pair_counts[bigram][nxt] += weight

    def _update_script_models(self, context, nxt, weight=1.0):
        if not nxt:
            return
        next_script = self._char_script(nxt)
        self.script_global_counts[next_script] += weight
        self.script_char_counts[next_script][nxt] += weight
        last_ch = self._last_informative_char(context)
        if last_ch:
            prev_script = self._char_script(last_ch)
            self.script_transition_counts[prev_script][next_script] += weight

    def _update_case_pattern(self, context, nxt, weight=1.0):
        if not nxt or not nxt.isalpha():
            return
        pattern = self._case_pattern(context)
        case_type = 'upper' if nxt.isupper() else 'lower'
        self.case_after_pattern[pattern][case_type] += weight

    def _train_on_sequence(self, text, weight=1.0):
        if not text:
            return
        truncated = text[:self.max_chars_per_line]
        padded = self.BOS * self.max_order + truncated
        for i, nxt in enumerate(truncated):
            context_end = self.max_order + i
            context = padded[context_end - self.max_order:context_end]
            self._update_counts(context, nxt, weight=weight)
            raw_context = context.replace(self.BOS, '')
            self._update_script_models(raw_context, nxt, weight=weight)
            self._update_char_pairs(raw_context, nxt, weight=weight)
            self._update_case_pattern(raw_context, nxt, weight=weight)

    def _update_word_models(self, text, weight=1.0):
        if not text:
            return
        truncated = text[:self.max_chars_per_line]
        token = []
        prev_token = None

        for ch in truncated:
            self.alphabet.add(ch)
            if self._is_word_char(ch):
                token.append(ch)
                if len(token) == 1:
                    self.word_start_counts[ch] += weight
                    if prev_token is not None:
                        self.token_bigram_start_counts[prev_token.lower()][ch] += weight
            else:
                if token:
                    word = ''.join(token)
                    for i, next_ch in enumerate(word):
                        prefix = word[:i]
                        self.word_prefix_counts[prefix.lower()][next_ch] += weight
                    self.word_full_counts[word.lower()] += weight
                    self.after_word_boundary_counts[ch] += weight
                    prev_token = word
                    token = []
                else:
                    prev_token = None

        if token:
            word = ''.join(token)
            for i, next_ch in enumerate(word):
                prefix = word[:i]
                self.word_prefix_counts[prefix.lower()][next_ch] += weight
            self.word_full_counts[word.lower()] += weight

    def _finalize(self):
        if not self.global_counts:
            for ch in self.DEFAULT_CHARS:
                self.global_counts[ch] += 1.0
                self.alphabet.add(ch)
        alphabet = sorted(self.alphabet)
        total = sum(self.global_counts.values())
        alpha = 0.25
        z = total + alpha * len(alphabet)
        self.global_probs = {
            ch: (self.global_counts.get(ch, 0.0) + alpha) / z
            for ch in alphabet
        }

    def run_train(self, train_data, work_dir):
        unlabeled = train_data.get('unlabeled', [])
        labeled = train_data.get('labeled', [])

        if unlabeled:
            self._log('Starting dense unlabeled LM pass')
            step = self._progress_step(len(unlabeled))
            start = time.monotonic()
            for i, text in enumerate(unlabeled, 1):
                self._train_on_sequence(text, weight=self.seq_weight)
                self._update_word_models(text, weight=self.seq_weight)
                self._maybe_log_progress('unlabeled LM pass', i, len(unlabeled), step, start)

        if labeled:
            self._log('Starting labeled completion pass')
            step = self._progress_step(len(labeled))
            start = time.monotonic()
            for i, (context, nxt) in enumerate(labeled, 1):
                padded_context = (self.BOS * self.max_order + context)[-self.max_order:]
                self._update_counts(padded_context, nxt, weight=self.labeled_weight)
                self._update_script_models(context, nxt, weight=self.labeled_weight)
                self._update_char_pairs(context, nxt, weight=self.labeled_weight)
                self._update_case_pattern(context, nxt, weight=self.labeled_weight)
                self.exact_context_counts[context][nxt] += self.labeled_weight
                self._update_word_models(context + nxt, weight=1.5)
                self._maybe_log_progress('labeled completion pass', i, len(labeled), step, start)

        self._finalize()

    # ── Prediction distributions ──

    def _ngram_distribution(self, context):
        """Modified Kneser-Ney smoothed n-gram distribution."""
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
            if total <= 0:
                continue

            # Modified Kneser-Ney discount
            n_unique = self.order_unique_follows[k].get(suffix, len(counts))
            if total < 3:
                discount = 0.1
            elif total < 10:
                discount = 0.5
            else:
                discount = 0.75

            # Interpolation weight for backoff
            backoff_weight = discount * n_unique / total
            backoff_weight = min(backoff_weight, 0.5)  # cap backoff

            new_dist = {}
            for ch, prev_p in dist.items():
                cnt = counts.get(ch, 0.0)
                mle = max(0.0, cnt - discount) / total if cnt > 0 else 0.0
                new_dist[ch] = backoff_weight * prev_p + mle

            # Add chars that are in counts but not yet in dist
            for ch, cnt in counts.items():
                if ch not in new_dist:
                    mle = max(0.0, cnt - discount) / total
                    new_dist[ch] = mle

            # Renormalize
            norm = sum(new_dist.values())
            if norm > 0:
                inv_norm = 1.0 / norm
                dist = {ch: p * inv_norm for ch, p in new_dist.items()}
            else:
                dist = new_dist

        return dist

    def _word_model_distribution(self, context):
        scores = Counter()
        fragment = self._trailing_fragment(context)
        if fragment:
            # We're mid-word: use word prefix model
            lower_frag = fragment.lower()
            counts = self.word_prefix_counts.get(lower_frag)
            if counts:
                total = sum(counts.values())
                if total > 0:
                    inv = 1.0 / total
                    for ch, cnt in counts.items():
                        scores[ch] += cnt * inv

            # Also check if this fragment matches any suffix pattern
            for suffix in self.COMMON_SUFFIXES:
                if suffix.startswith(lower_frag) and len(suffix) > len(lower_frag):
                    next_ch = suffix[len(lower_frag)]
                    scores[next_ch] += 0.05
        else:
            # At word boundary: predict word start
            total = sum(self.word_start_counts.values())
            if total > 0:
                inv = 1.0 / total
                for ch, cnt in self.word_start_counts.items():
                    scores[ch] += 0.35 * cnt * inv
            prev_word = self._previous_completed_word(context)
            if prev_word:
                counts = self.token_bigram_start_counts.get(prev_word.lower())
                if counts:
                    total = sum(counts.values())
                    if total > 0:
                        inv = 1.0 / total
                        for ch, cnt in counts.items():
                            scores[ch] += 0.65 * cnt * inv
        return scores

    def _char_pair_distribution(self, context):
        """Distribution based on character pair statistics."""
        scores = Counter()
        if not context:
            return scores

        # Single char context
        last = context[-1]
        counts1 = self.char_pair_counts.get(last)
        if counts1:
            total = sum(counts1.values())
            if total > 0:
                inv = 1.0 / total
                for ch, cnt in counts1.items():
                    scores[ch] += 0.4 * cnt * inv

        # Bigram context
        if len(context) >= 2:
            bigram = context[-2:]
            counts2 = self.char_pair_counts.get(bigram)
            if counts2:
                total = sum(counts2.values())
                if total > 0:
                    inv = 1.0 / total
                    for ch, cnt in counts2.items():
                        scores[ch] += 0.6 * cnt * inv

        return scores

    def _boundary_distribution(self):
        total = sum(self.after_word_boundary_counts.values())
        if total <= 0:
            return {}
        inv = 1.0 / total
        return {ch: cnt * inv for ch, cnt in self.after_word_boundary_counts.items()}

    def _script_distribution(self, context):
        scores = Counter()
        last_ch = self._last_informative_char(context)
        if last_ch:
            prev_script = self._char_script(last_ch)
            trans = self.script_transition_counts.get(prev_script)
            if trans:
                total_t = sum(trans.values())
                if total_t > 0:
                    for script, tcnt in trans.items():
                        chars = self.script_char_counts.get(script)
                        if not chars:
                            continue
                        total_c = sum(chars.values())
                        if total_c <= 0:
                            continue
                        script_mass = tcnt / total_t
                        for ch, ccnt in chars.items():
                            scores[ch] += script_mass * (ccnt / total_c)
                return scores

        total_g = sum(self.script_global_counts.values())
        if total_g <= 0:
            return scores
        for script, scnt in self.script_global_counts.items():
            chars = self.script_char_counts.get(script)
            if not chars:
                continue
            total_c = sum(chars.values())
            if total_c <= 0:
                continue
            script_mass = scnt / total_g
            for ch, ccnt in chars.items():
                scores[ch] += script_mass * (ccnt / total_c)
        return scores

    # ── Context-adaptive ensemble ──

    def _get_ensemble_weights(self, context):
        """Dynamically determine ensemble weights based on context."""
        fragment = self._trailing_fragment(context)
        at_word_boundary = len(fragment) == 0

        # Base weights
        w = {
            'ngram': 1.00,
            'exact': 2.00,
            'word': 0.80,
            'char_pair': 0.30,
            'boundary': 0.10,
            'script': 0.35,
            'global': 0.08,
        }

        if fragment:
            frag_len = len(fragment)
            if frag_len <= 2:
                # Short fragment: word model is very useful
                w['word'] = 1.50
                w['ngram'] = 0.90
            elif frag_len <= 5:
                # Medium fragment: balanced
                w['word'] = 1.20
                w['ngram'] = 1.00
            else:
                # Long fragment: n-gram knows more
                w['word'] = 0.60
                w['ngram'] = 1.20

            # Check if word prefix model has good data
            lower_frag = fragment.lower()
            prefix_counts = self.word_prefix_counts.get(lower_frag)
            if prefix_counts and sum(prefix_counts.values()) > 10:
                w['word'] *= 1.3  # boost word model when we have good data

        else:
            # At word boundary
            w['boundary'] = 0.25
            w['word'] = 0.90
            # If after sentence-ending punct, boost case-aware predictions
            stripped = context.rstrip() if context else ''
            if stripped and stripped[-1] in '.!?':
                w['ngram'] = 1.20

        # Exact context match is always very valuable when available
        if context in self.exact_context_counts:
            total = sum(self.exact_context_counts[context].values())
            if total > 5:
                w['exact'] = 3.00
            elif total > 1:
                w['exact'] = 2.50

        return w

    # ── Candidate scoring ──

    def _candidate_pool(self, context, distributions, limit=64):
        """Build candidate pool from all distributions."""
        merged = Counter()
        for name, (dist, wt) in distributions.items():
            for ch, score in dist.items():
                if ch in self.DISALLOWED_OUTPUT_CHARS:
                    continue
                merged[ch] += wt * score

        # Add global prior
        for ch, score in self.global_probs.items():
            if ch in self.DISALLOWED_OUTPUT_CHARS:
                continue
            merged[ch] += 0.02 * score

        ranked = [ch for ch, _ in merged.most_common(limit)]
        if len(ranked) < limit:
            for ch, _ in self.global_counts.most_common(limit * 2):
                if ch not in self.DISALLOWED_OUTPUT_CHARS and ch not in set(ranked):
                    ranked.append(ch)
                if len(ranked) >= limit:
                    break
        return ranked

    def _score_candidate(self, context, ch, distributions, weights):
        score = 0.0

        for name, (dist, _) in distributions.items():
            w = weights.get(name, 0.0)
            score += w * dist.get(ch, 0.0)

        score += weights.get('global', 0.08) * self.global_probs.get(ch, 0.0)

        # Case boost
        case_mult = self._case_boost(context, ch)
        score *= case_mult

        # Penalties for repeated characters
        if context:
            last = context[-1]
            if ch == last:
                score *= 0.80
            if len(context) >= 2 and context[-1] == context[-2] == ch:
                score *= 0.50  # triple char very unlikely

            # Script continuity bonus
            last_script = self._char_script(last)
            ch_script = self._char_script(ch)
            if last_script not in ('COMMON', 'DIGIT') and ch_script == last_script:
                score += 0.015

        return score

    def _top_guesses(self, context, n=3):
        # Build all distributions
        ngram = self._ngram_distribution(context)
        word = self._word_model_distribution(context)
        char_pair = self._char_pair_distribution(context)
        script = self._script_distribution(context)

        boundary = {}
        fragment = self._trailing_fragment(context)
        if fragment and self.word_full_counts.get(fragment.lower(), 0.0) > 0.0:
            boundary = self._boundary_distribution()

        exact = {}
        cache = self.exact_context_counts.get(context)
        if cache:
            total = sum(cache.values())
            if total > 0:
                exact = {ch: cnt / total for ch, cnt in cache.items()}

        distributions = {
            'ngram': (ngram, 1.0),
            'exact': (exact, 1.0),
            'word': (word, 1.0),
            'char_pair': (char_pair, 1.0),
            'boundary': (boundary, 1.0),
            'script': (script, 1.0),
        }

        weights = self._get_ensemble_weights(context)
        candidates = self._candidate_pool(context, distributions, limit=64)

        scored = []
        for ch in candidates:
            s = self._score_candidate(context, ch, distributions, weights)
            scored.append((s, ch))
        scored.sort(key=lambda x: (-x[0], x[1]))

        ranked = []
        seen = set()
        for _, ch in scored:
            if ch not in seen and ch not in self.DISALLOWED_OUTPUT_CHARS:
                ranked.append(ch)
                seen.add(ch)
            if len(ranked) >= n:
                break

        if len(ranked) < n:
            for ch in self._allowed_output_chars():
                if ch not in seen:
                    ranked.append(ch)
                    seen.add(ch)
                if len(ranked) >= n:
                    break

        return ''.join(ranked[:n])

    def run_pred(self, data):
        preds = []
        total = len(data)
        step = self._progress_step(total)
        start = time.monotonic()
        for i, inp in enumerate(data, 1):
            preds.append(self._top_guesses(inp))
            if i % step == 0 or i == total:
                elapsed = time.monotonic() - start
                rate = i / elapsed if elapsed > 0 else 0
                self._log(f"  predict: {i}/{total} ({100*i/total:.1f}%) | {rate:.0f} ctx/s")
        return preds

    # ── Save / Load ──

    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        payload = {
            'model_version': self.MODEL_VERSION,
            'max_order': self.max_order,
            'seq_weight': self.seq_weight,
            'labeled_weight': self.labeled_weight,
            'max_chars_per_line': self.max_chars_per_line,
            'global_counts': dict(self.global_counts),
            'order_counts': {
                str(k): {suffix: dict(cnts) for suffix, cnts in suffix_map.items()}
                for k, suffix_map in self.order_counts.items()
            },
            'order_unique_follows': {
                str(k): dict(counts) for k, counts in self.order_unique_follows.items()
            },
            'exact_context_counts': {ctx: dict(cnts) for ctx, cnts in self.exact_context_counts.items()},
            'word_prefix_counts': {prefix: dict(cnts) for prefix, cnts in self.word_prefix_counts.items()},
            'word_full_counts': dict(self.word_full_counts),
            'word_start_counts': dict(self.word_start_counts),
            'after_word_boundary_counts': dict(self.after_word_boundary_counts),
            'token_bigram_start_counts': {tok: dict(cnts) for tok, cnts in self.token_bigram_start_counts.items()},
            'case_after_pattern': {p: dict(cnts) for p, cnts in self.case_after_pattern.items()},
            'char_pair_counts': {k: dict(cnts) for k, cnts in self.char_pair_counts.items()},
            'script_transition_counts': {s: dict(cnts) for s, cnts in self.script_transition_counts.items()},
            'script_char_counts': {s: dict(cnts) for s, cnts in self.script_char_counts.items()},
            'script_global_counts': dict(self.script_global_counts),
            'alphabet': sorted(self.alphabet),
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
            max_order=payload.get('max_order', 15),
            seq_weight=payload.get('seq_weight', 1.0),
            labeled_weight=payload.get('labeled_weight', 5.0),
            max_chars_per_line=payload.get('max_chars_per_line', 600),
        )
        model.global_counts.update(payload.get('global_counts', {}))

        for key, suffix_map in payload.get('order_counts', {}).items():
            k = int(key)
            if k not in model.order_counts:
                model.order_counts[k] = defaultdict(Counter)
            for suffix, cnts in suffix_map.items():
                model.order_counts[k][suffix].update(cnts)
        for key, counts in payload.get('order_unique_follows', {}).items():
            k = int(key)
            model.order_unique_follows[k].update(counts)
        for ctx, cnts in payload.get('exact_context_counts', {}).items():
            model.exact_context_counts[ctx].update(cnts)
        for prefix, cnts in payload.get('word_prefix_counts', {}).items():
            model.word_prefix_counts[prefix].update(cnts)
        model.word_full_counts.update(payload.get('word_full_counts', {}))
        model.word_start_counts.update(payload.get('word_start_counts', {}))
        model.after_word_boundary_counts.update(payload.get('after_word_boundary_counts', {}))
        for tok, cnts in payload.get('token_bigram_start_counts', {}).items():
            model.token_bigram_start_counts[tok].update(cnts)
        for p, cnts in payload.get('case_after_pattern', {}).items():
            model.case_after_pattern[p].update(cnts)
        for k, cnts in payload.get('char_pair_counts', {}).items():
            model.char_pair_counts[k].update(cnts)
        for s, cnts in payload.get('script_transition_counts', {}).items():
            model.script_transition_counts[s].update(cnts)
        for s, cnts in payload.get('script_char_counts', {}).items():
            model.script_char_counts[s].update(cnts)
        model.script_global_counts.update(payload.get('script_global_counts', {}))
        model.alphabet.update(payload.get('alphabet', []))
        model._finalize()
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
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
            f"Using {len(train_data.get('unlabeled', []))} unlabeled contexts "
            f"and {len(train_data.get('labeled', []))} labeled completion pairs"
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