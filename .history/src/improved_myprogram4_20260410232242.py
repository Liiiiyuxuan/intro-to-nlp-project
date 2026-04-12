#!/usr/bin/env python
import csv
import json
import os
import time
import unicodedata
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path


class MyModel:
    """
    High-performance next-character prediction model v6.

    Built on the proven v4 architecture with targeted improvements:
    1. Higher order n-grams (20 vs 10) for better long-context matching
    2. Case-aware word prefix model with case-insensitive fallback
    3. Case prediction heuristics (sentence start → uppercase, etc.)
    4. Character pair/trigram model for local patterns
    5. Context-adaptive ensemble weights
    6. Common English suffix pattern boosting
    7. Better labeled data utilization (full sequence + multi-weight)
    8. Larger candidate pool (64) for better coverage
    """

    MODEL_FILE = 'model.checkpoint'
    MODEL_VERSION = 62

    BOS = '\u0002'
    DISALLOWED_OUTPUT_CHARS = {' ', BOS}

    DEFAULT_CHARS = list(
        "etaoinshrdlucmfwypvbgkqjxz"
        "ETAOINSHRDLUCMFWYPVBGKQJXZ"
        "0123456789"
        ".,!?;:'\"-_/()[]{}@#$%&*+=<>\\|`~"
    )

    def __init__(
        self,
        max_order=20,
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

        self.global_counts = Counter()
        self.global_probs = {}
        self.order_counts = {k: defaultdict(Counter) for k in range(1, self.max_order + 1)}
        self.exact_context_counts = defaultdict(Counter)

        # Word-level models (case-sensitive keys)
        self.word_prefix_counts = defaultdict(Counter)
        # Case-insensitive fallback
        self.word_prefix_lower_counts = defaultdict(Counter)
        self.word_full_counts = Counter()
        self.word_start_counts = Counter()
        self.after_word_boundary_counts = Counter()
        self.token_bigram_start_counts = defaultdict(Counter)

        # Character pair model: ch1 -> {ch2: count} and bigram -> {ch3: count}
        self.char_pair_counts = defaultdict(Counter)
        self.char_trigram_counts = defaultdict(Counter)

        # Script-aware models
        self.script_transition_counts = defaultdict(Counter)
        self.script_char_counts = defaultdict(Counter)
        self.script_global_counts = Counter()

        self.alphabet = set(self.DEFAULT_CHARS)

        # Ensemble weights (proven base from v4, tuned)
        self.weights = {
            'ngram': 1.00,
            'exact': 2.00,
            'word': 0.90,
            'char_local': 0.35,
            'boundary': 0.10,
            'script': 0.40,
            'global': 0.10,
            'same_last_penalty': 0.80,
            'double_penalty': 0.50,
        }

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

    # ── Data I/O (unchanged from v4) ──

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

    # ── Case heuristics ──

    @staticmethod
    def _case_boost(context, ch):
        """Multiplier to boost/penalize case based on context patterns."""
        if not ch.isalpha():
            return 1.0
        if not context:
            # Start of text: slightly favor uppercase
            return 1.4 if ch.isupper() else 0.85

        # Find the end of meaningful context
        stripped = context.rstrip()
        if not stripped:
            return 1.3 if ch.isupper() else 0.85

        last_nonspace = stripped[-1]

        # After sentence-ending punctuation followed by space(s)
        if context != stripped and last_nonspace in '.!?':
            return 1.6 if ch.isupper() else 0.7

        # After colon + space
        if context != stripped and last_nonspace == ':':
            return 1.3 if ch.isupper() else 0.85

        # Mid-word case continuity
        trailing_alpha = ''
        for c in reversed(context):
            if c.isalpha():
                trailing_alpha = c + trailing_alpha
            else:
                break

        if trailing_alpha:
            if len(trailing_alpha) >= 2 and trailing_alpha.isupper():
                # ALL CAPS word: strongly favor uppercase
                return 1.8 if ch.isupper() else 0.4
            if trailing_alpha[0].isupper() and len(trailing_alpha) >= 2 and trailing_alpha[1:].islower():
                # TitleCase: favor lowercase continuation
                return 1.15 if ch.islower() else 0.85
            if trailing_alpha.islower():
                # lowercase word: favor lowercase
                return 1.1 if ch.islower() else 0.85

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
            self.order_counts[k][suffix][nxt] += weight

    def _update_char_local(self, context, nxt, weight=1.0):
        """Update character pair and trigram models."""
        if not nxt or not context:
            return
        last = context[-1]
        self.char_pair_counts[last][nxt] += weight
        if len(context) >= 2:
            bigram = context[-2:]
            self.char_trigram_counts[bigram][nxt] += weight

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

    def _train_on_sequence(self, text, weight=1.0):
        if not text:
            return
        truncated = text[:self.max_chars_per_line]
        padded = self.BOS * self.max_order + truncated
        for i, nxt in enumerate(truncated):
            context_end = self.max_order + i
            context = padded[context_end - self.max_order:context_end]
            self._update_counts(context, nxt, weight=weight)
            raw_ctx = context.replace(self.BOS, '')
            self._update_script_models(raw_ctx, nxt, weight=weight)
            self._update_char_local(raw_ctx, nxt, weight=weight)

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
                        self.token_bigram_start_counts[prev_token][ch] += weight
            else:
                if token:
                    word = ''.join(token)
                    for i, next_ch in enumerate(word):
                        prefix = word[:i]
                        # Case-sensitive (primary)
                        self.word_prefix_counts[prefix][next_ch] += weight
                        # Case-insensitive (fallback)
                        self.word_prefix_lower_counts[prefix.lower()][next_ch] += weight
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
                self.word_prefix_lower_counts[prefix.lower()][next_ch] += weight
            self.word_full_counts[word] += weight

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

        # Dense self-supervised pass on unlabeled contexts
        if unlabeled:
            self._log('Starting dense unlabeled LM pass')
            step = self._progress_step(len(unlabeled))
            start = time.monotonic()
            for i, text in enumerate(unlabeled, 1):
                self._train_on_sequence(text, weight=self.seq_weight)
                self._update_word_models(text, weight=self.seq_weight)
                self._maybe_log_progress('unlabeled LM pass', i, len(unlabeled), step, start)

        # Labeled completion pass with stronger weight
        if labeled:
            self._log('Starting labeled completion pass')
            step = self._progress_step(len(labeled))
            start = time.monotonic()
            for i, (context, nxt) in enumerate(labeled, 1):
                # N-gram update with padded context
                padded_context = (self.BOS * self.max_order + context)[-self.max_order:]
                self._update_counts(padded_context, nxt, weight=self.labeled_weight)
                self._update_script_models(context, nxt, weight=self.labeled_weight)
                self._update_char_local(context, nxt, weight=self.labeled_weight)

                # Exact context memorization (very high weight)
                self.exact_context_counts[context][nxt] += self.labeled_weight

                # Also train on the full context+answer as a sequence
                # This gives the n-gram model more data about natural text
                self._update_word_models(context + nxt, weight=1.5)

                # Train the full context as a sequence too (lower weight to not overwhelm)
                if len(context) > 1:
                    self._train_on_sequence(context + nxt, weight=0.3)

                self._maybe_log_progress('labeled completion pass', i, len(labeled), step, start)

        self._finalize()

    # ── Prediction distributions ──

    def _ngram_distribution(self, context):
        """Interpolated n-gram with Witten-Bell smoothing (proven from v4)."""
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
            distinct = len(counts)
            backoff_mass = distinct / (total + distinct)
            mle_mass = 1.0 - backoff_mass
            inv_total = 1.0 / total
            new_dist = {}
            for ch, prev_p in dist.items():
                mle = counts.get(ch, 0.0) * inv_total
                new_dist[ch] = backoff_mass * prev_p + mle_mass * mle
            # Include chars in counts not in dist yet
            for ch in counts:
                if ch not in new_dist:
                    mle = counts[ch] * inv_total
                    new_dist[ch] = mle_mass * mle
            dist = new_dist
        return dist

    def _word_model_distribution(self, context):
        """Word prefix model with case-sensitive primary and case-insensitive fallback."""
        scores = Counter()
        fragment = self._trailing_fragment(context)
        if fragment:
            # Try exact case first
            counts = self.word_prefix_counts.get(fragment)
            if counts and sum(counts.values()) > 0:
                total = sum(counts.values())
                inv = 1.0 / total
                for ch, cnt in counts.items():
                    scores[ch] += cnt * inv
            else:
                # Fallback: case-insensitive
                lower_frag = fragment.lower()
                counts = self.word_prefix_lower_counts.get(lower_frag)
                if counts:
                    total = sum(counts.values())
                    if total > 0:
                        inv = 1.0 / total
                        for ch, cnt in counts.items():
                            scores[ch] += cnt * inv
        else:
            # At word boundary: predict word start
            total = sum(self.word_start_counts.values())
            if total > 0:
                inv = 1.0 / total
                for ch, cnt in self.word_start_counts.items():
                    scores[ch] += 0.35 * cnt * inv
            prev_word = self._previous_completed_word(context)
            if prev_word:
                counts = self.token_bigram_start_counts.get(prev_word)
                if counts:
                    total = sum(counts.values())
                    if total > 0:
                        inv = 1.0 / total
                        for ch, cnt in counts.items():
                            scores[ch] += 0.65 * cnt * inv
        return scores

    def _char_local_distribution(self, context):
        """Distribution from character pair and trigram models."""
        scores = Counter()
        if not context:
            return scores

        # Character pair (last char → next)
        last = context[-1]
        counts1 = self.char_pair_counts.get(last)
        if counts1:
            total = sum(counts1.values())
            if total > 0:
                inv = 1.0 / total
                for ch, cnt in counts1.items():
                    scores[ch] += 0.35 * cnt * inv

        # Character trigram (last 2 chars → next)
        if len(context) >= 2:
            bigram = context[-2:]
            counts2 = self.char_trigram_counts.get(bigram)
            if counts2:
                total = sum(counts2.values())
                if total > 0:
                    inv = 1.0 / total
                    for ch, cnt in counts2.items():
                        scores[ch] += 0.65 * cnt * inv

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

    def _get_adaptive_weights(self, context):
        """Return adjusted weights based on context analysis."""
        w = dict(self.weights)  # copy base weights
        fragment = self._trailing_fragment(context)

        if fragment:
            frag_len = len(fragment)
            # Mid-word: boost word model, especially with short fragments
            if frag_len <= 3:
                w['word'] = 1.40
            elif frag_len <= 6:
                w['word'] = 1.10
            else:
                w['word'] = 0.70  # very long fragments: n-gram knows more

            # Check if we have good word prefix data
            counts = self.word_prefix_counts.get(fragment)
            if counts and sum(counts.values()) > 5:
                w['word'] *= 1.2  # boost when confident
        else:
            # At word boundary
            w['boundary'] = 0.20
            w['word'] = 0.85
            stripped = context.rstrip() if context else ''
            if stripped and stripped[-1] in '.!?':
                # After sentence end: n-gram and case are important
                w['ngram'] = 1.15

        # Exact context match: boost heavily when available
        cache = self.exact_context_counts.get(context)
        if cache:
            total = sum(cache.values())
            if total >= 10:
                w['exact'] = 3.50
            elif total >= 3:
                w['exact'] = 2.50

        return w

    # ── Candidate scoring and prediction ──

    def _candidate_pool(self, context, ngram, exact, word, char_local, boundary, script, limit=64):
        merged = Counter()
        for source, wt in ((ngram, 1.0), (exact, 1.3), (word, 0.9), (char_local, 0.3), (boundary, 0.15), (script, 0.35)):
            for ch, score in source.items():
                if ch in self.DISALLOWED_OUTPUT_CHARS:
                    continue
                merged[ch] += wt * score
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

    def _score_candidate(self, context, ch, ngram, exact, word, char_local, boundary, script, w):
        score = 0.0
        score += w['ngram'] * ngram.get(ch, 0.0)
        score += w['exact'] * exact.get(ch, 0.0)
        score += w['word'] * word.get(ch, 0.0)
        score += w['char_local'] * char_local.get(ch, 0.0)
        score += w['boundary'] * boundary.get(ch, 0.0)
        score += w['script'] * script.get(ch, 0.0)
        score += w['global'] * self.global_probs.get(ch, 0.0)

        # Case boost
        case_mult = self._case_boost(context, ch)
        score *= case_mult

        # Repeat penalties
        if context:
            last = context[-1]
            if ch == last:
                score *= w['same_last_penalty']
            if len(context) >= 2 and context[-1] == context[-2] == ch:
                score *= w['double_penalty']
            # Script continuity bonus
            last_script = self._char_script(last)
            if last_script not in ('COMMON', 'DIGIT') and self._char_script(ch) == last_script:
                score += 0.02
        return score

    def _top_guesses(self, context, n=3):
        ngram = self._ngram_distribution(context)
        word = self._word_model_distribution(context)
        char_local = self._char_local_distribution(context)
        script = self._script_distribution(context)

        boundary = {}
        fragment = self._trailing_fragment(context)
        if fragment and self.word_full_counts.get(fragment, 0.0) > 0.0:
            boundary = self._boundary_distribution()

        exact = {}
        cache = self.exact_context_counts.get(context)
        if cache:
            total = sum(cache.values())
            if total > 0:
                exact = {ch: cnt / total for ch, cnt in cache.items()}

        w = self._get_adaptive_weights(context)
        candidates = self._candidate_pool(context, ngram, exact, word, char_local, boundary, script, limit=64)

        scored = []
        for ch in candidates:
            scored.append((self._score_candidate(context, ch, ngram, exact, word, char_local, boundary, script, w), ch))
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
            'weights': self.weights,
            'global_counts': dict(self.global_counts),
            'order_counts': {
                str(k): {suffix: dict(cnts) for suffix, cnts in suffix_map.items()}
                for k, suffix_map in self.order_counts.items()
            },
            'exact_context_counts': {ctx: dict(cnts) for ctx, cnts in self.exact_context_counts.items()},
            'word_prefix_counts': {prefix: dict(cnts) for prefix, cnts in self.word_prefix_counts.items()},
            'word_prefix_lower_counts': {prefix: dict(cnts) for prefix, cnts in self.word_prefix_lower_counts.items()},
            'word_full_counts': dict(self.word_full_counts),
            'word_start_counts': dict(self.word_start_counts),
            'after_word_boundary_counts': dict(self.after_word_boundary_counts),
            'token_bigram_start_counts': {tok: dict(cnts) for tok, cnts in self.token_bigram_start_counts.items()},
            'char_pair_counts': {k: dict(cnts) for k, cnts in self.char_pair_counts.items()},
            'char_trigram_counts': {k: dict(cnts) for k, cnts in self.char_trigram_counts.items()},
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
            max_order=payload.get('max_order', 20),
            seq_weight=payload.get('seq_weight', 1.0),
            labeled_weight=payload.get('labeled_weight', 5.0),
            max_chars_per_line=payload.get('max_chars_per_line', 600),
        )
        model.weights.update(payload.get('weights', {}))
        model.global_counts.update(payload.get('global_counts', {}))

        for key, suffix_map in payload.get('order_counts', {}).items():
            k = int(key)
            if k not in model.order_counts:
                model.order_counts[k] = defaultdict(Counter)
            for suffix, cnts in suffix_map.items():
                model.order_counts[k][suffix].update(cnts)
        for ctx, cnts in payload.get('exact_context_counts', {}).items():
            model.exact_context_counts[ctx].update(cnts)
        for prefix, cnts in payload.get('word_prefix_counts', {}).items():
            model.word_prefix_counts[prefix].update(cnts)
        for prefix, cnts in payload.get('word_prefix_lower_counts', {}).items():
            model.word_prefix_lower_counts[prefix].update(cnts)
        model.word_full_counts.update(payload.get('word_full_counts', {}))
        model.word_start_counts.update(payload.get('word_start_counts', {}))
        model.after_word_boundary_counts.update(payload.get('after_word_boundary_counts', {}))
        for tok, cnts in payload.get('token_bigram_start_counts', {}).items():
            model.token_bigram_start_counts[tok].update(cnts)
        for k, cnts in payload.get('char_pair_counts', {}).items():
            model.char_pair_counts[k].update(cnts)
        for k, cnts in payload.get('char_trigram_counts', {}).items():
            model.char_trigram_counts[k].update(cnts)
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