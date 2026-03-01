#!/usr/bin/env python
import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path


class MyModel:
    """
    Character-level n-gram model with simple backoff for next-character prediction.
    """

    MODEL_FILE = 'model.checkpoint'
    MODEL_VERSION = 2

    def __init__(self, max_order=8):
        self.max_order = max_order
        self.global_counts = Counter()
        self.order_counts = {k: defaultdict(Counter) for k in range(1, self.max_order + 1)}

    @classmethod
    def _repo_root(cls):
        return Path(__file__).resolve().parent.parent

    @classmethod
    def load_training_data(cls):
        """
        Discover training pairs by finding sibling input/answer files under ./data.
        """
        data = []
        data_dir = cls._repo_root() / 'data'
        if not data_dir.is_dir():
            return data

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
                        data.append((context, nxt[0]))
        return data

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname, 'rt', encoding='utf-8') as f:
            for line in f:
                data.append(line.rstrip('\n'))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write(f'{p}\n')

    def run_train(self, data, work_dir):
        for context, nxt in data:
            if not nxt:
                continue
            self.global_counts[nxt] += 1
            max_k = min(self.max_order, len(context))
            for k in range(1, max_k + 1):
                suffix = context[-k:]
                self.order_counts[k][suffix][nxt] += 1

        # Fallback distribution if no train data is available.
        if not self.global_counts:
            for ch in ' etaoinshrdlucmfwypvbgkqjxz,.!?':
                self.global_counts[ch] += 1

    def _top_guesses(self, context, n=3):
        ranked = []

        max_k = min(self.max_order, len(context))
        for k in range(max_k, 0, -1):
            suffix = context[-k:]
            counts = self.order_counts[k].get(suffix)
            if not counts:
                continue
            for ch, _ in counts.most_common():
                if ch not in ranked:
                    ranked.append(ch)
                if len(ranked) >= n:
                    return ''.join(ranked[:n])

        for ch, _ in self.global_counts.most_common():
            if ch not in ranked:
                ranked.append(ch)
            if len(ranked) >= n:
                break

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
            # Legacy or corrupted checkpoint: rebuild using current code.
            return cls._train_and_save_default(work_dir)

        if payload.get('model_version') != cls.MODEL_VERSION:
            # Force refresh when model logic changes so predictions reflect code updates.
            return cls._train_and_save_default(work_dir)

        model = cls(max_order=payload.get('max_order', 8))
        model.global_counts.update(payload.get('global_counts', {}))

        raw_order_counts = payload.get('order_counts', {})
        for key, suffix_map in raw_order_counts.items():
            k = int(key)
            for suffix, cnts in suffix_map.items():
                model.order_counts[k][suffix].update(cnts)

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
        print(f'Training on {len(train_data)} samples')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print(f'Loading test data from {args.test_data}')
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print(f'Writing predictions to {args.test_output}')
        assert len(pred) == len(test_data), f'Expected {len(test_data)} predictions but got {len(pred)}'
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError(f'Unknown mode {args.mode}')
