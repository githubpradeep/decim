import json
import tempfile
import unittest
from pathlib import Path

from decision_model.data import load_jsonl
from decision_model.datasets.support import DEMO, EVAL, TRAIN, questions, validate, write_support
from decision_model.schema import parse_request


class SupportDataTest(unittest.TestCase):
    def test_splits_are_labeled_and_disjoint(self) -> None:
        validate()
        for case_row in TRAIN + EVAL:
            request = parse_request(
                {
                    "state": {
                        "channel": case_row[3],
                        "customer": case_row[4],
                        "subject": case_row[5],
                        "body": case_row[6],
                    },
                    "questions": questions(case_row, labeled=True),
                },
                labeled=True,
            )
            self.assertEqual(list(request.questions), ["queue", "escalate", "priority"])
            self.assertEqual(
                request.questions["queue"].keys,
                ["billing", "shipping", "access", "product"],
            )
        bodies = [case[-1] for case in TRAIN + EVAL + DEMO]
        self.assertEqual(len(bodies), len(set(bodies)))

    def test_distractors_do_not_set_the_queue(self) -> None:
        shipping = [case for case in TRAIN if case[0] == "shipping"]
        self.assertTrue(any("refund" in case[-1].lower() for case in shipping))
        access = [case for case in TRAIN if case[0] == "access"]
        self.assertTrue(any("invoice" in case[-1].lower() for case in access))
        product = [case for case in TRAIN if case[0] == "product"]
        self.assertTrue(any("charge" in case[-1].lower() for case in product))
        billing = [case for case in TRAIN if case[0] == "billing"]
        self.assertTrue(any("tracking" in case[-1].lower() for case in billing))

    def test_written_files_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_support(Path(tmp))
            train_rows = load_jsonl(root / "train.jsonl")
            eval_rows = load_jsonl(root / "eval.jsonl")
            self.assertEqual(len(train_rows), len(TRAIN))
            self.assertEqual(len(eval_rows), len(EVAL))
            self.assertEqual(len(list((root / "demo").glob("*.json"))) - 1, len(DEMO))
            for row in train_rows + eval_rows:
                parse_request(row, labeled=True)
            manifest = json.loads((root / "demo" / "expected.json").read_text())
            self.assertEqual(len(manifest), len(DEMO))
            sample = json.loads((root / "demo" / manifest[0]["file"]).read_text())
            for question in sample["questions"].values():
                self.assertNotIn("label", question)
            parse_request(sample)
