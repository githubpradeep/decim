import unittest

from decision_model.datasets.public import choice_row, noul_row, score_row, steps_for, stratified_cap
from decision_model.schema import parse_request


def _labeled(kind: str, label) -> dict:
    if kind == "noul":
        return noul_row("text", "Is it positive?", label)
    if kind == "score":
        return score_row("text", "How positive?", ["low", "high"], label)
    return choice_row("text", "Which?", [("a", "one"), ("b", "two")], label)


class PublicDataTest(unittest.TestCase):
    def test_rows_match_the_schema(self) -> None:
        parse_request(
            choice_row("How far?", "What type?", [("NUM", "numeric values"), ("LOC", "locations")], "NUM"),
            labeled=True,
        )
        parse_request(noul_row("a charming film", "Is it positive?", True), labeled=True)
        parsed = parse_request(
            score_row("awful", "Stars?", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], 0),
            labeled=True,
        )
        self.assertEqual(parsed.questions["answer"].gold, 0)
        self.assertEqual(parsed.questions["answer"].levels[0], "1 star")

    def test_stratified_cap_keeps_every_label(self) -> None:
        rows = [_labeled("choice", "a")] * 50 + [_labeled("choice", "b")] * 10
        capped = stratified_cap(rows, 12, seed=0)
        labels = {row["questions"]["answer"]["label"] for row in capped}
        self.assertEqual(labels, {"a", "b"})
        self.assertEqual(len(capped), 12)
        self.assertEqual(stratified_cap(rows, 12, seed=0), capped)

    def test_step_count(self) -> None:
        self.assertEqual(steps_for(10, 8, 2), 4)
