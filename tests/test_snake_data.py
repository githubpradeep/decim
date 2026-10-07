import unittest

from decision_model.snake_data import ORDER, describe, snake_rows


class SnakeDataTest(unittest.TestCase):
    def test_gold_move_is_safe_and_rows_match_the_demo_wording(self) -> None:
        rows = snake_rows(30, seed=2)
        self.assertEqual(len(rows), 30)
        labels = set()
        for row in rows:
            move = row["questions"]["move"]
            self.assertEqual(list(move["criteria"]), list(ORDER))
            self.assertIn("Moves that reduce the distance to the food:", row["state"])
            self.assertIn("A safe route is available:", row["state"])
            labels.add(move["label"])
            self.assertNotIn(f"The cell {move['label'].lower()} of the head is a wall.", row["state"])
            self.assertNotIn(f"The cell {move['label'].lower()} of the head is the snake's body.", row["state"])
        self.assertGreater(len(labels), 1)

    def test_describe_names_the_food_step(self) -> None:
        body = [(5, 5), (4, 5), (3, 5)]
        food = (6, 5)
        state, gold, safe, reachable = describe(body, food)
        self.assertEqual(gold, "RIGHT")
        self.assertTrue(safe)
        self.assertTrue(reachable)
        self.assertIn("The cell right of the head is the food.", state)


if __name__ == "__main__":
    unittest.main()
