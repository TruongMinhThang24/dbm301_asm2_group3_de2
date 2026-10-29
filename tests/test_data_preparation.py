import tempfile
import unittest
from pathlib import Path

import pandas as pd

from groceries_group3.preparation.taxonomy import build_item_mapping
from groceries_group3.preparation.transactions import (
    DataContractError,
    _target_prevalence,
    build_transaction_matrix,
    load_assignment_data,
    make_group_presence_matrix,
    parse_items,
    simulate_test_observation,
)


class ParseItemsTests(unittest.TestCase):
    def test_splits_semicolon_and_trims_whitespace(self) -> None:
        self.assertEqual(parse_items(" milk ; bread "), ["milk", "bread"])

    def test_accepts_single_item_basket(self) -> None:
        self.assertEqual(parse_items("milk"), ["milk"])

    def test_rejects_empty_or_malformed_items(self) -> None:
        for value in ("", "milk;;bread", ";milk", "milk;", None):
            with self.subTest(value=value), self.assertRaises(DataContractError):
                parse_items(value)  # type: ignore[arg-type]


class DataContractTests(unittest.TestCase):
    def _write_inputs(self, root: Path, train: pd.DataFrame, test: pd.DataFrame) -> None:
        root.mkdir(parents=True, exist_ok=True)
        train.to_csv(root / "groceries_train.csv", index=False)
        test.to_csv(root / "groceries_test.csv", index=False)

    def test_loader_rejects_duplicate_ids_and_train_test_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = pd.DataFrame(
                {"basket_id": [1, 1], "items": ["milk", "bread"]}
            )
            test = pd.DataFrame({"basket_id": [3], "items": ["bread"]})
            self._write_inputs(root, train, test)
            with self.assertRaisesRegex(DataContractError, "duplicate basket_id"):
                load_assignment_data(root, expected_train_rows=2, expected_test_rows=1)

            train = pd.DataFrame({"basket_id": [1, 2], "items": ["milk", "bread"]})
            test = pd.DataFrame({"basket_id": [2], "items": ["bread"]})
            self._write_inputs(root, train, test)
            with self.assertRaisesRegex(DataContractError, "overlap"):
                load_assignment_data(root, expected_train_rows=2, expected_test_rows=1)

    def test_loader_rejects_nulls_and_invalid_delimiters(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = pd.DataFrame({"basket_id": [1], "items": ["milk;;bread"]})
            test = pd.DataFrame({"basket_id": [2], "items": ["bread"]})
            self._write_inputs(root, train, test)
            with self.assertRaisesRegex(DataContractError, "empty token"):
                load_assignment_data(root, expected_train_rows=1, expected_test_rows=1)

            train = pd.DataFrame({"basket_id": [1], "items": [None]})
            self._write_inputs(root, train, test)
            with self.assertRaises(DataContractError):
                load_assignment_data(root, expected_train_rows=1, expected_test_rows=1)

    def test_matrix_is_boolean_and_row_sums_match_basket_sizes(self) -> None:
        frame = pd.DataFrame(
            {
                "basket_id": [10, 11],
                "items_list": [["milk", "bread"], ["bread"]],
                "basket_size": [2, 1],
            }
        )
        matrix = build_transaction_matrix(frame, ["bread", "milk"])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(all(dtype == bool for dtype in matrix.dtypes))
        self.assertEqual(matrix.index.tolist(), [10, 11])
        self.assertEqual(matrix.sum(axis=1).tolist(), [2, 1])
        self.assertNotIn("basket_id", matrix.columns)

    def test_matrix_rejects_duplicate_items_and_unmapped_columns(self) -> None:
        repeated = pd.DataFrame(
            {"basket_id": [1], "items_list": [["milk", "milk"]], "basket_size": [2]}
        )
        with self.assertRaisesRegex(DataContractError, "repeats an item"):
            build_transaction_matrix(repeated)

        missing_column = pd.DataFrame(
            {"basket_id": [1], "items_list": [["milk"]], "basket_size": [1]}
        )
        with self.assertRaisesRegex(DataContractError, "missing from the matrix vocabulary"):
            build_transaction_matrix(missing_column, ["bread"])

    def test_simulation_filters_original_basket_size_before_target_removal(self) -> None:
        frame = pd.DataFrame(
            {
                "basket_id": [1, 2, 3],
                "items_list": [
                    ["other vegetables"],
                    ["other vegetables", "milk"],
                    ["bread", "milk"],
                ],
                "basket_size": [1, 2, 2],
            }
        )
        ids, observed, labels = simulate_test_observation(frame)
        self.assertEqual(ids.tolist(), [2, 3])
        self.assertEqual(observed, [["milk"], ["bread", "milk"]])
        self.assertEqual(labels.tolist(), [1, 0])

    def test_target_prevalence_uses_original_basket_size(self) -> None:
        frame = pd.DataFrame(
            {
                "items_list": [
                    ["other vegetables"],
                    ["other vegetables", "milk"],
                    ["bread", "milk"],
                    ["other vegetables", "bread", "milk"],
                ],
                "basket_size": [1, 2, 2, 3],
            }
        )
        prevalence = _target_prevalence(frame)
        self.assertAlmostEqual(prevalence.loc[0, "target_prevalence"], 0.75)
        self.assertAlmostEqual(prevalence.loc[1, "target_prevalence"], 2 / 3)

    def test_taxonomy_requires_exact_one_to_one_vocabulary_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "vocabulary mismatch"):
            build_item_mapping({"unknown item"})

    def test_group_features_can_exclude_target_item(self) -> None:
        mapping = pd.DataFrame(
            {
                "item": ["other vegetables", "milk"],
                "group": ["produce", "dairy"],
            }
        )
        features = make_group_presence_matrix(
            [["other vegetables", "milk"]], mapping, exclude_items={"other vegetables"}
        )
        self.assertEqual(features.columns.tolist(), ["dairy", "produce"])
        self.assertEqual(features.iloc[0].tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
