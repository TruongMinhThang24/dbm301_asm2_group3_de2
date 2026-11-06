"""Checks for WBS 4.0?4.7; synthetic fixtures are never submission data."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import importlib.util
import json
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, clone

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('classification_workflow', ROOT / 'src/groceries_group3/classification/workflow.py')
clf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clf)


def synthetic_data(n=240):
    rng = np.random.default_rng(123)
    mapping = {f'product_{i}': f'group_{i // 3}' for i in range(15)}
    rows = []
    for i in range(n):
        basket = [f'product_{j}' for j in rng.choice(15, size=int(rng.integers(2, 7)), replace=False)]
        if rng.random() < (.7 if 'product_0' in basket else .2):
            basket.append(clf.CLF_TARGET)
        rows.append(dict(basket_id=f'train_{i}', items_list=basket))
    return pd.DataFrame(rows), mapping


class HandoffStub(BaseEstimator):
    """Test-only handoff protocol; deliberately rejects labels in fit."""
    def fit(self, X, y=None):
        if y is not None:
            raise AssertionError('Labels were passed to unsupervised handoff.')
        self.n_fit_rows_ = len(X)
        self.items_ = sorted({item for basket in X for item in basket})
        return self

    def predict(self, X):
        return np.asarray(['product_0' in b for b in X], dtype=int)


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((ROOT / 'notebooks/notebook.ipynb').read_text(encoding='utf-8'))
        context = {}
        handoff = next(c for c in notebook['cells'] if c['id'] == '6f039145')
        exec(compile(''.join(handoff['source']), 'unchanged_association_handoff', 'exec'), context)
        cls.RuleTransformer = context['RuleFeatureTransformer']

    def test_original_size_eligibility_and_hidden_target(self):
        frame = pd.DataFrame(dict(basket_id=['a', 'b', 'c', 'd'], items_list=[
            [clf.CLF_TARGET], ['bread', clf.CLF_TARGET], ['bread', 'milk'], ['milk']]))
        X, y, ids = clf.clf_prepare_population(frame)
        self.assertEqual(ids.tolist(), ['b', 'c'])
        self.assertEqual(y.tolist(), [1, 0])
        self.assertEqual(X[0], ('bread',))
        self.assertTrue(all(clf.CLF_TARGET not in b for b in X))

    def test_fold_vocabulary_unknown_items_and_group_counts(self):
        f = clf.BasketClassificationFeatures({'a': 'food', 'b': 'food', 'c': 'food'}, 'item_size_group')
        f.fit([('a', 'b'), ('b',)])
        names = f.get_feature_names_out().tolist()
        row = f.transform([('a', 'c', 'new')]).toarray()[0]
        self.assertEqual(row[names.index('size::visible')], 3)
        self.assertEqual(row[names.index('group::food')], 2)
        self.assertNotIn('item::c', names)
        self.assertEqual(f.get_feature_names_out().tolist(), names)
        with self.assertRaisesRegex(ValueError, 'Target leaked'):
            f.transform([(clf.CLF_TARGET,)])

    def test_rule_score_matching_and_prior_are_visible(self):
        frame, _ = synthetic_data(120)
        X, y, _ = clf.clf_prepare_population(frame)
        model = clf.RuleBasedBasketClassifier(self.RuleTransformer(min_support=.05)).fit(X[:90], y[:90])
        details = model.score_details(X[90:])
        expected = []
        for basket in X[90:]:
            matches = [float(r.confidence) for _, r in model.transformer_.rules_.iterrows()
                       if r.antecedents.issubset(set(basket))]
            expected.append(max(matches) if matches else y[:90].mean())
        self.assertTrue(np.allclose(details.rule_score, expected))
        self.assertTrue(np.allclose(model.predict_proba(X[90:])[:, 1], expected))
        self.assertEqual(model.transformer_.n_fit_rows_, 90)
        fallback = clf.RuleBasedBasketClassifier(self.RuleTransformer(min_support=1.0)).fit(X[:90], y[:90])
        self.assertTrue(np.allclose(fallback.score_details(X[90:]).rule_score, y[:90].mean()))
        self.assertTrue(fallback.score_details(X[90:]).n_matched_rules.eq(0).all())

    def test_top10_rounding_and_threshold_objective(self):
        y = np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        p = np.asarray([.9, .9] + [.2] * 9)
        metric = clf.clf_metrics(y, p)
        self.assertEqual(metric['n_top10'], 2)
        self.assertEqual(metric['precision_at_top10'], .5)
        threshold, tradeoff = clf.clf_choose_threshold(y, p)
        self.assertEqual(int(tradeoff.selected.sum()), 1)
        self.assertAlmostEqual(clf.f1_score(y, p >= threshold), tradeoff.f1.max())

    def test_export_schema_order_bounds_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'predictions.csv'
            ids = np.asarray([f'{i:05}' for i in range(1535)])
            prob = np.linspace(0, 1, len(ids))
            clf.clf_write_predictions(ids, prob, p)
            saved = pd.read_csv(p, dtype={'basket_id': str})
            self.assertEqual(saved.columns.tolist(), ['basket_id', 'prob'])
            self.assertEqual(saved.basket_id.tolist(), ids.tolist())
            for invalid in [np.full(1535, np.nan), np.full(1535, 1.1)]:
                with self.assertRaises(ValueError): clf.clf_write_predictions(ids, invalid, p)
            with self.assertRaises(ValueError): clf.clf_write_predictions(ids[:-1], prob[:-1], p)
            with self.assertRaises(ValueError): clf.clf_write_predictions(['same'] * 1535, prob, p)

    def test_cluster_adapter_never_receives_y_or_refits_on_validation(self):
        frame, _ = synthetic_data(60)
        X, y, _ = clf.clf_prepare_population(frame)
        features = clf.ClusterLabelFeatures(HandoffStub()).fit(X[:40], y[:40])
        known = features.estimator_.items_.copy()
        rows = features.transform([('new_validation_only_item',), ('product_0',)])
        self.assertEqual(features.estimator_.n_fit_rows_, 40)
        self.assertEqual(features.estimator_.items_, known)
        self.assertEqual(rows.shape[0], 2)
        self.assertTrue(np.allclose(rows.sum(axis=1), 1))

    def test_separate_ml_rules_and_pending_cluster(self):
        frame, mapping = synthetic_data(150)
        X, y, _ = clf.clf_prepare_population(frame)
        splits = list(clf.StratifiedKFold(3, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
        ml = clf.clf_search_candidates(clf.clf_ml_candidates(mapping, forest_trees=8), X, y, splits)
        rules = clf.clf_search_candidates(clf.clf_rule_candidates(self.RuleTransformer()), X, y, splits)
        self.assertEqual(len(ml['table']), 18)
        self.assertEqual(set(ml['table'].family), {'Logistic', 'MultinomialNB', 'RandomForest'})
        self.assertEqual(len(rules['table']), 2)
        name = rules['table'].iloc[0]['name']
        self.assertEqual(rules['details'][name]['row'].tolist(), list(range(len(y))))
        self.assertTrue(np.allclose(rules['details'][name].rule_score, rules['oof'][name]))
        search = dict(table=clf.clf_rank(pd.concat([ml['table'], rules['table']], ignore_index=True)),
            registry={**ml['registry'], **rules['registry']}, oof={**ml['oof'], **rules['oof']})
        old_count = len(search['table'])
        pending, ready = clf.clf_compare_clusters(search, X, y, splits)
        self.assertFalse(ready)
        self.assertTrue(pending.iloc[1][['auc', 'ap', 'delta_auc', 'delta_ap']].isna().all())
        self.assertEqual(len(search['table']), old_count)
        compared, ready = clf.clf_compare_clusters(search, X, y, splits, HandoffStub())
        self.assertTrue(ready)
        self.assertEqual(len(search['table']), old_count + 1)
        self.assertAlmostEqual(compared.iloc[1].delta_ap, compared.iloc[1].ap - compared.iloc[0].ap)

    def test_notebook_outputs_are_under_their_wbs_sections(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        notebook = json.loads((ROOT / 'notebooks/notebook.ipynb').read_text(encoding='utf-8'))
        added = [c for c in notebook['cells'] if c['id'].startswith('clf-')]
        helper = next(c for c in added if 'classification_helper_sha256' in c['metadata'])
        self.assertEqual(''.join(helper['source']), (ROOT / 'src/groceries_group3/classification/workflow.py').read_text(encoding='utf-8'))
        train, mapping = synthetic_data()
        original = train.copy(deep=True)
        test = pd.DataFrame(dict(basket_id=[f'test_{i}' for i in range(1535)], items_list=[
            [x for x in train.iloc[i % len(train)].items_list if x != clf.CLF_TARGET] for i in range(1535)]))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            context = dict(PROJECT_ROOT=Path(folder), train=train, test=test, ITEM_TO_GROUP=mapping,
                           RuleFeatureTransformer=self.RuleTransformer, display=lambda *args: None)
            for cell in added:
                if cell['cell_type'] == 'code':
                    exec(compile(''.join(cell['source']), cell['id'], 'exec'), context)
            out = Path(folder) / 'outputs/classification'
            for name in clf.CLF_FEATURE_SETS:
                a = sparse.load_npz(out / f'4_1_{name}_X_train.npz')
                b = sparse.load_npz(out / f'4_1_{name}_X_validation.npz')
                self.assertEqual(a.shape[0], len(context['clf_y_train']))
                self.assertEqual(b.shape[0], len(context['clf_y_validation']))
                self.assertEqual(a.shape[1], b.shape[1])
            scores = pd.read_csv(out / '4_3_rule_scores.csv')
            self.assertEqual(len(scores), len(context['clf_y_train']))
            self.assertEqual(len(pd.read_csv(out / '4_3_model_comparison.csv')), 4)
            self.assertEqual(len(pd.read_csv(out / '4_4_validation_metrics.csv')), 5)
            self.assertEqual(int(context['clf_confusion'].to_numpy().sum()), len(context['clf_y_validation']))
            gate = json.loads((out / '4_7_status.json').read_text(encoding='utf-8'))
            self.assertFalse(gate['wbs_4_6_complete'])
            self.assertFalse(gate['finalized_after_cluster_comparison'])
            self.assertFalse(gate['test_used_for_selection'])
            self.assertTrue(context['clf_csv_checks'].iloc[:, -1].all())
            self.assertEqual(len(pd.read_csv(Path(folder) / 'outputs/predictions.csv')), 1535)
            pd.testing.assert_frame_equal(train, original)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
