"""Helpers for the eight classification tasks WBS 4.0–4.7."""
from pathlib import Path
from time import perf_counter
from collections import Counter
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.utils.validation import check_is_fitted

CLF_TARGET = 'other vegetables'
CLF_SEED = 42
# Configuration already handed over by WBS 5.4 on the supplied dataset.
CLF_WBS54_PARAMS = {'n_clusters': 8, 'n_components': 20, 'random_state': 42}
CLF_FEATURE_SETS = ('item', 'item_size', 'item_size_group')
CLF_MODEL_NAMES = {'Logistic': 'Logistic Regression', 'MultinomialNB': 'MultinomialNB (NB)',
                   'RandomForest': 'Random Forest (tree-family)', 'Rules': 'Rule-based',
                   'Baseline': 'Baseline prevalence train'}
CLF_FEATURE_NAMES = {'item': 'item', 'item_size': 'item + size',
                    'item_size_group': 'item + size + group counts', 'rule_matches': 'luật'}



def clf_visible_baskets(baskets):
    """Validate visible inputs; reject a leaked target instead of silently hiding it."""
    result = []
    for basket in baskets:
        if isinstance(basket, str):
            raise ValueError('Expected lists of item names, not CSV strings.')
        items = list(basket)
        if CLF_TARGET in items:
            raise ValueError('Target leaked into classification features.')
        if len(items) != len(set(items)) or not all(isinstance(x, str) and x for x in items):
            raise ValueError('Items must be nonempty unique strings.')
        result.append(tuple(sorted(items)))
    return result


def clf_prepare_population(frame):
    """Eligibility uses ORIGINAL size; y is kept before the target is removed."""
    if frame.basket_id.isna().any() or frame.basket_id.duplicated().any():
        raise ValueError('Training basket IDs must be nonmissing and unique.')
    original = frame.items_list.map(tuple)
    if any(len(b) != len(set(b)) for b in original):
        raise ValueError('Repeated item in original basket.')
    eligible = original.map(len).ge(2)
    selected = original[eligible]
    visible = clf_visible_baskets([set(b) - {CLF_TARGET} for b in selected])
    labels = np.asarray([CLF_TARGET in b for b in selected], dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError('Both target classes are required for validation.')
    ids = frame.loc[eligible, 'basket_id'].to_numpy(copy=True)
    return visible, labels, ids


def clf_take(baskets, indices):
    return [baskets[int(i)] for i in indices]


class BasketClassificationFeatures(TransformerMixin, BaseEstimator):
    """Fit item/group feature vocabulary on the current training fold only."""
    def __init__(self, mapping=None, feature_set='item'):
        self.mapping = mapping
        self.feature_set = feature_set

    def _records(self, X):
        if self.feature_set not in CLF_FEATURE_SETS:
            raise ValueError('Unknown feature set.')
        rows = []
        for basket in clf_visible_baskets(X):
            row = {'item::' + item: 1.0 for item in basket}
            if self.feature_set != 'item':
                row['size::visible'] = float(len(basket))
            if self.feature_set == 'item_size_group':
                mapping = self.mapping if self.mapping is not None else {}
                counts = Counter(mapping.get(item, '__unmapped__') for item in basket)
                row.update({'group::' + g: float(n) for g, n in counts.items()})
            rows.append(row)
        return rows

    def fit(self, X, y=None):
        records = self._records(X)
        self.vectorizer_ = DictVectorizer(sparse=True, sort=True).fit(records)
        self.n_fit_rows_ = len(records)
        return self

    def transform(self, X):
        check_is_fitted(self, 'vectorizer_')
        matrix = sparse.csr_matrix(self.vectorizer_.transform(self._records(X)))
        # liblinear/tree estimators require 32-bit sparse indices at this scale.
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
        return matrix

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'vectorizer_')
        return self.vectorizer_.get_feature_names_out()


class RuleBasedBasketClassifier(ClassifierMixin, BaseEstimator):
    """Reuse Hưng's transformer without editing it; max matched confidence or prior."""
    def __init__(self, rule_transformer=None):
        self.rule_transformer = rule_transformer

    def fit(self, X, y):
        clf_visible_baskets(X)
        if self.rule_transformer is None:
            raise ValueError('Pass the RuleFeatureTransformer handed over in WBS 3.8.')
        self.transformer_ = clone(self.rule_transformer).fit(X, y)
        self.prior_ = float(np.mean(y))
        self.classes_ = np.asarray([0, 1])
        return self

    def score_details(self, X):
        check_is_fitted(self, 'transformer_')
        clf_visible_baskets(X)
        rules = self.transformer_.rules_
        if rules.empty:
            scores = np.full(len(X), self.prior_)
            counts = np.zeros(len(X), dtype=int)
        else:
            matched = self.transformer_.transform(X).astype(bool)
            counts = matched.sum(axis=1)
            scores = np.where(matched, rules.confidence.to_numpy()[None, :], -1.0).max(axis=1)
            scores = np.where(counts == 0, self.prior_, scores)
        return pd.DataFrame({'rule_score': np.clip(scores, 0, 1), 'n_matched_rules': counts,
            'score_source': np.where(counts > 0, 'max confidence', 'train-fold prevalence')})

    def predict_proba(self, X):
        scores = self.score_details(X).rule_score.to_numpy()
        return np.column_stack([1 - scores, scores])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


def clf_metrics(y, prob):
    y, prob = np.asarray(y), np.asarray(prob, dtype=float)
    if len(y) != len(prob) or not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError('Probabilities must align and be finite in [0, 1].')
    n_top = max(1, int(np.ceil(.10 * len(y))))
    top = np.argsort(-prob, kind='stable')[:n_top]
    return dict(auc=roc_auc_score(y, prob), ap=average_precision_score(y, prob),
                precision_at_top10=float(y[top].mean()), n_top10=n_top,
                brier=brier_score_loss(y, prob), log_loss=log_loss(y, prob, labels=[0, 1]),
                prevalence=float(y.mean()))


def clf_ml_candidates(mapping, forest_trees=120):
    candidates = []
    for feature_set in CLF_FEATURE_SETS:
        for family, settings in [
            ('Logistic', [dict(C=.1), dict(C=1.0)]),
            ('MultinomialNB', [dict(alpha=.5), dict(alpha=2.0)]),
            ('RandomForest', [dict(min_samples_leaf=2, max_depth=None),
                              dict(min_samples_leaf=5, max_depth=12)])]:
            for params in settings:
                if family == 'Logistic':
                    model = LogisticRegression(max_iter=1500, solver='liblinear', random_state=CLF_SEED, **params)
                elif family == 'MultinomialNB':
                    model = MultinomialNB(**params)
                else:
                    # Serial tree accumulation also makes exported probability bytes reproducible.
                    model = RandomForestClassifier(n_estimators=forest_trees, n_jobs=1,
                                                   random_state=CLF_SEED, **params)
                estimator = Pipeline([('features', BasketClassificationFeatures(mapping, feature_set)),
                                      ('model', model)])
                candidates.append(dict(name=f'{family}__{feature_set}__{len(candidates):02d}',
                                       family=family, feature_set=feature_set,
                                       params=json.dumps(model.get_params(), sort_keys=True), estimator=estimator))
    return candidates



def clf_item_effects(logistic_pipeline, X_dev, y_dev):
    """Conditional coefficient and univariate train-only lift are different views."""
    features = logistic_pipeline.named_steps['features']
    matrix = features.transform(X_dev)
    coefficients = logistic_pipeline.named_steps['model'].coef_[0]
    rows = []
    for j, name in enumerate(features.get_feature_names_out()):
        if not name.startswith('item::'):
            continue
        mask = matrix[:, j].toarray().ravel() > 0
        confidence = float(y_dev[mask].mean())
        support = float(np.mean(mask & (y_dev == 1)))
        rows.append(dict(item=name.removeprefix('item::'), coefficient=float(coefficients[j]),
                         odds_ratio=float(np.exp(np.clip(coefficients[j], -700, 700))),
                         n_item=int(mask.sum()), support=support, confidence=confidence,
                         lift=confidence / float(y_dev.mean()),
                         direction_agrees=bool((coefficients[j] >= 0) == (confidence >= y_dev.mean()))))
    return pd.DataFrame(rows).sort_values('coefficient', ascending=False)


def clf_write_predictions(ids, prob, output_path, expected_rows=1535):
    ids, prob = np.asarray(ids), np.asarray(prob, dtype=float)
    if len(ids) != expected_rows or len(prob) != expected_rows:
        raise ValueError(f'Expected exactly {expected_rows} test rows.')
    if pd.isna(ids).any() or pd.Series(ids).duplicated().any():
        raise ValueError('Test IDs must be nonmissing and unique.')
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError('Predictions must be finite probabilities in [0, 1].')
    frame = pd.DataFrame({'basket_id': ids, 'prob': prob})
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g')
    saved = pd.read_csv(path, dtype={'basket_id': str})
    if list(saved.columns) != ['basket_id', 'prob'] or len(saved) != expected_rows:
        raise AssertionError('CSV schema or row count changed during export.')
    if saved.basket_id.tolist() != [str(i) for i in ids]:
        raise AssertionError('CSV changed ID order or values.')
    if not np.allclose(saved.prob.to_numpy(), prob, atol=1e-15, rtol=1e-14):
        raise AssertionError('CSV changed probabilities during export.')
    return frame

def clf_rank(table):
    return table.sort_values(['ap', 'auc', 'name'], ascending=[False, False, True]).reset_index(drop=True)


def clf_rule_candidates(transformer):
    result = []
    for confidence in [.2, .3]:
        local = clone(transformer).set_params(min_confidence=confidence)
        result.append(dict(name=f'Rules__confidence_{confidence}', family='Rules',
            feature_set='rule_matches', params=json.dumps(local.get_params(), sort_keys=True),
            estimator=RuleBasedBasketClassifier(local)))
    return result


def clf_cv_evaluate(estimator, X, y, splits):
    oof = np.full(len(y), np.nan)
    counts = np.zeros(len(y), dtype=int)
    metrics, details = [], []
    for fold, (it, iv) in enumerate(splits, 1):
        if set(it) & set(iv):
            raise ValueError('CV training and validation rows overlap.')
        model = clone(estimator).fit(clf_take(X, it), y[it])
        visible_val = clf_take(X, iv)
        prob = model.predict_proba(visible_val)[:, 1]
        oof[iv] = prob
        counts[iv] += 1
        metrics.append(dict(fold=fold, n_train=len(it), n_validation=len(iv), **clf_metrics(y[iv], prob)))
        if isinstance(model, RuleBasedBasketClassifier):
            part = model.score_details(visible_val)
            part['row'] = iv
            part['fold'] = fold
            details.append(part)
    if not np.isfinite(oof).all() or not np.all(counts == 1):
        raise ValueError('Every development basket must have exactly one OOF prediction.')
    table = pd.DataFrame(metrics)
    summary = {c: float(table[c].mean()) for c in ['auc', 'ap', 'precision_at_top10', 'brier']}
    summary.update(auc_std=float(table.auc.std(ddof=0)), ap_std=float(table.ap.std(ddof=0)))
    detail = pd.concat(details, ignore_index=True).sort_values('row').reset_index(drop=True) if details else None
    return dict(summary=summary, oof=oof, folds=table, details=detail)


def clf_search_candidates(candidates, X, y, splits):
    records, registry, oof, details, fold_tables = [], {}, {}, {}, []
    for candidate in candidates:
        name = candidate['name']
        start = perf_counter()
        result = clf_cv_evaluate(candidate['estimator'], X, y, splits)
        records.append({k: v for k, v in candidate.items() if k != 'estimator'} |
                       result['summary'] | {'seconds': perf_counter() - start})
        registry[name], oof[name], details[name] = candidate['estimator'], result['oof'], result['details']
        fold_tables.append(result['folds'].assign(candidate=name))
    return dict(table=clf_rank(pd.DataFrame(records)), registry=registry, oof=oof, details=details,
                fold_metrics=pd.concat(fold_tables, ignore_index=True))


def clf_best_per_family(table):
    order = ['Logistic', 'MultinomialNB', 'RandomForest', 'Rules']
    best = clf_rank(table).groupby('family', sort=False).head(1).set_index('family')
    return best.loc[[family for family in order if family in best.index]].reset_index()


def clf_comparison_table(table):
    rows = []
    for _, r in table.iterrows():
        p = json.loads(r.params)
        if r.family == 'Logistic':
            setting = f"C={p['C']:g}; solver={p['solver']}; max_iter={p['max_iter']}"
        elif r.family == 'MultinomialNB':
            setting = f"alpha={p['alpha']:g}"
        elif r.family == 'RandomForest':
            setting = f"trees={p['n_estimators']}; depth={p['max_depth']}; min_leaf={p['min_samples_leaf']}"
        else:
            setting = f"support={p['min_support']}; confidence={p['min_confidence']}; lift={p['min_lift']}; max_rules={p['max_rules']}"
        rows.append({'Mô hình': CLF_MODEL_NAMES[r.family],
            'Bộ đặc trưng': CLF_FEATURE_NAMES.get(r.feature_set, r.feature_set), 'Tham số': setting,
            'AUC CV': r.auc, 'AP CV': r.ap, 'Precision@10% CV': r.precision_at_top10})
    return pd.DataFrame(rows)


def clf_feature_preview(matrix, feature_names, ids, n_rows=3):
    selected = np.asarray(matrix[:n_rows].sum(axis=0)).ravel() > 0
    preview = pd.DataFrame(matrix[:n_rows, selected].toarray(),
                           columns=np.asarray(feature_names)[selected], index=ids[:n_rows])
    preview.index.name = 'basket_id'
    return preview


def clf_validation_evaluation(search, best, X_train, y_train, X_val, y_val):
    rows, models, probabilities = [], {}, {}
    for _, r in best.iterrows():
        model = clone(search['registry'][r['name']]).fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        rows.append(dict(candidate=r['name'], family=r.family, **clf_metrics(y_val, prob)))
        models[r['name']], probabilities[r['name']] = model, prob
    prior = np.full(len(y_val), y_train.mean())
    rows.append(dict(candidate='Prior_baseline', family='Baseline', **clf_metrics(y_val, prior)))
    probabilities['Prior_baseline'] = prior
    return dict(metrics=pd.DataFrame(rows), models=models, probabilities=probabilities)


def clf_metric_table(table):
    result = table.copy()
    result['Mô hình'] = result.family.map(CLF_MODEL_NAMES)
    return result[['Mô hình', 'auc', 'ap', 'precision_at_top10', 'brier']].rename(columns={
        'auc': 'AUC validation', 'ap': 'AP validation',
        'precision_at_top10': 'Precision@10% validation', 'brier': 'Brier (calibration)'})


def clf_plot_calibration(evaluation, y_val, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    bins = []
    for _, row in evaluation['metrics'].iterrows():
        name = row.candidate
        observed, predicted = calibration_curve(y_val, evaluation['probabilities'][name],
                                                 n_bins=8, strategy='quantile')
        ax.plot(predicted, observed, marker='o', label=CLF_MODEL_NAMES[row.family])
        bins.extend({'candidate': name, 'mean_probability': float(p), 'fraction_positive': float(f)}
                    for p, f in zip(predicted, observed))
    ax.set(xlabel='Mean predicted probability', ylabel='Observed positive fraction',
           title='4.4 — Calibration on held-out validation')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    return pd.DataFrame(bins)


def clf_choose_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    index = np.flatnonzero(scores == scores.max())[-1]
    threshold = float(thresholds[index])
    rows = []
    for t in sorted(set([.2, .3, .5, threshold])):
        hard = probability >= t
        tn, fp, fn, tp = confusion_matrix(y, hard, labels=[0, 1]).ravel()
        rows.append(dict(threshold=t, precision=precision_score(y, hard, zero_division=0),
            recall=recall_score(y, hard, zero_division=0), f1=f1_score(y, hard, zero_division=0),
            false_positive=int(fp), false_negative=int(fn), n_recommended=int(hard.sum()),
            selected=t == threshold))
    return threshold, pd.DataFrame(rows)


def clf_threshold_validation(y, probability, threshold):
    hard = probability >= threshold
    matrix = pd.DataFrame(confusion_matrix(y, hard, labels=[0, 1]),
                           index=['actual_0', 'actual_1'], columns=['pred_0', 'pred_1'])
    report = pd.DataFrame(classification_report(y, hard, labels=[0, 1],
        output_dict=True, zero_division=0)).T.loc[['0', '1'], ['precision', 'recall', 'f1-score', 'support']]
    return matrix, report


def clf_cluster_test_matrix(test_frame, feature_names):
    """Supply WBS 5.0 with a fixed train vocabulary and record unknown test items."""
    names = list(feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError('Clustering vocabulary must contain unique training items.')
    columns = {item: j for j, item in enumerate(names)}
    baskets = clf_visible_baskets(test_frame.items_list.tolist())
    matrix = np.zeros((len(baskets), len(names)), dtype=bool)
    unseen = []
    for row, (basket_id, basket) in enumerate(zip(test_frame.basket_id, baskets)):
        for item in basket:
            if item in columns:
                matrix[row, columns[item]] = True
            else:
                unseen.append({'basket_id': basket_id, 'item': item})
    index = pd.Index(test_frame.basket_id.to_numpy(copy=True), name='basket_id')
    return (pd.DataFrame(matrix, index=index, columns=names),
            pd.DataFrame(unseen, columns=['basket_id', 'item']))


class WBS54ClassificationPredictor(BaseEstimator):
    """Same SVD/L2/KMeans as WBS 5.4, with explicit vocabulary instead of later globals."""
    def __init__(self, feature_names=None, n_clusters=8, n_components=20, random_state=42):
        self.feature_names = feature_names
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state

    def _to_matrix(self, X):
        if isinstance(X, pd.DataFrame):
            return X.to_numpy().astype(np.uint8)
        if sparse.issparse(X):
            return X.toarray().astype(np.uint8)
        if isinstance(X, np.ndarray):
            return X.astype(np.uint8)
        columns = {item: index for index, item in enumerate(self.feature_names)}
        matrix = np.zeros((len(X), len(columns)), dtype=np.uint8)
        for row, basket in enumerate(clf_visible_baskets(X)):
            for item in basket:
                if item in columns:
                    matrix[row, columns[item]] = 1
        return matrix

    def fit(self, X, y=None):
        matrix = self._to_matrix(X)
        n_components = min(self.n_components, matrix.shape[1] - 1)
        self.svd_ = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        reduced = normalize(self.svd_.fit_transform(matrix), norm='l2')
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(reduced)
        return self

    def predict(self, X):
        matrix = self._to_matrix(X)
        reduced = normalize(self.svd_.transform(matrix), norm='l2')
        return self.kmeans_.predict(reduced)


class ClusterLabelFeatures(TransformerMixin, BaseEstimator):
    """Adapt the unfitted predictor handed over in WBS 5.4; no clustering algorithm is invented here."""
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y=None):
        clf_visible_baskets(X)
        if self.estimator is None:
            raise ValueError('WBS 5.4 cluster predictor has not been handed over.')
        self.estimator_ = clone(self.estimator).fit(X)  # Never pass target labels to clustering.
        labels = np.asarray(self.estimator_.predict(X))
        if labels.ndim != 1 or len(labels) != len(X):
            raise ValueError('WBS 5.4 predict must return one cluster label per basket.')
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=True).fit(labels.reshape(-1, 1))
        self.n_fit_rows_ = len(X)
        return self

    def transform(self, X):
        check_is_fitted(self, 'estimator_')
        clf_visible_baskets(X)
        labels = np.asarray(self.estimator_.predict(X))
        if labels.ndim != 1 or len(labels) != len(X):
            raise ValueError('Invalid out-of-sample cluster labels.')
        return self.encoder_.transform(labels.reshape(-1, 1)).tocsr()


def clf_compare_clusters(search, X, y, splits, cluster_predictor=None):
    ordinary_rows = search['table'].family.ne('Rules') & ~search['table'].feature_set.str.endswith('_cluster')
    base = clf_rank(search['table'].loc[ordinary_rows]).iloc[0]
    rows = [dict(candidate=base['name'], with_cluster=False, auc=base.auc, ap=base.ap,
                 delta_auc=0.0, delta_ap=0.0, status='Đã có mô hình cơ sở')]
    if cluster_predictor is None:
        rows.append(dict(candidate='WBS 5.4 cluster predictor', with_cluster=True,
            auc=np.nan, ap=np.nan, delta_auc=np.nan, delta_ap=np.nan, status='Chờ bàn giao WBS 5.4'))
        return pd.DataFrame(rows), False
    ordinary = search['registry'][base['name']]
    estimator = Pipeline([
        ('features', FeatureUnion([('basket', clone(ordinary.named_steps['features'])),
                                  ('cluster', ClusterLabelFeatures(cluster_predictor))])),
        ('model', clone(ordinary.named_steps['model']))])
    result = clf_cv_evaluate(estimator, X, y, splits)
    name = base['name'] + '__cluster'
    record = dict(name=name, family=base.family, feature_set=base.feature_set + '_cluster',
                  params=base.params, seconds=np.nan, **result['summary'])
    remaining = search['table'].loc[search['table'].name.ne(name)]
    search['table'] = clf_rank(pd.concat([remaining, pd.DataFrame([record])], ignore_index=True))
    search['registry'][name], search['oof'][name] = estimator, result['oof']
    if 'fold_metrics' in search:
        folds = search['fold_metrics'].loc[search['fold_metrics'].candidate.ne(name)]
        search['fold_metrics'] = pd.concat([folds, result['folds'].assign(candidate=name)], ignore_index=True)
    rows.append(dict(candidate=name, with_cluster=True, auc=result['summary']['auc'], ap=result['summary']['ap'],
        delta_auc=result['summary']['auc'] - base.auc, delta_ap=result['summary']['ap'] - base.ap,
        status='Đã so cùng các fold'))
    return pd.DataFrame(rows), True
