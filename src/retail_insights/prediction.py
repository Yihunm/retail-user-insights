"""Chronological, training-only preprocessing and frozen holdout evaluation.

Artifacts contain pickle-based joblib objects: load only trusted local directories.
Probabilities are raw model scores and have not been calibrated.
"""
from __future__ import annotations

import json
import pickle
import struct
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

FEATURES = ['recency_days', 'orders_7', 'orders_30', 'orders_90', 'revenue_7', 'revenue_30',
            'revenue_90', 'quantity_90', 'unique_products_90', 'active_days_90',
            'average_order_value_90', 'max_order_value_90', 'interval_mean_days',
            'interval_missing', 'history_span_days']
FEATURE_VERSION = '1'
TRAIN_CUTOFFS = ['2011-03-02', '2011-04-01', '2011-05-01', '2011-06-01']
VALIDATION_CUTOFF = '2011-08-01'
TEST_CUTOFF = '2011-10-01'
DEFAULT_CUTOFFS = TRAIN_CUTOFFS + [VALIDATION_CUTOFF, TEST_CUTOFF]
STRATEGIES = ['none', 'under', 'over', 'smote', 'weight']
INNER_FOLDS = [(['2011-03-02'], '2011-05-01'),
               (['2011-03-02', '2011-04-01'], '2011-06-01')]


def _timestamp(value):
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid timestamp / 无效日期') from error
    if pd.isna(timestamp) or timestamp.tzinfo is not None:
        raise ValueError('Use a valid timezone-naive timestamp / 请使用不含时区的有效日期')
    return timestamp


def _validate_samples(samples, task):
    if task not in ('purchase', 'churn'):
        raise ValueError('Task must be purchase or churn')
    required = ['CustomerID', 'Cutoff', 'Task', *FEATURES, 'Target']
    if not isinstance(samples, pd.DataFrame) or not set(required).issubset(samples.columns):
        raise ValueError('Sample schema is missing required columns / 样本缺少必要列')
    if samples.columns.duplicated().any():
        raise ValueError('Duplicate feature columns')
    attrs = samples.attrs
    if attrs.get('task') != task or not samples.Task.eq(task).all():
        raise ValueError('Sample Task does not match requested task')
    horizon = 30 if task == 'purchase' else 60
    if attrs.get('horizon_days') != horizon or attrs.get('window_days') != 90:
        raise ValueError('Expected 90 day history and task-specific 30/60 day horizon')
    if attrs.get('features') != FEATURES or str(attrs.get('feature_version')) != FEATURE_VERSION:
        raise ValueError('Feature schema/version mismatch / 特征模式或版本不匹配')
    data = samples.copy()
    data['Cutoff'] = data.Cutoff.map(_timestamp)
    if set(data.Cutoff) != set(map(pd.Timestamp, DEFAULT_CUTOFFS)):
        raise ValueError('All six standard chronological cutoffs are required')
    if data.CustomerID.isna().any() or data.CustomerID.astype(str).str.strip().eq('').any():
        raise ValueError('CustomerID cannot be missing')
    if data.duplicated(['CustomerID', 'Cutoff']).any():
        raise ValueError('Duplicate CustomerID + Cutoff sample key')
    if data.Target.isna().any() or not data.Target.isin([0, 1]).all():
        raise ValueError('Target must contain binary labels 0/1')
    for feature in FEATURES:
        if not pd.api.types.is_numeric_dtype(data[feature]):
            raise ValueError(f'Feature must be numeric: {feature}')
        values = data[feature].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or (values[~np.isnan(values)] < 0).any():
            raise ValueError(f'Invalid feature values: {feature}')
        if feature != 'interval_mean_days' and np.isnan(values).any():
            raise ValueError(f'Missing feature values: {feature}')
    if not data.interval_missing.isin([0, 1]).all():
        raise ValueError('interval_missing must be binary')
    if not data.interval_mean_days.isna().eq(data.interval_missing.eq(1)).all():
        raise ValueError('Missing interval must have interval_missing=1')
    start, end = _timestamp(attrs.get('coverage_start')), _timestamp(attrs.get('coverage_end'))
    if start > min(data.Cutoff) - timedelta(days=90):
        raise ValueError('Incomplete 90 day historical coverage')
    if end < max(data.Cutoff) + timedelta(days=horizon):
        raise ValueError('Incomplete label coverage')
    if (pd.Timestamp(TRAIN_CUTOFFS[-1]) + timedelta(days=horizon) > pd.Timestamp(VALIDATION_CUTOFF)
            or pd.Timestamp(VALIDATION_CUTOFF) + timedelta(days=horizon) > pd.Timestamp(TEST_CUTOFF)):
        raise ValueError('Label embargo is violated')
    for cutoffs, name in [(TRAIN_CUTOFFS, 'training'), ([VALIDATION_CUTOFF], 'validation')]:
        if data.loc[data.Cutoff.isin(pd.to_datetime(cutoffs)), 'Target'].nunique() != 2:
            raise ValueError(f'{name} requires two classes / 训练集和验证集必须有两类标签')
    return data, horizon


def _configs(model):
    if model == 'logistic':
        return [{'C': .1, 'max_iter': 2000}, {'C': 1., 'max_iter': 2000}]
    if model == 'tree':
        return [{'max_depth': depth, 'min_samples_leaf': 10} for depth in [4, 8]]
    if model == 'forest':
        return [{'n_estimators': 120, 'max_depth': depth, 'min_samples_leaf': 5} for depth in [6, 12]]
    return [{}]


def _pipeline(model, strategy, params, target, seed):
    weight = 'balanced' if strategy == 'weight' else None
    if model == 'baseline':
        classifier = DummyClassifier(strategy='prior')
    elif model == 'logistic':
        classifier = LogisticRegression(**params, class_weight=weight, random_state=seed)
    elif model == 'tree':
        classifier = DecisionTreeClassifier(**params, class_weight=weight, random_state=seed)
    else:
        classifier = RandomForestClassifier(**params, class_weight=weight, random_state=seed, n_jobs=1)
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
             ('scaler', StandardScaler())]
    if strategy == 'under':
        steps.append(('sampler', RandomUnderSampler(random_state=seed)))
    elif strategy == 'over':
        steps.append(('sampler', RandomOverSampler(random_state=seed)))
    elif strategy == 'smote':
        minority = int(pd.Series(target).value_counts().min())
        if minority < 2:
            raise ValueError('SMOTE needs at least two minority training samples')
        steps.append(('sampler', SMOTE(k_neighbors=min(5, minority - 1), random_state=seed)))
    steps.append(('classifier', classifier))
    return Pipeline(steps)


def _best_threshold(target, probability):
    # Only actual decision boundaries are needed; >= implements the threshold.
    precision, recall, thresholds = precision_recall_curve(target, probability)
    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return float(thresholds[np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]])


def _scores(target, probability, threshold):
    predicted = np.asarray(probability) >= threshold
    return {
        'AveragePrecision': float(average_precision_score(target, probability)) if np.any(np.asarray(target) == 1) else None,
        'Accuracy': float(accuracy_score(target, predicted)),
        'Precision': float(precision_score(target, predicted, zero_division=0)),
        'Recall': float(recall_score(target, predicted, zero_division=0)),
        'F1': float(f1_score(target, predicted, zero_division=0)),
        'ROC_AUC': float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
    }


def _tune(data, model, strategy, horizon, seed):
    records = []
    for params in _configs(model):
        scores, folds = [], []
        for train_dates, validation_date in INNER_FOLDS:
            train = data[data.Cutoff.isin(pd.to_datetime(train_dates))]
            validation = data[data.Cutoff == pd.Timestamp(validation_date)]
            reason = ''
            if max(pd.to_datetime(train_dates)) + timedelta(days=horizon) > pd.Timestamp(validation_date):
                reason = 'Label embargo'
            elif train.Target.nunique() < 2 or validation.Target.nunique() < 2:
                reason = 'Both classes required in inner training and validation'
            elif strategy == 'smote' and train.Target.value_counts().min() < 2:
                reason = 'Insufficient minority training samples for SMOTE'
            fold = {'train_cutoffs': train_dates, 'validation_cutoff': validation_date,
                    'train_rows': len(train), 'validation_rows': len(validation), 'reason': reason}
            if not reason:
                pipeline = _pipeline(model, strategy, params, train.Target, seed)
                pipeline.fit(train[FEATURES], train.Target)
                score = float(average_precision_score(validation.Target, pipeline.predict_proba(validation[FEATURES])[:, 1]))
                scores.append(score)
                fold['AveragePrecision'] = score
            folds.append(fold)
        records.append({'params': params, 'mean_ap': float(np.mean(scores)) if scores else None,
                        'valid_folds': len(scores), 'folds': folds})
    # Stable ordering selects the first limited configuration on ties or no usable folds.
    best = max(range(len(records)), key=lambda i: records[i]['mean_ap'] if records[i]['mean_ap'] is not None else -1)
    return records[best], records


def train_experiment(samples, task, output_dir, dataset_id=None, random_state=42):
    """Compare 16 candidates on August; score October once after selection.

    Two forward folds tune exactly two configurations per nonbaseline candidate.
    Identifiers are excluded and transformations/resampling are fit on train only.
    No refitting follows validation selection; the saved object is that fitted model.
    """
    data, horizon = _validate_samples(samples, task)
    train = data[data.Cutoff.isin(pd.to_datetime(TRAIN_CUTOFFS))]
    validation = data[data.Cutoff == pd.Timestamp(VALIDATION_CUTOFF)]
    test = data[data.Cutoff == pd.Timestamp(TEST_CUTOFF)]
    rows, tuning, winner = [], {}, None
    candidates = [('baseline', 'baseline')] + [(model, strategy) for model in ['logistic', 'tree', 'forest'] for strategy in STRATEGIES]
    with threadpool_limits(limits=1):
        for model, strategy in candidates:
            candidate = f'{model}_{strategy}'
            row = {'Candidate': candidate, 'Model': model, 'Strategy': strategy, 'Split': 'validation',
                   'Threshold': None, 'AveragePrecision': None, 'Accuracy': None, 'Precision': None,
                   'Recall': None, 'F1': None, 'ROC_AUC': None, 'TrainRows': len(train),
                   'TrainPositiveRate': float(train.Target.mean()), 'InnerCV_AP': None,
                   'Params': '{}', 'Status': 'ok', 'Reason': '', 'InnerFolds': 0}
            if strategy == 'smote' and train.Target.value_counts().min() < 2:
                row.update(Status='skipped', Reason='SMOTE needs at least two minority training samples')
                rows.append(row)
                continue
            params = {}
            if model != 'baseline':
                selected, records = _tune(data, model, strategy, horizon, random_state)
                tuning[candidate] = records
                params = selected['params']
                row.update(InnerCV_AP=selected['mean_ap'], InnerFolds=selected['valid_folds'])
                if not selected['valid_folds']:
                    row['Reason'] = 'No valid forward folds; first configuration used without tuning'
            row['Params'] = json.dumps(params, sort_keys=True)
            pipeline = _pipeline(model, strategy, params, train.Target, random_state)
            pipeline.fit(train[FEATURES], train.Target)
            probabilities = pipeline.predict_proba(validation[FEATURES])[:, 1]
            threshold = _best_threshold(validation.Target.to_numpy(), probabilities)
            row.update(Threshold=threshold, **_scores(validation.Target, probabilities, threshold))
            rows.append(row)
            rank = (row['AveragePrecision'], row['F1'])
            if winner is None or rank > winner['rank']:
                winner = {'rank': rank, 'row': row.copy(), 'pipeline': pipeline, 'params': params}
        pipeline = winner['pipeline']
        threshold = winner['row']['Threshold']
        # The holdout has not been transformed or scored before this point.
        probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
    test_row = {**winner['row'], 'Split': 'test', **_scores(test.Target, probabilities, threshold)}
    rows.append(test_row)
    metrics = pd.DataFrame(rows)
    predictions = test[['CustomerID', 'Cutoff', 'Target']].reset_index(drop=True)
    predictions['Probability'] = probabilities
    predictions['Prediction'] = (probabilities >= threshold).astype(int)
    if test.Target.nunique() == 2:
        fpr, tpr, _ = roc_curve(test.Target, probabilities)
        roc = pd.DataFrame({'FalsePositiveRate': fpr, 'TruePositiveRate': tpr})
    else:
        roc = pd.DataFrame(columns=['FalsePositiveRate', 'TruePositiveRate'])
    if test.Target.eq(1).any():
        precision, recall, _ = precision_recall_curve(test.Target, probabilities)
        pr = pd.DataFrame({'Recall': recall, 'Precision': precision})
    else:
        pr = pd.DataFrame(columns=['Recall', 'Precision'])
    classifier = pipeline.named_steps['classifier']
    importance_values = getattr(classifier, 'feature_importances_', np.zeros(len(FEATURES)))
    if hasattr(classifier, 'coef_'):
        importance_values = np.abs(classifier.coef_[0])
    importance = pd.DataFrame({'Feature': FEATURES, 'Importance': importance_values})
    split_details = {}
    for name, frame in [('train', train), ('validation', validation), ('test', test)]:
        split_details[name] = {'cutoffs': sorted(frame.Cutoff.dt.strftime('%Y-%m-%d').unique().tolist()),
                               'rows': len(frame), 'positive_rate': float(frame.Target.mean()),
                               'class_counts': {str(int(k)): int(v) for k, v in frame.Target.value_counts().items()}}
    metadata = {
        'artifact_version': '1', 'task': task, 'features': FEATURES.copy(), 'feature_version': FEATURE_VERSION,
        'window_days': 90, 'horizon_days': horizon, 'dataset_id': dataset_id,
        'seed': random_state, 'sklearn_version': sklearn.__version__, 'splits': split_details,
        'coverage_start': _timestamp(samples.attrs['coverage_start']).isoformat(),
        'coverage_end': _timestamp(samples.attrs['coverage_end']).isoformat(),
        'training_cutoff_start': train.Cutoff.min().isoformat(), 'training_cutoff_end': train.Cutoff.max().isoformat(),
        'available_from': (pd.Timestamp(VALIDATION_CUTOFF) + timedelta(days=horizon)).isoformat(),
        'selected_candidate': winner['row']['Candidate'], 'best_model': winner['row']['Model'],
        'selected_strategy': winner['row']['Strategy'], 'threshold': threshold, 'selected_params': winner['params'],
        'probabilities_calibrated': False, 'refit_after_selection': False,
        'probability_note': 'Raw model probabilities; not calibrated / 原始模型概率，未校准',
        'selection_metric': 'AveragePrecision then validation F1; stable candidate order on ties',
        'threshold_rule': 'Max validation F1; highest threshold on ties', 'tuning': tuning,
        'tuning_folds': [{'train_cutoffs': train_dates, 'validation_cutoff': date} for train_dates, date in INNER_FOLDS],
        'confusion_matrix': confusion_matrix(test.Target, predictions.Prediction, labels=[0, 1]).tolist(),
        'validation_scores': {key: winner['row'][key] for key in _scores(validation.Target, np.zeros(len(validation)), 1)},
        'test_scores': {key: test_row[key] for key in _scores(test.Target, probabilities, threshold)},
        'importance_method': 'absolute standardized coefficient' if hasattr(classifier, 'coef_') else
                             ('impurity importance' if hasattr(classifier, 'feature_importances_') else 'baseline zeros'),
        'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
    }
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, folder / 'pipeline.joblib')
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    metrics.to_csv(folder / 'metrics.csv', index=False)
    roc.to_csv(folder / 'curves_roc.csv', index=False)
    pr.to_csv(folder / 'curves_pr.csv', index=False)
    importance.to_csv(folder / 'importance.csv', index=False)
    predictions.to_parquet(folder / 'predictions.parquet', index=False)
    return {'metrics': metrics, 'metadata': metadata, 'curves': {'roc': roc, 'pr': pr},
            'importance': importance, 'predictions': predictions, 'artifact_dir': str(folder.resolve()),
            'best_model': winner['row']['Model'], 'threshold': threshold}


def read_artifact(folder, expected_dataset_id=None):
    """Read only application-owned, trusted local artifacts (joblib is executable)."""
    folder = Path(folder)
    try:
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('Cannot read model metadata / 无法读取模型元数据') from error
    if (metadata.get('features') != FEATURES or str(metadata.get('feature_version')) != FEATURE_VERSION
            or metadata.get('artifact_version') != '1'):
        raise ValueError('Feature schema/version mismatch / 特征模式或版本不匹配')
    if metadata.get('task') not in ('purchase', 'churn'):
        raise ValueError('Invalid model task')
    if metadata.get('window_days') != 90 or metadata.get('horizon_days') != (30 if metadata['task'] == 'purchase' else 60):
        raise ValueError('Invalid model history window or label horizon')
    if expected_dataset_id is not None and metadata.get('dataset_id') != expected_dataset_id:
        raise ValueError('Model dataset mismatch / 模型与当前数据集不匹配')
    threshold = metadata.get('threshold')
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid model threshold')
    if not metadata.get('available_from'):
        raise ValueError('Model available_from is required / 模型可用日期缺失')
    _timestamp(metadata['available_from'])
    try:
        pipeline = joblib.load(folder / 'pipeline.joblib')
    except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError,
            ImportError, AttributeError, IndexError, KeyError, struct.error, OverflowError) as error:
        # Truncated joblib streams can also fail inside pickle's opcode dispatch
        # (KeyError/IndexError/struct.error), rather than raising UnpicklingError.
        raise ValueError('模型文件损坏或不兼容，请重新训练模型 / Model file is corrupt or incompatible') from error
    if (not isinstance(pipeline, Pipeline) or getattr(pipeline, 'n_features_in_', None) != len(FEATURES)
            or list(getattr(pipeline, 'feature_names_in_', [])) != FEATURES
            or list(getattr(pipeline, 'classes_', [])) != [0, 1]):
        raise ValueError('Model feature schema/class mismatch')
    return pipeline, metadata


def infer(sales, folder, cutoff=None, expected_dataset_id=None):
    """Score known customers with the original, frozen feature schema and threshold."""
    pipeline, metadata = read_artifact(folder, expected_dataset_id)
    if cutoff is not None:
        cutoff = _timestamp(cutoff).normalize()
        if cutoff < _timestamp(metadata['available_from']):
            raise ValueError('Model is not available before selection labels end / 早于模型可用日期')
    if not isinstance(sales, pd.DataFrame) or 'InvoiceDate' not in sales or sales.empty:
        raise ValueError('Nonempty valid sales are required')
    dates = pd.to_datetime(sales.InvoiceDate, errors='coerce')
    if dates.isna().any():
        raise ValueError('Invalid sales dates')
    if cutoff is None:
        cutoff = dates.max().normalize() + timedelta(days=1)
    if cutoff < _timestamp(metadata['available_from']):
        raise ValueError('Model is not available before selection labels end / 早于模型可用日期')
    if dates.min().normalize() > cutoff - timedelta(days=90):
        raise ValueError('At least 90 days of historical coverage are required')
    from . import prediction_features
    if (list(prediction_features.FEATURES) != FEATURES
            or str(prediction_features.FEATURE_VERSION) != metadata['feature_version']):
        raise ValueError('Inference feature schema/version mismatch')
    features = prediction_features.feature_frame(sales, cutoff, window_days=90)
    if list(features.columns) != ['CustomerID', *FEATURES]:
        raise ValueError('Inference feature schema mismatch')
    if features.empty:
        raise ValueError('No known customers with positive sales in the feature window')
    if features.CustomerID.isna().any() or features.CustomerID.duplicated().any():
        raise ValueError('Invalid customer identifiers in feature frame')
    result = features.copy()
    result.insert(1, 'Cutoff', cutoff)
    with threadpool_limits(limits=1):
        result['Probability'] = pipeline.predict_proba(features[FEATURES])[:, 1]
    result['Prediction'] = (result.Probability >= metadata['threshold']).astype(int)
    return result
