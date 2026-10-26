import json

import numpy as np
import pandas as pd
import pytest

from retail_insights import prediction as p


def sample_table(task='purchase'):
    rows = []
    for cutoff in p.DEFAULT_CUTOFFS:
        for i in range(24):
            row = {'CustomerID': str(i), 'Cutoff': pd.Timestamp(cutoff), 'Task': task, 'Target': int(i % 4 == 0)}
            row.update({name: float((i * 7) % 13 + 1) for name in p.FEATURES})
            row['interval_missing'] = 0
            if i == 1:
                row['interval_mean_days'] = np.nan
                row['interval_missing'] = 1
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs.update(task=task, horizon_days=30 if task == 'purchase' else 60, window_days=90,
                       coverage_start='2010-12-01', coverage_end='2011-12-09',
                       features=p.FEATURES, feature_version='1', cutoffs=p.DEFAULT_CUTOFFS)
    return frame


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    folder = tmp_path_factory.mktemp('prediction')
    samples = sample_table()
    return samples, folder, p.train_experiment(samples, 'purchase', folder, dataset_id='example')


def test_candidate_comparison_and_frozen_test(experiment):
    samples, folder, result = experiment
    metrics = result['metrics']
    assert len(metrics) == 17
    val = metrics[metrics.Split == 'validation']
    assert len(val) == 16
    for model in ['logistic', 'tree', 'forest']:
        assert set(val[val.Model == model].Strategy) == {'none', 'under', 'over', 'smote', 'weight'}
    assert len(metrics[metrics.Split == 'test']) == 1
    assert set(val.TrainRows) == {96}
    assert set(val.TrainPositiveRate) == {.25}
    assert set(val[val.Model != 'baseline'].InnerFolds) == {2}
    assert len(result['predictions']) == 24
    assert result['metadata']['refit_after_selection'] is False
    assert result['metadata']['available_from'] == '2011-08-31T00:00:00'
    assert result['metadata']['probabilities_calibrated'] is False
    assert np.asarray(result['metadata']['confusion_matrix']).shape == (2, 2)
    assert set(result['importance'].Feature) == set(p.FEATURES)
    assert set(result['curves']['roc'].columns) == {'FalsePositiveRate', 'TruePositiveRate'}
    pipeline, metadata = p.read_artifact(folder, expected_dataset_id='example')
    test = samples[samples.Cutoff == pd.Timestamp('2011-10-01')]
    np.testing.assert_allclose(pipeline.predict_proba(test[p.FEATURES])[:, 1], result['predictions'].Probability)
    for filename in ['pipeline.joblib', 'metadata.json', 'metrics.csv', 'curves_roc.csv',
                     'curves_pr.csv', 'importance.csv', 'predictions.parquet']:
        assert (folder / filename).is_file()
    json.dumps(metadata, allow_nan=False)
    with pytest.raises(ValueError, match='dataset|数据'):
        p.read_artifact(folder, expected_dataset_id='different')


@pytest.mark.parametrize('problem', ['oneclass_train', 'oneclass_validation', 'duplicate', 'task',
                                      'missingcutoff', 'invalidfeature', 'window', 'coverage', 'horizon'])
def test_reject_invalid_samples(tmp_path, problem):
    samples = sample_table()
    if problem == 'oneclass_train':
        samples.loc[samples.Cutoff < '2011-08-01', 'Target'] = 1
    elif problem == 'oneclass_validation':
        samples.loc[samples.Cutoff == '2011-08-01', 'Target'] = 0
    elif problem == 'duplicate':
        samples.loc[1, 'CustomerID'] = samples.loc[0, 'CustomerID']
    elif problem == 'task':
        samples.loc[0, 'Task'] = 'churn'
    elif problem == 'missingcutoff':
        samples = samples[samples.Cutoff != '2011-04-01']
    elif problem == 'invalidfeature':
        samples.loc[0, 'recency_days'] = np.inf
    elif problem == 'window':
        samples.attrs['window_days'] = 60
    elif problem == 'coverage':
        samples.attrs['coverage_end'] = '2011-10-10'
    elif problem == 'horizon':
        samples.attrs['horizon_days'] = 90
    with pytest.raises(ValueError):
        p.train_experiment(samples, 'purchase', tmp_path)


def test_corrupt_schema_rejected(experiment, tmp_path):
    import shutil
    _, folder, _ = experiment
    for filename in ['pipeline.joblib', 'metadata.json']:
        shutil.copyfile(folder / filename, tmp_path / filename)
    metadata = json.loads((tmp_path / 'metadata.json').read_text(encoding='utf-8'))
    metadata['features'] = list(reversed(p.FEATURES))
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='schema|特征'):
        p.read_artifact(tmp_path)


def test_inference_rejects_model_before_selection_available(experiment):
    _, folder, _ = experiment
    with pytest.raises(ValueError, match='available|可用'):
        p.infer(pd.DataFrame(), folder, cutoff='2011-08-01', expected_dataset_id='example')


def test_threshold_ties_choose_higher_and_single_class_metrics_are_safe():
    assert p._best_threshold(np.array([0, 1]), np.array([.2, .8])) == .8
    scores = p._scores(np.array([0, 0]), np.array([.1, .2]), .5)
    assert scores['AveragePrecision'] is None
    assert scores['ROC_AUC'] is None


def test_training_and_resampling_never_fit_holdouts(monkeypatch, tmp_path):
    """Inspect real pipeline inputs, while retaining real model fits/resampling."""
    samples = sample_table()
    # Date-coded features let us detect each split even after IDs are removed.
    codes = {pd.Timestamp(date): i + 1 for i, date in enumerate(p.DEFAULT_CUTOFFS)}
    samples['history_span_days'] = samples.Cutoff.map(codes).astype(float)
    calls = []
    original = p.Pipeline.fit

    def recording_fit(self, features, target, **kwargs):
        calls.append((features.copy(), target.copy()))
        return original(self, features, target, **kwargs)

    monkeypatch.setattr(p.Pipeline, 'fit', recording_fit)
    p.train_experiment(samples, 'purchase', tmp_path)
    assert len(calls) == 76  # baseline + 15 * (2 configs * 2 inner folds + final fit)
    for features, target in calls:
        assert list(features.columns) == p.FEATURES
        assert set(features.history_span_days) in [{1.}, {1., 2.}, {1., 2., 3., 4.}]
        assert len(features) in [24, 48, 96]
        assert target.mean() == .25


def test_minority_shortage_skips_smote_and_invalid_inner_folds(tmp_path):
    samples = sample_table()
    train_mask = samples.Cutoff < '2011-08-01'
    samples.loc[train_mask, 'Target'] = 0
    samples.loc[(samples.Cutoff == '2011-06-01') & (samples.CustomerID == '0'), 'Target'] = 1
    samples.loc[samples.Cutoff == '2011-10-01', 'Target'] = 0
    result = p.train_experiment(samples, 'purchase', tmp_path)
    metrics = result['metrics']
    assert len(metrics) == 17
    skipped = metrics[metrics.Status == 'skipped']
    assert len(skipped) == 3
    assert set(skipped.Strategy) == {'smote'}
    assert skipped.Reason.str.contains('minority').all()
    val = metrics[(metrics.Split == 'validation') & (metrics.Model != 'baseline') & (metrics.Status == 'ok')]
    assert set(val.InnerFolds) == {0}
    assert val.Reason.str.contains('without tuning').all()
    assert result['metadata']['test_scores']['AveragePrecision'] is None
    assert result['metadata']['test_scores']['ROC_AUC'] is None
    assert result['curves']['roc'].empty
    assert result['curves']['pr'].empty


def test_real_feature_module_inference_roundtrip(experiment):
    from retail_insights.prediction_features import feature_frame
    _, folder, result = experiment
    sales = pd.DataFrame([
        ['A', '1', '2010-12-01', 'X', 1, 100],
        ['A', '2', '2011-09-20', 'X', 2, 300],
        ['B', '3', '2011-09-29', 'Y', 3, 600],
        ['A', '4', '2011-12-09', 'Z', 4, 800],
    ], columns=['CustomerID', 'InvoiceNo', 'InvoiceDate', 'StockCode', 'Quantity', 'AmountCents'])
    sales.InvoiceDate = pd.to_datetime(sales.InvoiceDate)
    predicted = p.infer(sales, folder, expected_dataset_id='example')
    features = feature_frame(sales, '2011-12-10')
    pipeline, _ = p.read_artifact(folder)
    assert list(predicted.columns) == ['CustomerID', 'Cutoff', *p.FEATURES, 'Probability', 'Prediction']
    assert predicted.Cutoff.eq(pd.Timestamp('2011-12-10')).all()
    assert set(predicted.CustomerID) == {'A', 'B'}
    np.testing.assert_allclose(predicted.Probability, pipeline.predict_proba(features[p.FEATURES])[:, 1])
    np.testing.assert_array_equal(predicted.Prediction, predicted.Probability.ge(result['threshold']).astype(int))
    noon = p.infer(sales, folder, cutoff=pd.Timestamp('2011-10-01 12:00'))
    midnight = p.infer(sales, folder, cutoff=pd.Timestamp('2011-10-01'))
    pd.testing.assert_frame_equal(noon, midnight)


def test_inference_rejects_runtime_feature_version(experiment, monkeypatch):
    from retail_insights import prediction_features
    _, folder, _ = experiment
    monkeypatch.setattr(prediction_features, 'FEATURE_VERSION', '999')
    sales = pd.DataFrame({'InvoiceDate': pd.to_datetime(['2010-12-01', '2011-12-09'])})
    with pytest.raises(ValueError, match='schema|version|特征'):
        p.infer(sales, folder)


def test_missing_available_from_rejected(experiment, tmp_path):
    import shutil
    _, folder, _ = experiment
    shutil.copyfile(folder / 'pipeline.joblib', tmp_path / 'pipeline.joblib')
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    metadata.pop('available_from')
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='available|可用'):
        p.read_artifact(tmp_path)


@pytest.mark.parametrize('content', [b'', b'\x80\x04\x95', b'not a pickle'])
def test_corrupt_pipeline_reports_readable_error(experiment, tmp_path, content):
    import shutil
    _, folder, _ = experiment
    shutil.copyfile(folder / 'metadata.json', tmp_path / 'metadata.json')
    (tmp_path / 'pipeline.joblib').write_bytes(content)
    with pytest.raises(ValueError, match='损坏或不兼容'):
        p.read_artifact(tmp_path)


def test_pipeline_missing_dependency_reports_incompatibility(experiment, monkeypatch):
    _, folder, _ = experiment

    def missing_dependency(*args, **kwargs):
        raise ModuleNotFoundError('Historical model module was removed')

    monkeypatch.setattr(p.joblib, 'load', missing_dependency)
    with pytest.raises(ValueError, match='损坏或不兼容') as caught:
        p.read_artifact(folder)
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


def test_pipeline_load_keyboard_interrupt_propagates(experiment, monkeypatch):
    _, folder, _ = experiment

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(p.joblib, 'load', interrupt)
    with pytest.raises(KeyboardInterrupt):
        p.read_artifact(folder)
