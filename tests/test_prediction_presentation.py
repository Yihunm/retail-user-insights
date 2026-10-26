import io
import json
import zipfile

import pandas as pd
import pytest


def test_prediction_export_preserves_task_threshold_and_filtered_customers():
    from retail_insights.prediction_presentation import export_customer_predictions

    frame = pd.DataFrame({'CustomerID': ['A00'], 'Cutoff': pd.to_datetime(['2011-12-10']),
                          'Probability': [.7], 'Prediction': [1], 'recency_days': [3]})
    context = {'task': 'churn', 'threshold': .6, 'dataset_id': 'dataset-a', 'run_id': 'run-a',
                'horizon_days': 60, 'available_from': '2011-09-30', 'probabilities_calibrated': False}
    content = export_customer_predictions(frame, context, customer_search='A00')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        data = pd.read_csv(io.BytesIO(archive.read('客户预测.csv')))
        assert data['预测判定'].tolist() == ['高风险']
        assert data['预测分数'].tolist() == [.7]
        description = json.loads(archive.read('预测说明.json'))
        assert description['任务'] == '未来60天未购买风险'
        assert description['判定阈值'] == .6
        assert description['已校准'] is False
        assert description['客户编号搜索'] == 'A00'


def record_files(tmp_path, metadata):
    folder = tmp_path / 'artifacts' / 'prediction' / 'broken'
    folder.mkdir(parents=True)
    for name in ['pipeline.joblib', 'metrics.csv', 'predictions.parquet', 'importance.csv',
                 'curves_roc.csv', 'curves_pr.csv']:
        (folder / name).write_bytes(b'')
    (folder / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    return folder


def test_saved_records_skip_metadata_missing_required_fields(tmp_path):
    from retail_insights.prediction_presentation import saved_records

    record_files(tmp_path, {'dataset_id': 'v', 'task': 'purchase'})
    assert saved_records(tmp_path, 'v') == []


def test_saved_records_ignore_incomplete_run_even_if_core_artifacts_exist(tmp_path):
    from retail_insights.prediction_presentation import saved_records

    record_files(tmp_path, {'dataset_id': 'v', 'task': 'purchase', 'created_at': '2011-01-01',
                            'best_model': 'forest'})
    assert saved_records(tmp_path, 'v') == []


def test_samples_write_failure_does_not_publish_half_finished_model(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file, load_dataset
    from retail_insights.prediction_presentation import run_prediction, saved_records
    from test_prediction_page import prediction_source

    metadata = prepare_file(prediction_source(tmp_path), root=tmp_path)
    data = load_dataset(tmp_path)
    original = pd.DataFrame.to_parquet

    def disk_full(frame, path, *args, **kwargs):
        if str(path).endswith('samples.parquet'):
            raise OSError('simulated disk full')
        return original(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, 'to_parquet', disk_full)
    with pytest.raises(OSError, match='disk full'):
        run_prediction(data['sales'], 'purchase', tmp_path, metadata['dataset_id'])
    assert saved_records(tmp_path, metadata['dataset_id']) == []
