from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'


def prediction_source(tmp_path):
    from retail_insights.prediction_features import DEFAULT_CUTOFFS

    rows = []
    for i in range(12):
        for j, day in enumerate(pd.date_range('2010-12-02', '2011-12-08', freq='10D')):
            rows.append([f'A{i}-{j}', 'SKU', 'item', 1, str(day), i + 1, f'A{i:02d}', 'UK'])
        for j, cutoff in enumerate(DEFAULT_CUTOFFS):
            day = pd.Timestamp(cutoff) - timedelta(days=1)
            rows.append([f'B{i}-{j}', 'SKU', 'item', 1, str(day), i + 1, f'B{i:02d}', 'UK'])
    for invoice, day in [('start', '2010-12-01 08:00:00'), ('end', '2011-12-09 12:00:00')]:
        rows.append([invoice, 'SKU', 'item', 1, day, 1, None, 'UK'])
    source = tmp_path / 'prediction-orders.csv'
    pd.DataFrame(rows, columns=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
                                'UnitPrice', 'CustomerID', 'Country']).to_csv(source, index=False)
    return source


def open_prediction(tmp_path, monkeypatch, imported=False):
    from retail_insights.preparation import prepare_file

    if imported:
        prepare_file(prediction_source(tmp_path), root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.sidebar.radio(key='module').set_value('复购与流失预测').run(timeout=30)
    return app


def test_prediction_without_data_guides_import(tmp_path, monkeypatch):
    app = open_prediction(tmp_path, monkeypatch)
    assert not app.exception
    assert any('导入' in info.value for info in app.info)


def test_prediction_training_inference_loading_and_version_invalidation(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file
    from test_preparation import source_file

    app = open_prediction(tmp_path, monkeypatch, imported=True)
    assert not app.exception
    next(button for button in app.button if button.label == '构建样本并训练').click().run(timeout=120)
    assert not app.exception
    context = app.session_state['prediction_context']
    run_id = context['run_id']
    assert context['task'] == 'purchase'
    assert app.session_state['prediction_result']['metrics'].Split.eq('validation').sum() == 16
    app.date_input(key='prediction_infer_date').set_value(date(2011, 10, 1))
    next(button for button in app.button if button.label == '生成客户预测').click().run(timeout=60)
    assert not app.exception
    predictions = app.session_state['prediction_inference']
    assert predictions.CustomerID.nunique() == 24
    assert 'Target' not in predictions
    app.text_input(key='prediction_customer_search').set_value('A00').run()
    assert not app.exception
    assert app.session_state['prediction_context']['run_id'] == run_id
    assert app.date_input(key='prediction_infer_date').value == date(2011, 10, 1)
    app.sidebar.radio(key='module').set_value('数据导入与清洗').run()
    app.sidebar.radio(key='module').set_value('复购与流失预测').run()
    assert not app.exception
    assert app.session_state['prediction_context']['run_id'] == run_id
    assert app.date_input(key='prediction_infer_date').value == date(2011, 10, 1)
    next(button for button in app.button if button.label == '加载已保存模型').click().run(timeout=30)
    assert not app.exception
    assert app.session_state['prediction_context']['run_id'] == run_id
    assert 'prediction_inference' not in app.session_state
    prepare_file(source_file(tmp_path), root=tmp_path)
    app.run()
    assert not app.exception
    assert 'prediction_result' not in app.session_state


def test_insufficient_history_reports_error_without_creating_model(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file
    from test_preparation import source_file

    prepare_file(source_file(tmp_path), root=tmp_path)
    app = open_prediction(tmp_path, monkeypatch)
    next(button for button in app.button if button.label == '构建样本并训练').click().run(timeout=30)
    assert not app.exception
    assert app.error
    assert not list((tmp_path / 'artifacts' / 'prediction').glob('*/metadata.json'))
