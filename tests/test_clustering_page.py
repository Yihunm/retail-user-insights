from pathlib import Path
import json

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'


def cluster_source(tmp_path):
    path = tmp_path / 'cluster-orders.csv'
    rows = []
    for i in range(12):
        for order in range(i % 4 + 1):
            rows.append([f'{i}-{order}', 'A', 'item', i % 3 + 1,
                         f'2011-01-{i + 1:02d}', (i + 1) * 2.55, f'C{i:03d}', 'UK'])
    pd.DataFrame(rows, columns=['InvoiceNo', 'StockCode', 'Description', 'Quantity',
                               'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']).to_csv(path, index=False)
    return path


def open_cluster(tmp_path, monkeypatch, imported=True):
    from retail_insights.preparation import prepare_file

    if imported:
        prepare_file(cluster_source(tmp_path), root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.sidebar.radio(key='module').set_value('用户聚类分析').run(timeout=30)
    return app


def train(app):
    next(button for button in app.button if button.label == '训练并分析').click().run(timeout=60)
    return app


def test_clustering_without_data_guides_import(tmp_path, monkeypatch):
    app = open_cluster(tmp_path, monkeypatch, imported=False)
    assert not app.exception
    assert any('导入' in info.value for info in app.info)


def test_clustering_trains_explicitly_and_customer_search_does_not_retrain(tmp_path, monkeypatch):
    app = open_cluster(tmp_path, monkeypatch)
    assert not app.exception
    assert len(app.metric) == 0
    app.selectbox(key='cluster_k_mode').set_value('3组')
    train(app)
    assert not app.exception
    assert app.metric[0].value == '12'
    assert app.metric[1].value == '3'
    assert len(app.get('plotly_chart')) >= 4
    colors = {}
    for chart in app.get('plotly_chart'):
        for trace in json.loads(chart.proto.spec)['data']:
            name = trace.get('name', '')
            if name.startswith('群体'):
                colors.setdefault(name, set()).add(trace['marker']['color'])
    assert all(len(values) == 1 for values in colors.values())
    run_id = app.session_state['cluster_context']['run_id']
    app.text_input(key='cluster_customer_search').set_value('C001').run()
    assert not app.exception
    assert app.session_state['cluster_context']['run_id'] == run_id
    details = next(table.value for table in app.dataframe if '客户编号' in table.value.columns)
    assert details['客户编号'].tolist() == ['C001']
    app.sidebar.radio(key='module').set_value('数据导入与清洗').run()
    app.sidebar.radio(key='module').set_value('用户聚类分析').run()
    assert not app.exception
    assert app.session_state['cluster_context']['run_id'] == run_id
    assert app.text_input(key='cluster_customer_search').value == 'C001'


def test_clustering_rejects_too_small_customer_population_without_traceback(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file
    from test_preparation import source_file

    prepare_file(source_file(tmp_path), root=tmp_path)
    app = open_cluster(tmp_path, monkeypatch, imported=False)
    train(app)
    assert not app.exception
    assert len(app.error) + len(app.info) > 0
    assert not (tmp_path / 'artifacts' / 'clustering').exists()


def test_new_dataset_invalidates_previous_cluster_result(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file
    from test_preparation import source_file

    app = train(open_cluster(tmp_path, monkeypatch))
    assert not app.exception
    prepare_file(source_file(tmp_path), root=tmp_path)
    app.run()
    assert not app.exception
    assert len(app.metric) == 0
    assert 'cluster_result' not in app.session_state
