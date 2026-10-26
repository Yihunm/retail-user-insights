from datetime import date
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from test_preparation import source_file

APP = Path(__file__).resolve().parents[1] / 'app.py'


def open_rfm(tmp_path, monkeypatch, imported=True):
    from retail_insights.preparation import prepare_file

    if imported:
        prepare_file(source_file(tmp_path), root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.sidebar.radio(key='module').set_value('RFM用户价值分析').run(timeout=30)
    return app


def test_rfm_no_data_guides_import(tmp_path, monkeypatch):
    app = open_rfm(tmp_path, monkeypatch, imported=False)
    assert not app.exception
    assert any('导入' in item.value for item in app.info)


def test_rfm_uses_identified_sales_and_shows_chinese_tables(tmp_path, monkeypatch):
    app = open_rfm(tmp_path, monkeypatch)
    assert not app.exception
    assert app.title[0].value == 'RFM用户价值分析'
    assert app.metric[0].value == '1'
    assert app.metric[1].value == '£5.10'
    assert app.date_input(key='rfm_reference').value == date(2011, 1, 3)
    assert any('客户类型' in list(table.value.columns) for table in app.dataframe)
    assert len(app.get('plotly_chart')) == 3


def test_rfm_customer_filter_empty_segments_and_page_switch_restore(tmp_path, monkeypatch):
    app = open_rfm(tmp_path, monkeypatch)
    app.text_input(key='rfm_customer_search').set_value('not-a-customer').run()
    assert not app.exception
    assert any('没有' in item.value for item in app.info)
    app.text_input(key='rfm_customer_search').set_value('123').run()
    assert app.metric[0].value == '1'
    app.selectbox(key='rfm_window').set_value('最近90天').run()
    app.sidebar.radio(key='module').set_value('数据导入与清洗').run()
    app.sidebar.radio(key='module').set_value('RFM用户价值分析').run()
    assert not app.exception
    assert app.text_input(key='rfm_customer_search').value == '123'
    assert app.selectbox(key='rfm_window').value == '最近90天'
    app.multiselect(key='rfm_segments').set_value([]).run()
    assert any('没有' in item.value for item in app.info)


def test_reference_date_excludes_same_day_purchases(tmp_path, monkeypatch):
    app = open_rfm(tmp_path, monkeypatch)
    app.date_input(key='rfm_reference').set_value(date(2011, 1, 1)).run()
    assert not app.exception
    assert any('没有' in item.value for item in app.info)


def test_customer_filter_does_not_rescore_against_smaller_population(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    path = source_file(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[frame.InvoiceNo == '2', 'CustomerID'] = 456
    frame.to_csv(path, index=False)
    prepare_file(path, root=tmp_path)
    app = open_rfm(tmp_path, monkeypatch, imported=False)
    app.text_input(key='rfm_customer_search').set_value('123').run()
    assert not app.exception
    details = next(table.value for table in app.dataframe if '客户编号' in table.value.columns)
    assert details['M评分'].tolist() == [5]  # Would become 1 if recomputed after searching.
