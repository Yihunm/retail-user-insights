from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from test_preparation import source_file


APP = Path(__file__).resolve().parents[1] / 'app.py'


def test_sales_page_empty_workspace_has_import_guidance(tmp_path, monkeypatch):
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio(key='module').set_value('交易统计与可视化').run()
    assert not app.exception
    assert app.title[0].value == '交易统计与可视化'
    assert any('导入' in message.value for message in app.info)


def test_sales_page_filters_and_exports_current_scope(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    prepare_file(source_file(tmp_path), root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio(key='module').set_value('交易统计与可视化').run()
    assert not app.exception
    assert app.metric[0].value == '£6.10'
    assert app.metric[1].value == '2'
    assert app.metric[2].value == '1'
    app.sidebar.radio(key='module').set_value('数据导入与清洗').run()
    app.sidebar.radio(key='module').set_value('交易统计与可视化').run()
    assert not app.exception
    assert app.metric[0].value == '£6.10'
    app.multiselect(key='sales_countries').set_value([]).run()
    assert not app.exception
    assert any('没有' in message.value for message in app.info)


def test_return_only_country_is_selectable_and_not_hidden(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    source = source_file(tmp_path)
    frame = pd.read_csv(source)
    frame.loc[frame.InvoiceNo == 'C3', 'Country'] = 'France'
    frame.to_csv(source, index=False)
    prepare_file(source, root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio(key='module').set_value('交易统计与可视化').run()
    assert 'France' in app.multiselect(key='sales_countries').options
    assert any('-2.55' in message.value for message in app.caption)
    app.multiselect(key='sales_countries').set_value(['France']).run()
    assert not app.exception
    assert any('没有' in message.value for message in app.info)
    assert any('-2.55' in message.value for message in app.caption)
