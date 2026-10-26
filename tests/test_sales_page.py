from pathlib import Path
from datetime import date
import json

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
    app.button(key='sales_countries_clear').click().run()
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
    assert app.checkbox(key='sales_country_choice_France').label == '法国'
    assert any('-2.55' in message.value for message in app.caption)
    app.checkbox(key='sales_country_choice_UK').uncheck().run()
    assert not app.exception
    assert any('没有' in message.value for message in app.info)
    assert any('-2.55' in message.value for message in app.caption)


def test_separate_dates_include_end_day_and_reject_reversed_range(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    prepare_file(source_file(tmp_path), root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio(key='module').set_value('交易统计与可视化').run()
    assert app.date_input(key='sales_start').label == '开始日期'
    assert app.date_input(key='sales_end').label == '结束日期'
    app.date_input(key='sales_end').set_value(date(2011, 1, 1)).run()
    assert not app.exception
    assert app.metric[0].value == '£5.10'
    app.date_input(key='sales_start').set_value(date(2011, 1, 2)).run()
    assert not app.exception
    assert any('开始日期' in error.value for error in app.error)


def test_all_countries_are_chinese_search_keeps_selection_and_all_buttons_work(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    source = source_file(tmp_path)
    frame = pd.read_csv(source)
    frame.loc[frame.InvoiceNo == '2', 'Country'] = 'France'
    frame.to_csv(source, index=False)
    prepare_file(source, root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio(key='module').set_value('交易统计与可视化').run()
    assert app.checkbox(key='sales_country_choice_UK').label == '英国'
    assert app.checkbox(key='sales_country_choice_France').label == '法国'
    app.text_input(key='sales_country_search').set_value('法国').run()
    app.checkbox(key='sales_country_choice_France').uncheck().run()
    assert app.metric[0].value == '£5.10'  # Search must not clear hidden UK.
    app.text_input(key='sales_country_search').set_value('').run()
    assert app.checkbox(key='sales_country_choice_UK').value is True
    assert app.checkbox(key='sales_country_choice_France').value is False
    app.button(key='sales_countries_clear').click().run()
    assert any('没有' in info.value for info in app.info)
    app.button(key='sales_countries_all').click().run()
    assert not app.exception
    assert app.metric[0].value == '£6.10'


def test_charts_and_display_tables_have_chinese_labels(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    prepare_file(source_file(tmp_path), root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio(key='module').set_value('交易统计与可视化').run()
    assert not app.exception
    charts = app.get('plotly_chart')
    assert len(charts) == 5
    for chart in charts:
        spec = json.loads(chart.proto.spec)
        for axis in ('xaxis', 'yaxis'):
            title = spec['layout'].get(axis, {}).get('title', {}).get('text', '')
            assert title not in {'Month', 'Revenue', 'StockCode', 'Country', 'count', 'variable'}
        spec_text = chart.proto.spec
        assert 'Revenue=' not in spec_text and 'Description=' not in spec_text
    assert list(app.dataframe[0].value.columns) == ['月份', '销售额（英镑）', '订单数']


def test_previous_filter_session_is_upgraded_without_missing_state(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    metadata = prepare_file(source_file(tmp_path), root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run()
    app.session_state['sales_version'] = metadata['dataset_id']
    app.session_state['sales_dates'] = (date(2011, 1, 1), date(2011, 1, 3))
    app.session_state['sales_countries'] = ['UK']
    app.sidebar.radio(key='module').set_value('交易统计与可视化').run()
    assert not app.exception
    assert app.metric[0].value == '£6.10'
