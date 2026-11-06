"""用户从工作台进入业务页面，无需理解或重复运行模型。"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from test_preparation import source_file

APP = Path(__file__).resolve().parents[1] / 'app.py'


def test_first_visit_has_one_clear_import_action(tmp_path, monkeypatch):
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == '经营工作台'
    app.button(key='home_import').click().run()
    assert not app.exception
    assert app.sidebar.radio(key='module').value == '数据管理'
    assert any(expander.label == '导入数据' for expander in app.expander)


def test_home_summarizes_real_sales_and_shortcut_preserves_customer_scope(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    prepare_file(source_file(tmp_path), root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == '£6.10'
    assert app.metric[1].value == '2'
    assert app.metric[2].value == '1'
    assert not list((tmp_path / 'artifacts').glob('*/*'))
    app.button(key='home_value').click().run()
    assert not app.exception
    assert app.sidebar.radio(key='module').value == '客户价值'
    assert app.multiselect(key='rfm_segments').value == ['高价值客户']
    assert app.selectbox(key='rfm_window').value == '全部历史'


def test_business_risk_page_without_models_points_to_setup_without_training(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    prepare_file(source_file(tmp_path), root=tmp_path)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.sidebar.radio(key='module').set_value('复购与流失').run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 0
    assert not any(button.label == '构建样本并训练' for button in app.button)
    app.button(key='prediction_setup').click().run()
    assert not app.exception
    assert app.sidebar.radio(key='module').value == '模型实验'
    assert not list((tmp_path / 'artifacts').glob('*/*'))


def test_previous_navigation_session_migrates_to_business_label(tmp_path, monkeypatch):
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state['module'] = 'RFM用户价值分析'
    app.run()
    assert not app.exception
    assert app.sidebar.radio(key='module').value == '客户价值'
