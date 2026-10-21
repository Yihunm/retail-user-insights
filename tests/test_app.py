from pathlib import Path

from streamlit.testing.v1 import AppTest

from test_preparation import source_file


APP = Path(__file__).resolve().parents[1] / 'app.py'


def test_empty_workspace_explains_import_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(tmp_path / 'empty'))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == '数据导入与质量检查'
    assert any('尚未导入' in entry.value for entry in app.info)


def test_existing_dataset_shows_balanced_audit_and_previews(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    root = tmp_path / 'workspace'
    prepare_file(source_file(tmp_path), root=root)
    monkeypatch.setenv('RETAIL_DATA_ROOT', str(root))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert any('核对通过' in entry.value for entry in app.success)
    assert app.metric[0].value == '3'
    assert len(app.dataframe) >= 3
