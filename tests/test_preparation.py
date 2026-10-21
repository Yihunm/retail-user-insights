import json
from pathlib import Path

import pandas as pd
import pytest


def source_file(tmp_path):
    path = tmp_path / 'orders.csv'
    pd.DataFrame([
        ['1', 'A', '商品', 2, '2011-01-01', 2.55, '123', 'UK'],
        ['2', 'B', '商品B', 1, '2011-01-02', 1, None, 'UK'],
        ['C3', 'A', '商品', -1, '2011-01-03', 2.55, '123', 'UK'],
    ], columns=['InvoiceNo', 'StockCode', 'Description', 'Quantity',
                'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']).to_csv(path, index=False)
    return path


def test_prepare_persists_reloadable_partition_and_reuses_same_version(tmp_path):
    from retail_insights.preparation import load_dataset, prepare_file

    source = source_file(tmp_path)
    first = prepare_file(source, root=tmp_path / 'workspace')
    second = prepare_file(source, root=tmp_path / 'workspace')
    assert first['dataset_id'] == second['dataset_id']
    assert second['reused'] is True
    loaded = load_dataset(tmp_path / 'workspace')
    assert len(loaded['sales']) == 2
    assert len(loaded['returns']) == 1
    assert loaded['sales']['Amount'].sum() == pytest.approx(6.1)
    assert Path(first['artifact_dir'], 'quality.json').is_file()
    json.dumps(loaded['report'], allow_nan=False)


def test_cleaning_configuration_creates_distinct_versions_and_path_is_validated(tmp_path):
    from retail_insights.preparation import load_dataset, prepare_file

    source = source_file(tmp_path)
    one = prepare_file(source, root=tmp_path / 'workspace', drop_duplicates=True)
    two = prepare_file(source, root=tmp_path / 'workspace', drop_duplicates=False)
    assert one['dataset_id'] != two['dataset_id']
    with pytest.raises(ValueError):
        load_dataset(tmp_path / 'workspace', dataset_id='../elsewhere')


def test_failed_import_does_not_replace_active_dataset(tmp_path):
    from retail_insights.preparation import load_dataset, prepare_file

    source = source_file(tmp_path)
    root = tmp_path / 'workspace'
    first = prepare_file(source, root=root)
    bad = tmp_path / 'bad.csv'
    bad.write_text('bad\n1\n', encoding='utf-8')
    with pytest.raises(ValueError):
        prepare_file(bad, root=root)
    assert load_dataset(root)['metadata']['dataset_id'] == first['dataset_id']


def test_catalog_handles_are_closed_after_import(tmp_path):
    from retail_insights.preparation import prepare_file

    root = tmp_path / 'workspace'
    prepare_file(source_file(tmp_path), root=root)
    catalog = root / 'data' / 'catalog.sqlite3'
    backup = catalog.with_suffix('.backup')
    # Windows refuses rename when sqlite connections still hold the file open.
    catalog.rename(backup)
    assert backup.exists()


def test_import_retries_temporary_windows_directory_lock(tmp_path, monkeypatch):
    from retail_insights.preparation import prepare_file

    original = Path.rename
    attempts = []

    def locked_once(path, target):
        if path.name.startswith('.import-'):
            attempts.append(path)
            if len(attempts) == 1:
                raise PermissionError(13, 'temporary Windows file lock')
        return original(path, target)

    monkeypatch.setattr(Path, 'rename', locked_once)
    result = prepare_file(source_file(tmp_path), root=tmp_path / 'workspace')
    assert result['rows'] == 3
    assert len(attempts) == 2
