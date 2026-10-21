"""Versioned imports. Only a completely written dataset becomes active."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .data import clean_transactions, read_transactions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANING_VERSION = '2'


def _json_write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _activate(root: Path, metadata: dict) -> None:
    """Register the completed import, then atomically switch the active pointer."""
    data_dir = root / 'data'
    with closing(sqlite3.connect(data_dir / 'catalog.sqlite3')) as db:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS datasets '
                       '(id TEXT PRIMARY KEY, imported_at TEXT, source TEXT, metadata TEXT)')
            db.execute('INSERT OR REPLACE INTO datasets VALUES (?,?,?,?)', (
                metadata['dataset_id'], datetime.now(timezone.utc).isoformat(),
                metadata['source_name'], json.dumps(metadata, ensure_ascii=False)))
    pointer = data_dir / f'.active-{uuid4().hex}.json'
    _json_write(pointer, {'dataset_id': metadata['dataset_id']})
    pointer.replace(data_dir / 'active.json')


def prepare_file(input_path, root=None, drop_duplicates=True) -> dict:
    root = Path(root or PROJECT_ROOT).resolve()
    source = Path(input_path).resolve()
    if not source.is_file():
        raise ValueError('找不到数据文件，请检查路径。')
    if source.suffix.lower() not in {'.xlsx', '.csv'}:
        raise ValueError('仅支持 .xlsx 和 .csv 文件。')
    with source.open('rb') as handle:
        file_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    config = {'drop_duplicates': bool(drop_duplicates), 'cleaning_version': CLEANING_VERSION}
    dataset_id = hashlib.sha256((file_hash + json.dumps(config, sort_keys=True)).encode()).hexdigest()
    target = root / 'data' / 'processed' / dataset_id
    if (target / 'metadata.json').exists():
        metadata = json.loads((target / 'metadata.json').read_text(encoding='utf-8'))
        # Reuse only a complete readable artifact, never partially written output.
        load_dataset(root, dataset_id)
        _activate(root, metadata)
        return {**metadata, 'artifact_dir': str(target), 'reused': True}

    raw = read_transactions(source)
    result = clean_transactions(raw, drop_duplicates=drop_duplicates)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f'.import-{uuid4().hex}'
    stage.mkdir()
    try:
        for name in ['sales', 'returns', 'quarantine']:
            result[name].to_parquet(stage / f'{name}.parquet', index=False)
        _json_write(stage / 'quality.json', result['report'])
        metadata = {
            'dataset_id': dataset_id, 'source_sha256': file_hash, 'source_name': source.name,
            'created_at': datetime.now(timezone.utc).isoformat(), 'config': config,
            'rows': len(raw), 'report': result['report'],
        }
        raw_dir = root / 'data' / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        preserved = raw_dir / f'{file_hash}{source.suffix.lower()}'
        if not preserved.exists():
            shutil.copyfile(source, preserved)
        _json_write(stage / 'metadata.json', metadata)
        # Windows can briefly deny a directory rename just after Parquet writes.
        # Retry only permission failures, finitely; never activate a failed import.
        for attempt in range(5):
            try:
                stage.rename(target)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1 * 2**attempt)
        _activate(root, metadata)
    finally:
        # Only our freshly generated staging directory may be removed; no user paths.
        if stage.exists():
            shutil.rmtree(stage)
    return {**metadata, 'artifact_dir': str(target), 'reused': False}


def load_dataset(root=None, dataset_id=None) -> dict:
    root = Path(root or PROJECT_ROOT).resolve()
    if dataset_id is None:
        pointer = root / 'data' / 'active.json'
        if not pointer.exists():
            raise ValueError('尚未导入数据，请先选择 Excel 或 CSV 文件。')
        dataset_id = json.loads(pointer.read_text(encoding='utf-8'))['dataset_id']
    if not isinstance(dataset_id, str) or not re.fullmatch(r'[a-f0-9]{64}', dataset_id):
        raise ValueError('无效的数据版本编号。')
    folder = root / 'data' / 'processed' / dataset_id
    try:
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        report = json.loads((folder / 'quality.json').read_text(encoding='utf-8'))
        result = {name: pd.read_parquet(folder / f'{name}.parquet')
                  for name in ['sales', 'returns', 'quarantine']}
    except (OSError, ValueError) as error:
        raise ValueError('数据产物缺失或损坏，请重新导入。') from error
    return {**result, 'metadata': metadata, 'report': report}
