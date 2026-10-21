"""Online Retail 导入与可核对的行级清洗。

退货、销售、隔离互斥；重复行独立计数。问题计数允许重叠，不能相加。
"""

import csv
import io
import warnings
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
DEFAULT_SPECIAL_CODES = frozenset({'POST', 'D', 'DOT', 'M', 'BANK CHARGES', 'C2', 'AMAZONFEE', 'CRUK'})


def _validate_schema(frame):
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError('缺少必需列：' + ', '.join(missing))
    if frame.columns.duplicated().any():
        raise ValueError('文件包含重复列名')


def _csv_text(source):
    content = source.read() if hasattr(source, 'read') else Path(source).read_bytes()
    return content.decode('utf-8-sig') if isinstance(content, bytes) else content.removeprefix('\ufeff')


def _validate_csv(text):
    # csv.reader counts logical records, including correctly quoted embedded newlines.
    lines = text.splitlines(keepends=True)
    reader = csv.reader(lines, strict=True)
    header = None
    previous_line = 0
    try:
        for record in reader:
            raw_record = ''.join(lines[previous_line:reader.line_num])
            previous_line = reader.line_num
            if not raw_record.strip():
                continue
            if header is None:
                header = record
                _validate_schema(pd.DataFrame(columns=header))
            elif len(record) != len(header):
                raise ValueError(f'CSV 第 {reader.line_num} 行字段数为 {len(record)}，表头为 {len(header)}')
    except csv.Error as exc:
        raise ValueError(f'CSV 格式无效（第 {reader.line_num} 行）：{exc}') from exc
    if header is None:
        raise ValueError('CSV 文件为空，缺少表头')


def read_transactions(source, filename=None, sheets=None):
    """读取 CSV 或 XLSX；默认 Excel 首张表，可显式选择多张同构表。

    sheets 接受工作表名称/序号或其列表。额外列保留；多表必须列名、顺序一致。
    文件对象的文件名可通过 filename 提供。
    """
    name = filename or getattr(source, 'name', None)
    if name is None and isinstance(source, (str, Path)):
        name = str(source)
    extension = Path(str(name or '')).suffix.lower()
    if hasattr(source, 'seek'):
        source.seek(0)
    if extension == '.csv':
        if sheets is not None:
            raise ValueError('CSV 不支持工作表选择')
        text = _csv_text(source)
        _validate_csv(text)
        frame = pd.read_csv(io.StringIO(text), dtype={'InvoiceNo': 'string', 'StockCode': 'string', 'CustomerID': 'string'})
        _validate_schema(frame)
        return frame
    if extension != '.xlsx':
        raise ValueError('仅支持 .csv 或 .xlsx 文件')
    selection = 0 if sheets is None else sheets
    if isinstance(selection, (list, tuple)) and not selection:
        raise ValueError('请至少选择一个工作表')
    selection = list(selection) if isinstance(selection, (list, tuple)) else [selection]
    try:
        # ExcelFile holds one read-only openpyxl workbook. Inspect just the first
        # row of selected sheets before pandas can rename duplicate headers.
        with pd.ExcelFile(source, engine='openpyxl') as workbook:
            selected = []
            for sheet in selection:
                if isinstance(sheet, int):
                    if not 0 <= sheet < len(workbook.sheet_names):
                        raise ValueError(f'工作表序号不存在：{sheet}')
                    sheet = workbook.sheet_names[sheet]
                if sheet not in workbook.sheet_names:
                    raise ValueError(f'工作表不存在：{sheet}')
                if sheet not in selected:
                    selected.append(sheet)
            headers = []
            for sheet in selected:
                header = list(next(workbook.book[sheet].iter_rows(min_row=1, max_row=1, values_only=True), ()))
                _validate_schema(pd.DataFrame(columns=header))
                headers.append(header)
            if any(header != headers[0] for header in headers[1:]):
                raise ValueError('所选工作表结构不同，无法合并')
            frames = [workbook.parse(sheet_name=sheet, dtype={'InvoiceNo': 'string', 'StockCode': 'string', 'CustomerID': 'string'}) for sheet in selected]
    except Exception as exc:
        raise ValueError(f'XLSX 工作簿读取失败：{exc}') from exc
    for frame in frames:
        _validate_schema(frame)
    if any(list(frame.columns) != list(frames[0].columns) for frame in frames[1:]):
        raise ValueError('所选工作表结构不同，无法合并')
    return pd.concat(frames, ignore_index=True)


def _identifier(values):
    normalized = values.astype('string').str.strip().str.replace(r'^(\d+)\.0+$', r'\1', regex=True)
    return normalized.mask(normalized.eq(''))


def _amount_cents(quantity, price):
    try:
        value = Decimal(str(quantity)) * Decimal(str(price)) * 100
        return int(value.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, OverflowError):
        return None


def _naive_dates(values):
    def parse(value):
        # Numeric dates have no declared unit and must not become epoch nanos.
        if isinstance(value, Number):
            return pd.NaT
        try:
            # Some timezone abbreviations are silently dropped with a warning;
            # reject these rather than turning them into local naive timestamps.
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                parsed = pd.Timestamp(value)
            return pd.NaT if parsed.tzinfo is not None else parsed
        except (ValueError, TypeError, OverflowError, Warning):
            return pd.NaT

    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return pd.Series(pd.NaT, index=values.index, dtype='datetime64[ns]')
    if pd.api.types.is_datetime64_dtype(values.dtype):
        return pd.to_datetime(values, errors='coerce').astype('datetime64[ns]')
    # Caching by unique value keeps large files with repeated timestamps fast.
    parsed = {value: parse(value) for value in values.dropna().unique()}
    return pd.to_datetime(values.map(parsed), errors='coerce').astype('datetime64[ns]')


def clean_transactions(frame, drop_duplicates=True, special_codes=None):
    """分类并返回 sales / returns / quarantine / report。

    特殊业务代码被隔离（Reason=special_code），可用 special_codes=[] 禁用。
    缺失客户保留为 pd.NA；未知描述和国家填“未知”。AmountCents 为逐行
    HALF_UP 舍入的整数分，金额汇总应先汇总此列再除以 100。
    """
    _validate_schema(frame)
    working = frame.copy().reset_index(drop=True)
    raw_rows = len(working)
    duplicate_mask = working.duplicated(subset=COLUMNS, keep='first')
    duplicate_rows = int(duplicate_mask.sum()) if drop_duplicates else 0
    if drop_duplicates:
        working = working.loc[~duplicate_mask].copy()
    for column in ['InvoiceNo', 'StockCode', 'CustomerID']:
        working[column] = _identifier(working[column])
    for column in ['Description', 'Country']:
        working[column] = working[column].astype('string').str.strip().replace('', pd.NA).fillna('未知')
    working['InvoiceDate'] = _naive_dates(working['InvoiceDate'])
    for column in ['Quantity', 'UnitPrice']:
        working[column] = pd.to_numeric(working[column], errors='coerce')
    numeric_valid = pd.Series(np.isfinite(working['Quantity']) & np.isfinite(working['UnitPrice']), index=working.index)
    cancel = working['InvoiceNo'].str.upper().str.startswith('C', na=False)
    codes = DEFAULT_SPECIAL_CODES if special_codes is None else frozenset(str(code).strip().upper() for code in special_codes)
    special = working['StockCode'].str.upper().isin(codes)
    issues = {
        'invalid_numeric': ~numeric_valid,
        'invalid_date': working['InvoiceDate'].isna(),
        'missing_invoice': working['InvoiceNo'].isna(),
        'missing_stock_code': working['StockCode'].isna(),
        'nonpositive_price': working['UnitPrice'].le(0),
        'zero_quantity': working['Quantity'].eq(0),
        'contradictory_cancellation': cancel & working['Quantity'].gt(0),
        'special_code': special,
    }
    cents = [_amount_cents(q, p) if valid else None for q, p, valid in zip(working.Quantity, working.UnitPrice, numeric_valid)]
    # Nullable integers allow invalid rows to remain inspectable without NaN serialization.
    within_range = pd.Series([value is not None and -(2**63) < value < 2**63 for value in cents], index=working.index)
    issues['invalid_amount'] = ~within_range
    working['AmountCents'] = pd.array([value if valid else None for value, valid in zip(cents, within_range)], dtype='Int64')
    working['Amount'] = working['AmountCents'].astype('Float64') / 100
    issue_frame = pd.DataFrame(issues).fillna(False)
    invalid = issue_frame.any(axis=1)
    returned = ~invalid & (cancel | working['Quantity'].lt(0))
    sold = ~invalid & ~returned & working['Quantity'].gt(0)
    quarantine = working.loc[invalid].copy()
    quarantine['Reason'] = pd.Series(
        [';'.join(issue_frame.columns[flags]) for flags in issue_frame.loc[invalid].to_numpy(dtype=bool)],
        index=quarantine.index, dtype='string',
    )
    sales = working.loc[sold].copy().reset_index(drop=True)
    returns = working.loc[returned].copy().reset_index(drop=True)
    quarantine = quarantine.reset_index(drop=True)
    report = {
        'raw_rows': raw_rows,
        'duplicate_rows': duplicate_rows,
        'sales_rows': len(sales),
        'return_rows': len(returns),
        'quarantine_rows': len(quarantine),
        'special_code_rows': int(special.sum()),
        'missing_customer_rows': int(working.CustomerID.isna().sum()),
        'issue_counts': {name: int(mask.sum()) for name, mask in issues.items()},
        'issue_counts_overlap': True,
        'special_codes': sorted(codes),
        'drop_duplicates': bool(drop_duplicates),
        'sales_amount_cents': sum(int(value) for value in sales.AmountCents),
        'return_amount_cents': sum(int(value) for value in returns.AmountCents),
    }
    report['partition_verified'] = raw_rows == duplicate_rows + len(sales) + len(returns) + len(quarantine)
    return {'sales': sales, 'returns': returns, 'quarantine': quarantine, 'report': report}
