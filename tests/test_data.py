import io
import json
import csv
from datetime import datetime, timezone
from zipfile import ZipFile

import pandas as pd
import pytest

from retail_insights.data import COLUMNS, clean_transactions, read_transactions


def row(invoice='1', code='A123', qty=2, price=1.25, customer=123.0, date='2011-01-01', description='item', country='UK'):
    return [invoice, code, description, qty, date, price, customer, country]


def test_clean_partitions_and_audits_without_dropping_anonymous_sales():
    rows = [row(), row(), row('2', customer=None), row('C3', qty=-2), row('C4'), row('5', price=0), row('6', code='POST'), row('7', date='bad'), row('8', qty='bad')]
    result = clean_transactions(pd.DataFrame(rows, columns=COLUMNS))
    assert [len(result[k]) for k in ['sales', 'returns', 'quarantine']] == [2, 1, 5]
    assert result['sales'].iloc[0].CustomerID == '123'
    assert pd.isna(result['sales'].iloc[1].CustomerID)
    assert result['sales'].Amount.sum() == 5
    report = result['report']
    assert report['raw_rows'] == report['duplicate_rows'] + report['sales_rows'] + report['return_rows'] + report['quarantine_rows']
    assert report['special_code_rows'] == 1
    json.dumps(report, allow_nan=False)


def test_row_rounding_missing_labels_and_provenance_dedup():
    frame = pd.DataFrame([row(price=.335, qty=1, description=None, country=None), row(price=.335, qty=1, description=None, country=None)], columns=COLUMNS)
    frame['SourceFile'] = ['a', 'b']
    result = clean_transactions(frame)
    assert len(result['sales']) == 1
    assert result['sales'].iloc[0].Amount == .34
    assert result['sales'].iloc[0].Country == '未知'
    assert result['sales'].iloc[0].SourceFile == 'a'


def test_csv_read_and_schema_validation():
    frame = pd.DataFrame([row()], columns=COLUMNS)
    actual = read_transactions(io.StringIO(frame.to_csv(index=False)), filename='orders.csv')
    assert len(actual) == 1
    with pytest.raises(ValueError, match='CustomerID'):
        read_transactions(io.StringIO(frame.drop(columns='CustomerID').to_csv(index=False)), filename='orders.csv')


def test_xlsx_combines_selected_compatible_sheets():
    stream = io.BytesIO()
    frame = pd.DataFrame([row()], columns=COLUMNS)
    with pd.ExcelWriter(stream, engine='openpyxl') as writer:
        frame.to_excel(writer, index=False, sheet_name='a')
        frame.to_excel(writer, index=False, sheet_name='b')
    stream.seek(0)
    assert len(read_transactions(stream, filename='orders.xlsx', sheets=['a', 'b'])) == 2


def test_rejects_unsupported_type_and_mismatched_sheets():
    with pytest.raises(ValueError, match='csv|xlsx'):
        read_transactions(io.BytesIO(b'bad'), filename='orders.pdf')
    stream = io.BytesIO()
    frame = pd.DataFrame([row()], columns=COLUMNS)
    with pd.ExcelWriter(stream, engine='openpyxl') as writer:
        frame.to_excel(writer, index=False, sheet_name='a')
        frame.drop(columns='CustomerID').to_excel(writer, index=False, sheet_name='b')
    stream.seek(0)
    with pytest.raises(ValueError, match='CustomerID'):
        read_transactions(stream, filename='orders.xlsx', sheets=['a', 'b'])


def test_numeric_errors_and_contradictory_cancellations_have_reasons():
    frame = pd.DataFrame([row('C1'), row('2', qty=-1), row('3', price=-1), row('4', qty=float('inf')), row('5', code='A12B'), row('6', code='POST')], columns=COLUMNS)
    result = clean_transactions(frame, special_codes=[])
    assert len(result['sales']) == 2
    assert len(result['returns']) == 1
    assert len(result['quarantine']) == 3
    assert result['quarantine'].Reason.notna().all()
    assert result['report']['issue_counts_overlap'] is True


def test_empty_input_preserves_schema_and_balanced_audit():
    result = clean_transactions(pd.DataFrame(columns=COLUMNS))
    assert result['report']['partition_verified']
    assert result['report']['raw_rows'] == 0
    assert result['sales'].empty
    assert 'Reason' in result['quarantine']


def test_disable_dedup_and_half_up_cent_rounding_does_not_mutate_input():
    original = pd.DataFrame([row(price=.145, qty=1)]*2, columns=COLUMNS)
    before = original.copy(deep=True)
    result = clean_transactions(original, drop_duplicates=False)
    assert result['sales'].AmountCents.tolist() == [15, 15]
    assert result['report']['sales_amount_cents'] == 30
    assert result['report']['duplicate_rows'] == 0
    assert result['report']['partition_verified']
    pd.testing.assert_frame_equal(original, before)


@pytest.mark.parametrize('extension', ['csv', 'xlsx'])
def test_rejects_duplicate_original_headers(extension):
    headers = COLUMNS + ['Country']
    if extension == 'csv':
        stream = io.StringIO(','.join(headers) + '\n' + ','.join(map(str, row() + ['US'])))
    else:
        stream = io.BytesIO()
        pd.DataFrame([row() + ['US']], columns=headers).to_excel(stream, index=False)
        stream.seek(0)
    with pytest.raises(ValueError, match='重复列名'):
        read_transactions(stream, filename='orders.' + extension)


@pytest.mark.parametrize('values', [row() + ['extra'], row()[:-1]])
def test_csv_rejects_wrong_field_count_before_index_can_shift(values):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(COLUMNS)
    writer.writerow(values)
    stream.seek(0)
    with pytest.raises(ValueError, match='字段数'):
        read_transactions(stream, filename='orders.csv')


def test_csv_allows_extra_columns_blank_lines_and_quoted_newlines():
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(COLUMNS + ['SourceFile'])
    writer.writerow([])
    writer.writerow(row(description='item\nwith, comma') + ['source.csv'])
    stream.seek(0)
    frame = read_transactions(stream, filename='orders.csv')
    assert len(frame) == 1
    assert frame.iloc[0].Description == 'item\nwith, comma'
    assert frame.iloc[0].SourceFile == 'source.csv'


def test_report_sums_cents_without_int64_overflow():
    frame = pd.DataFrame([row('1', qty=1, price=50_000_000_000_000_000), row('2', qty=1, price=50_000_000_000_000_000), row('C3', qty=-1, price=50_000_000_000_000_000), row('C4', qty=-1, price=50_000_000_000_000_000)], columns=COLUMNS)
    report = clean_transactions(frame)['report']
    assert report['sales_amount_cents'] == 10_000_000_000_000_000_000
    assert report['return_amount_cents'] == -10_000_000_000_000_000_000
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('date', [-1, 1, 1.5, '2011-01-01T00:00:00Z', '2011-01-01T00:00:00+08:00', '2011-01-01 12:00:00 EST', datetime(2011, 1, 1, tzinfo=timezone.utc)])
def test_numeric_and_timezone_dates_are_quarantined(date):
    frame = pd.DataFrame([row('1', date=date), row('2')], columns=COLUMNS)
    result = clean_transactions(frame)
    assert result['sales'].InvoiceNo.tolist() == ['2']
    assert result['quarantine'].Reason.tolist() == ['invalid_date']
    assert str(result['sales'].InvoiceDate.dtype) == 'datetime64[ns]'
    assert str(result['quarantine'].InvoiceDate.dtype) == 'datetime64[ns]'


def test_corrupt_workbook_is_clear_value_error():
    with pytest.raises(ValueError, match='XLSX|xlsx|工作簿'):
        read_transactions(io.BytesIO(b'broken zip archive'), filename='orders.xlsx')


def test_valid_zip_without_workbook_is_clear_value_error():
    stream = io.BytesIO()
    with ZipFile(stream, 'w') as archive:
        archive.writestr('not_a_workbook.txt', 'bad')
    stream.seek(0)
    with pytest.raises(ValueError, match='XLSX|xlsx|工作簿'):
        read_transactions(stream, filename='orders.xlsx')
