import json
from datetime import date

import pandas as pd
import pytest

from retail_insights.analytics import filter_transactions, sales_summary, sales_tables
from retail_insights.data import COLUMNS, clean_transactions


def sales_frame():
    rows = [
        ['1', 'A123', 'Z label', 2, '2011-01-01 10:00', .335, '10', 'UK'],
        ['1', 'B123', 'Second', 1, '2011-01-01 10:00', 2, '10', 'UK'],
        ['2', 'A123', 'A label', 1, '2011-01-31 23:59:59', 3, '10', 'UK'],
        ['3', 'A123', 'A label', 3, '2011-03-01 15:00', 1, '20', 'France'],
        ['4', 'B123', 'Second', 1, '2011-03-02 16:00', 4, None, 'UK'],
    ]
    return clean_transactions(pd.DataFrame(rows, columns=COLUMNS))['sales']


def test_summary_distinct_orders_anonymous_sales_and_repeat_buyers():
    summary = sales_summary(sales_frame())
    assert summary == dict(revenue=12.67, orders=4, customers=2, items=8,
                           aov=12.67 / 4, repeat_rate=.5)
    json.dumps(summary, allow_nan=False)


def test_filter_includes_entire_end_day_and_repeat_is_within_selection():
    sales = sales_frame()
    original = sales.copy(deep=True)
    selected = filter_transactions(sales, start=date(2011, 1, 31), end='2011-01-31')
    assert selected.InvoiceNo.tolist() == ['2']
    assert sales_summary(selected)['repeat_rate'] == 0
    pd.testing.assert_frame_equal(sales, original)


def test_country_selection_none_all_and_empty_none():
    sales = sales_frame()
    assert len(filter_transactions(sales)) == 5
    assert filter_transactions(sales, countries=[]).empty
    assert filter_transactions(sales, countries=['France']).InvoiceNo.tolist() == ['3']


@pytest.mark.parametrize('start,end', [
    ('2011-02-01', '2011-01-01'), ('2011', None), (None, '2011-01'),
    ('bad', None), ('2011-02-30', None), (1, None), ('2011-01-01Z', None),
])
def test_filter_rejects_reversed_or_invalid_or_partial_dates(start, end):
    with pytest.raises(ValueError):
        filter_transactions(sales_frame(), start=start, end=end)


def test_empty_outputs_have_stable_schemas_and_undefined_ratios():
    empty = sales_frame().iloc[:0]
    assert sales_summary(empty) == dict(revenue=0, orders=0, customers=0, items=0,
                                       aov=None, repeat_rate=None)
    schemas = {
        'monthly': ['Month', 'Revenue', 'Orders'],
        'products': ['StockCode', 'Description', 'Revenue', 'Quantity'],
        'countries': ['Country', 'Revenue', 'Orders'],
        'hour_week': ['Weekday', 'Hour', 'Orders'],
        'customer_spend': ['CustomerID', 'Revenue', 'Orders'],
    }
    for name, table in sales_tables(empty).items():
        assert list(table.columns) == schemas[name]
        assert table.empty


def test_tables_fill_missing_month_and_count_distinct_orders_per_group():
    tables = sales_tables(sales_frame())
    assert tables['monthly'].to_dict('records') == [
        dict(Month='2011-01', Revenue=5.67, Orders=2),
        dict(Month='2011-02', Revenue=0, Orders=0),
        dict(Month='2011-03', Revenue=7, Orders=2),
    ]
    assert tables['products'].to_dict('records') == [
        dict(StockCode='A123', Description='A label', Revenue=6.67, Quantity=6),
        dict(StockCode='B123', Description='Second', Revenue=6, Quantity=2),
    ]
    assert tables['countries'].to_dict('records') == [
        dict(Country='UK', Revenue=9.67, Orders=3),
        dict(Country='France', Revenue=3, Orders=1),
    ]
    assert tables['customer_spend'].to_dict('records') == [
        dict(CustomerID='10', Revenue=5.67, Orders=2),
        dict(CustomerID='20', Revenue=3, Orders=1),
    ]
    hour = tables['hour_week'].set_index(['Weekday', 'Hour'])
    assert hour.loc[(5, 10), 'Orders'] == 1
    assert hour.Orders.sum() == 4


def test_product_description_ties_are_lexical_and_no_identified_ratio():
    sales = sales_frame().iloc[:2].copy()
    sales['StockCode'] = 'A123'
    sales['Description'] = ['Z label', 'A label']
    sales['CustomerID'] = pd.NA
    assert sales_tables(sales)['products'].iloc[0].Description == 'A label'
    assert sales_tables(sales)['customer_spend'].empty
    assert sales_summary(sales)['repeat_rate'] is None


def test_money_sum_uses_python_integers_before_currency_conversion():
    sales = sales_frame().iloc[:2].copy()
    sales['AmountCents'] = pd.array([5_000_000_000_000_000_000] * 2, dtype='Int64')
    sales['StockCode'] = 'A123'
    expected = 100_000_000_000_000_000
    assert sales_summary(sales)['revenue'] == expected
    tables = sales_tables(sales)
    for name in ['monthly', 'products', 'countries', 'customer_spend']:
        assert tables[name].iloc[0].Revenue == expected


def test_orders_crossing_bins_are_distinct_within_each_bin():
    sales = sales_frame().iloc[:2].copy()
    sales.loc[sales.index[1], 'Country'] = 'France'
    sales.loc[sales.index[1], 'InvoiceDate'] = pd.Timestamp('2011-03-01 11:00')
    assert sales_summary(sales)['orders'] == 1
    tables = sales_tables(sales)
    assert tables['countries'].Orders.sum() == 2
    assert tables['monthly'].Orders.sum() == 2
    assert tables['hour_week'].Orders.sum() == 2


def test_quantity_is_not_truncated_when_clean_source_contains_fractional_units():
    sales = sales_frame().iloc[:2].copy()
    sales['Quantity'] = [1.5, 2.25]
    assert sales_summary(sales)['items'] == 3.75
    assert sales_tables(sales)['products'].Quantity.sum() == 3.75
