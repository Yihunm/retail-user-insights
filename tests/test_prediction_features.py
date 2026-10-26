from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from retail_insights.prediction_features import (
    DEFAULT_CUTOFFS, FEATURES, FEATURE_VERSION, feature_frame, make_samples,
)


EXPECTED_FEATURES = ['recency_days', 'orders_7', 'orders_30', 'orders_90', 'revenue_7', 'revenue_30',
                     'revenue_90', 'quantity_90', 'unique_products_90', 'active_days_90',
                     'average_order_value_90', 'max_order_value_90', 'interval_mean_days',
                     'interval_missing', 'history_span_days']


def sales(rows):
    frame = pd.DataFrame(rows, columns=['CustomerID', 'InvoiceNo', 'InvoiceDate', 'StockCode', 'Quantity', 'AmountCents'])
    frame['InvoiceDate'] = pd.to_datetime(frame.InvoiceDate, format='mixed')
    frame['CustomerID'] = frame.CustomerID.astype('string')
    frame['AmountCents'] = pd.Series(frame.AmountCents.tolist(), dtype=object)
    return frame


def boundary_sales():
    return sales([
        (None, 'coverage', '2010-12-01 00:00', 'X', 1, 100),
        ('a', 'old', '2010-12-01 23:59', 'OLD', 1, 900),
        ('a', 'start', '2010-12-02 00:00', 'A', .5, 100),
        ('a', 'month', '2011-01-31 00:00', 'B', 1.5, 200),
        ('a', 'week', '2011-02-23 00:00', 'C', 2.25, 300),
        ('a', 'last', '2011-03-01 23:59', 'C', 1, 400),
        ('a', 'last', '2011-03-01 23:59', 'D', .5, 500),
        ('b', 'single', '2011-03-01 12:00', 'Z', 1, 100),
        (None, 'anon', '2011-03-01 12:00', 'Z', 1, 100),
        ('a', 'label_start', '2011-03-02 00:00', 'F', 1, 100),
        ('b', 'label_end', '2011-04-01 00:00', 'F', 1, 100),
        (None, 'coverage_end', '2011-06-01 13:00', 'F', 1, 100),
    ])


def test_exact_feature_contract_boundaries_order_aggregation_and_no_mutation():
    frame = boundary_sales()
    before = frame.copy(deep=True)
    result = feature_frame(frame, '2011-03-02')
    assert FEATURES == EXPECTED_FEATURES
    assert FEATURE_VERSION == '1'
    assert result.columns.tolist() == ['CustomerID', *EXPECTED_FEATURES]
    assert result.CustomerID.tolist() == ['a', 'b']
    row = result.iloc[0]
    assert row.recency_days == 1
    assert [row.orders_7, row.orders_30, row.orders_90] == [2, 3, 4]
    assert [row.revenue_7, row.revenue_30, row.revenue_90] == [12, 14, 15]
    assert row.quantity_90 == 5.75
    assert row.unique_products_90 == 4
    assert row.active_days_90 == 4
    assert row.average_order_value_90 == 15/4
    assert row.max_order_value_90 == 9
    expected_interval = (pd.Timestamp('2011-03-01 23:59') - pd.Timestamp('2010-12-02')).total_seconds()/86400/3
    assert row.interval_mean_days == pytest.approx(expected_interval)
    assert row.interval_missing == 0
    assert row.history_span_days == 89
    assert np.isnan(result.iloc[1].interval_mean_days)
    assert result.iloc[1].interval_missing == 1
    assert result.iloc[1].history_span_days == 0
    pd.testing.assert_frame_equal(frame, before)


def test_future_changes_and_input_order_cannot_change_features():
    frame = boundary_sales()
    expected = feature_frame(frame, '2011-03-02')
    future = frame.InvoiceDate.ge(pd.Timestamp('2011-03-02'))
    frame.loc[future, 'Quantity'] = 99999
    frame.loc[future, 'AmountCents'] = 999999999
    frame.loc[future, 'CustomerID'] = 'changed'
    pd.testing.assert_frame_equal(expected, feature_frame(frame.sample(frac=1, random_state=42), '2011-03-02'))


@pytest.mark.parametrize('cutoff', ['2011-03-01', '2010-12-15'])
def test_incomplete_ninety_day_history_is_rejected(cutoff):
    # First source day was partial, so eligible coverage starts December 2.
    frame = boundary_sales()
    frame.loc[0, 'InvoiceDate'] = pd.Timestamp('2010-12-01 12:00')
    with pytest.raises(ValueError, match='90|历史|覆盖'):
        feature_frame(frame, cutoff)


def test_first_partial_day_excluded_and_fixed_march_two_cutoff_valid():
    frame = boundary_sales()
    frame.loc[0, 'InvoiceDate'] = pd.Timestamp('2010-12-01 12:00')
    assert len(feature_frame(frame, '2011-03-02')) == 2
    assert DEFAULT_CUTOFFS == ['2011-03-02', '2011-04-01', '2011-05-01',
                               '2011-06-01', '2011-08-01', '2011-10-01']


@pytest.mark.parametrize('window', [True, False, 0, 30, 90.0, 90.5, None])
def test_only_integer_ninety_day_window_is_supported(window):
    with pytest.raises(ValueError, match='90|窗口|window_days'):
        feature_frame(boundary_sales(), '2011-03-02', window_days=window)


@pytest.mark.parametrize('cutoff', ['2011-03', '2011', 1, True, pd.NaT,
                                    '2011-03-02T01:00', datetime(2011, 3, 2, tzinfo=timezone.utc)])
def test_invalid_cutoffs_are_rejected(cutoff):
    with pytest.raises(ValueError, match='截点|cutoff|日期'):
        feature_frame(boundary_sales(), cutoff)


def test_empty_source_and_future_cutoff_rejected():
    with pytest.raises(ValueError, match='空|交易|销售'):
        feature_frame(sales([]), '2011-03-02')
    with pytest.raises(ValueError, match='未来|覆盖|cutoff|截点'):
        feature_frame(boundary_sales(), '2011-06-03')


def test_purchase_and_churn_label_boundaries_and_metadata():
    import json
    frame = boundary_sales()
    purchase = make_samples(frame, ['2011-03-02'])
    churn = make_samples(frame, ['2011-03-02'], task='churn', horizon_days=30)
    assert purchase.columns.tolist() == ['CustomerID', 'Cutoff', 'Task', *FEATURES, 'Target']
    assert purchase.CustomerID.tolist() == ['a', 'b']
    assert purchase.Target.tolist() == [1, 0]
    assert churn.Target.tolist() == [0, 1]
    assert purchase.Cutoff.tolist() == ['2011-03-02']*2
    assert purchase.Task.tolist() == ['purchase']*2
    assert purchase.attrs == {'task': 'purchase', 'horizon_days': 30, 'window_days': 90,
                              'coverage_start': '2010-12-01', 'coverage_end': '2011-06-01',
                              'features': FEATURES, 'feature_version': '1', 'cutoffs': ['2011-03-02']}
    json.dumps(purchase.attrs, allow_nan=False)
    assert make_samples(frame, ['2011-03-02'], task='churn').attrs['horizon_days'] == 60


def test_label_coverage_is_complete_not_guessed_from_last_transaction():
    frame = boundary_sales()
    with pytest.raises(ValueError, match='标签|覆盖|coverage'):
        make_samples(frame, ['2011-05-03'])
    with pytest.raises(ValueError, match='标签|覆盖|coverage'):
        make_samples(frame, ['2011-03-02', '2011-05-03'])
    with pytest.raises(ValueError, match='coverage_end|覆盖'):
        make_samples(frame, ['2011-03-02'], coverage_end='2011-06-02')
    with pytest.raises(ValueError, match='标签|覆盖'):
        make_samples(frame, ['2011-03-02'], coverage_end='2011-03-31')
    assert len(make_samples(frame, ['2011-03-02'], coverage_end='2011-04-01')) == 2


@pytest.mark.parametrize('task', ['risk', None, 'PURCHASE'])
def test_invalid_task_is_rejected(task):
    with pytest.raises(ValueError, match='task|任务'):
        make_samples(boundary_sales(), ['2011-03-02'], task=task)


@pytest.mark.parametrize('horizon', [True, 0, -1, .5, '30'])
def test_invalid_horizon_is_rejected(horizon):
    with pytest.raises(ValueError, match='horizon|标签|预测|天数'):
        make_samples(boundary_sales(), ['2011-03-02'], horizon_days=horizon)


def test_duplicate_cutoff_dates_are_rejected_and_rows_unique():
    with pytest.raises(ValueError, match='重复'):
        make_samples(boundary_sales(), ['2011-03-02', datetime(2011, 3, 2, 12)])
    result = make_samples(boundary_sales(), ['2011-03-02', '2011-04-01'])
    assert not result.duplicated(['CustomerID', 'Cutoff']).any()


def test_empty_customer_cohort_retains_stable_feature_and_sample_schemas():
    frame = boundary_sales()
    frame['CustomerID'] = pd.NA
    features = feature_frame(frame, '2011-03-02')
    assert features.empty
    assert features.columns.tolist() == ['CustomerID', *FEATURES]
    samples = make_samples(frame, ['2011-03-02'])
    assert samples.empty
    assert samples.columns.tolist() == ['CustomerID', 'Cutoff', 'Task', *FEATURES, 'Target']
    assert samples.attrs['feature_version'] == '1'


def test_huge_cents_and_integer_quantities_do_not_overflow_before_conversion():
    frame = boundary_sales()
    frame.loc[frame.CustomerID.eq('a'), 'AmountCents'] = 6_000_000_000_000_000_000
    frame['Quantity'] = frame.Quantity.astype(object)
    frame.loc[frame.CustomerID.eq('a'), 'Quantity'] = 6_000_000_000_000_000_000
    row = feature_frame(frame, '2011-03-02').iloc[0]
    assert row.revenue_90 == pytest.approx(30_000_000_000_000_000_000 / 100)
    assert row.max_order_value_90 == pytest.approx(12_000_000_000_000_000_000 / 100)
    assert row.quantity_90 == pytest.approx(30_000_000_000_000_000_000)
    assert np.isfinite(row[FEATURES].drop('interval_mean_days').to_numpy(dtype=float)).all()


def test_interval_uses_distinct_order_first_times_not_line_times():
    frame = boundary_sales()
    extra = sales([('b', 'single', '2011-03-01 23:59', 'Z2', 1, 100),
                   ('b', 'second', '2011-03-01 18:00', 'Z3', 1, 100)])
    result = feature_frame(pd.concat([frame, extra], ignore_index=True), '2011-03-02').set_index('CustomerID')
    assert result.loc['b', 'orders_90'] == 2
    assert result.loc['b', 'interval_mean_days'] == .25
    assert result.loc['b', 'history_span_days'] == 0


def test_midnight_last_transaction_provides_exact_exclusive_label_boundary():
    frame = boundary_sales()
    frame.loc[frame.InvoiceNo.eq('coverage_end'), 'InvoiceDate'] = pd.Timestamp('2011-04-01')
    result = make_samples(frame, ['2011-03-02'])
    assert result.attrs['coverage_end'] == '2011-04-01'
    assert result.Target.tolist() == [1, 0]


def test_appending_future_sales_cannot_change_feature_values_or_metadata():
    frame = boundary_sales()
    before = feature_frame(frame, '2011-03-02')
    extra = sales([('a', 'far_future', '2012-01-01', 'FUTURE', 999, 999999)])
    after = feature_frame(pd.concat([frame, extra], ignore_index=True), '2011-03-02')
    pd.testing.assert_frame_equal(before, after)
    assert before.attrs == after.attrs


def test_date_object_cutoff_normalizes_and_preserves_numeric_feature_types():
    result = feature_frame(boundary_sales(), datetime(2011, 3, 2, 12))
    expected = feature_frame(boundary_sales(), '2011-03-02')
    pd.testing.assert_frame_equal(result, expected)
    assert all(pd.api.types.is_numeric_dtype(result[column]) for column in FEATURES)


def test_sample_construction_does_not_mutate_sales_or_fabricate_second_class():
    frame = boundary_sales()
    before = frame.copy(deep=True)
    frame.loc[frame.CustomerID.eq('b') & frame.InvoiceNo.eq('label_end'), 'InvoiceDate'] = pd.Timestamp('2011-03-03')
    changed = frame.copy(deep=True)
    result = make_samples(frame, ['2011-03-02'])
    assert result.Target.tolist() == [1, 1]
    pd.testing.assert_frame_equal(frame, changed)
    assert before.InvoiceDate.iloc[-2] != frame.InvoiceDate.iloc[-2]
