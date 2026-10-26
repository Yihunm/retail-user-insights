from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from retail_insights.rfm import build_rfm, rfm_profile, rfm_rules


RFM_COLUMNS = ['CustomerID', 'LastPurchase', 'Recency', 'Frequency', 'MonetaryCents', 'Monetary',
               'Quantity', 'AverageOrderValue', 'RScore', 'FScore', 'MScore', 'Segment']
SEGMENTS = ['高价值客户', '流失风险客户', '沉睡客户', '潜力客户', '一般客户']


def sales(rows):
    frame = pd.DataFrame(rows, columns=['CustomerID', 'InvoiceNo', 'InvoiceDate', 'AmountCents', 'Quantity'])
    frame['InvoiceDate'] = pd.to_datetime(frame.InvoiceDate)
    frame['CustomerID'] = frame.CustomerID.astype('string')
    frame['AmountCents'] = frame.AmountCents.astype('Int64')
    return frame


def test_window_includes_start_excludes_reference_and_future_and_counts_orders():
    frame = sales([
        ('a', 'old', '2011-01-04 23:59', 900, 9),
        ('a', 'start', '2011-01-05 00:00', 100, .5),
        ('a', 'start', '2011-01-05 01:00', 200, 1.5),
        ('a', 'last', '2011-01-09 23:59', 300, 2.25),
        ('a', 'cutoff', '2011-01-10 00:00', 800, 8),
        ('b', 'future', '2011-01-11 00:00', 900, 9),
        (None, 'anonymous', '2011-01-09 00:00', 900, 9),
    ])
    before = frame.copy(deep=True)
    result = build_rfm(frame, reference_date='2011-01-10', window_days=5)
    customer = result.iloc[0]
    assert result.CustomerID.tolist() == ['a']
    assert customer.LastPurchase == pd.Timestamp('2011-01-09 23:59')
    assert customer.Recency == 1
    assert customer.Frequency == 2
    assert customer.MonetaryCents == 600
    assert customer.Monetary == 6
    assert customer.Quantity == 4.25
    assert customer.AverageOrderValue == 3
    assert result.attrs['reference_date'] == '2011-01-10'
    assert result.attrs['window_days'] == 5
    pd.testing.assert_frame_equal(frame, before)
    without_future = frame.iloc[[0, 1, 2, 3, 6]].copy()
    pd.testing.assert_frame_equal(result, build_rfm(without_future, '2011-01-10', 5))


def test_default_reference_uses_anonymous_sales_last_day():
    frame = sales([('a', '1', '2011-01-01', 100, 1), (None, '2', '2011-01-10', 200, 2)])
    result = build_rfm(frame)
    assert result.attrs['reference_date'] == '2011-01-11'
    assert result.iloc[0].Recency == 10


def test_full_history_and_90_day_window_have_distinct_population_and_amounts():
    frame = sales([('a', '1', '2010-12-31', 100, 1), ('a', '2', '2011-04-01', 200, 2),
                   ('b', '3', '2010-12-31', 300, 3)])
    full = build_rfm(frame, '2011-04-02')
    recent = build_rfm(frame, '2011-04-02', 90)
    assert full.CustomerID.tolist() == ['a', 'b']
    assert full.iloc[0].MonetaryCents == 300
    assert recent.CustomerID.tolist() == ['a']
    assert recent.iloc[0].MonetaryCents == 200


@pytest.mark.parametrize('reference', ['2011-01', '2011', '1', '', '2011-02-30', '2011-01-10T12:00',
                                       '2011-01-10T00:00Z', 1, True, pd.NaT,
                                       datetime(2011, 1, 10, tzinfo=timezone.utc)])
def test_rejects_invalid_reference_date(reference):
    with pytest.raises(ValueError, match='reference_date|参考日期'):
        build_rfm(sales([]), reference_date=reference)


@pytest.mark.parametrize('reference', [date(2011, 1, 10), datetime(2011, 1, 10, 23),
                                       pd.Timestamp('2011-01-10 12:00')])
def test_naive_date_objects_normalize_to_midnight(reference):
    frame = sales([('a', '1', '2011-01-09 23:59', 100, 1), ('b', '2', '2011-01-10 00:01', 200, 2)])
    assert build_rfm(frame, reference).CustomerID.tolist() == ['a']


@pytest.mark.parametrize('window', [True, False, 0, -1, .5, 90.5, '90', float('nan')])
def test_rejects_invalid_window(window):
    with pytest.raises(ValueError, match='window_days|窗口'):
        build_rfm(sales([]), window_days=window)


@pytest.mark.parametrize('frame', [sales([]), sales([(None, '1', '2011-01-01', 100, 1)]),
                                   sales([('a', '1', '2012-01-01', 100, 1)])])
def test_empty_population_has_stable_columns_and_metadata(frame):
    result = build_rfm(frame, reference_date='2011-01-01')
    assert result.empty
    assert result.columns.tolist() == RFM_COLUMNS
    assert result.attrs['thresholds'] == {'R': [], 'F': [], 'M': []}
    assert result.attrs['reference_date'] == '2011-01-01'
    assert result.attrs['window_days'] is None
    assert result.attrs['score_method']


def test_empty_without_reference_and_all_anonymous_default_metadata():
    assert build_rfm(sales([])).attrs['reference_date'] is None
    result = build_rfm(sales([(None, '1', '2011-01-01', 100, 1)]))
    assert result.empty
    assert result.attrs['reference_date'] == '2011-01-02'


def test_quantile_boundaries_ties_direction_and_input_order_are_deterministic():
    frame = sales([(str(i), f'order{i}', f'2011-01-0{i+1}', (i+1)*100, 1) for i in range(5)])
    result = build_rfm(frame, '2011-01-06')
    assert result.MScore.tolist() == [1, 2, 3, 4, 5]
    assert result.RScore.tolist() == [1, 2, 3, 4, 5]
    assert result.FScore.tolist() == [1]*5
    assert result.attrs['thresholds']['F'] == [1., 1., 1., 1.]
    assert result.attrs['thresholds']['R'] == pytest.approx([1.8, 2.6, 3.4, 4.2])
    pd.testing.assert_frame_equal(result, build_rfm(frame.sample(frac=1, random_state=42), '2011-01-06'))
    tie = build_rfm(sales([('b', '1', '2011-01-01', 100, 1), ('a', '2', '2011-01-01', 100, 1)]))
    assert tie[['RScore', 'FScore', 'MScore']].values.tolist() == [[5, 1, 1], [5, 1, 1]]


def test_repeated_quantile_boundaries_keep_one_to_five_scale():
    frame = sales([(str(i), str(i), '2011-01-01', cents, 1)
                   for i, cents in enumerate([100]*9 + [1000])])
    result = build_rfm(frame)
    assert result.MScore.tolist() == [1]*9 + [5]


def test_segments_cover_population_with_ordered_rules():
    rows = []
    frequencies = [10, 1, 2, 3, 4, 5, 6, 7, 1, 8]
    for i, frequency in enumerate(frequencies):
        for order in range(frequency):
            rows.append((str(i), f'{i}-{order}', pd.Timestamp('2011-01-11') - timedelta(days=i+1), frequency*100, 1))
    result = build_rfm(sales(rows), '2011-01-11').set_index('CustomerID')
    assert result.loc['0', 'Segment'] == '高价值客户'
    assert result.loc['1', 'Segment'] == '潜力客户'
    assert result.loc['4', 'Segment'] == '一般客户'
    assert result.loc['8', 'Segment'] == '沉睡客户'
    assert result.loc['9', 'Segment'] == '流失风险客户'
    assert set(result.Segment) == set(SEGMENTS)
    rules = rfm_rules()
    assert rules.columns.tolist() == ['客户类型', '判定条件', '运营建议']
    assert rules['客户类型'].tolist() == SEGMENTS
    assert rules['判定条件'].str.contains('分数').all()
    assert rules['运营建议'].str.len().gt(0).all()


def test_large_cent_totals_profile_and_zero_groups_are_exact():
    frame = sales([('a', '1', '2011-01-01', 5_000_000_000_000_000_000, 1),
                   ('a', '2', '2011-01-02', 5_000_000_000_000_000_000, 1),
                   ('b', '3', '2011-01-02', 100, .5)])
    result = build_rfm(frame)
    assert result.iloc[0].MonetaryCents == 10_000_000_000_000_000_000
    assert isinstance(result.iloc[0].MonetaryCents, int)
    profile = rfm_profile(result)
    assert profile.columns.tolist() == ['Segment', 'Count', 'MonetaryCents', 'Monetary', 'CustomerShare', 'RevenueShare']
    assert profile.Segment.tolist() == SEGMENTS
    assert sum(profile.MonetaryCents) == 10_000_000_000_000_000_100
    assert sum(profile.Count) == 2
    assert profile.CustomerShare.sum() == pytest.approx(1)
    assert profile.RevenueShare.sum() == pytest.approx(1)
    assert (profile.loc[profile.Count.eq(0), 'MonetaryCents'] == 0).all()


def test_empty_profile_and_zero_revenue_have_undefined_shares():
    empty = rfm_profile(build_rfm(sales([])))
    assert empty.Count.tolist() == [0]*5
    assert empty.CustomerShare.tolist() == [None]*5
    assert empty.RevenueShare.tolist() == [None]*5
    frame = pd.DataFrame({'Segment': ['一般客户'], 'MonetaryCents': [0]})
    profile = rfm_profile(frame)
    assert profile.CustomerShare.tolist() == [0., 0., 0., 0., 1.]
    assert profile.RevenueShare.tolist() == [None]*5


def test_numpy_integer_window_is_supported():
    result = build_rfm(sales([]), window_days=np.int64(90))
    assert result.attrs['window_days'] == 90
    assert type(result.attrs['window_days']) is int


def test_monetary_scores_compare_exact_cents_even_beyond_float_precision():
    frame = sales([(str(i), str(i), '2011-01-01', 9_000_000_000_000_000_000+i, 1)
                   for i in range(5)])
    result = build_rfm(frame)
    assert result.MScore.tolist() == [1, 2, 3, 4, 5]
    assert result.attrs['thresholds_exact'] == {
        'R': ['1']*4, 'F': ['1']*4,
        'M': ['45000000000000000004/5', '45000000000000000008/5',
              '45000000000000000012/5', '45000000000000000016/5'],
    }
    assert all(isinstance(value, float) for value in result.attrs['thresholds']['M'])
    assert '整数分' in result.attrs['score_method']
    assert '展示' in result.attrs['score_method']
    assert build_rfm(sales([])).attrs['thresholds_exact'] == {'R': [], 'F': [], 'M': []}


def test_integer_quantity_sum_exceeds_int64_without_losing_fraction_support():
    frame = sales([('a', '1', '2011-01-01', 600, 6_000_000_000_000_000_000),
                   ('a', '2', '2011-01-02', 600, 6_000_000_000_000_000_000)])
    result = build_rfm(frame)
    assert result.iloc[0].Quantity == 12_000_000_000_000_000_000
    assert isinstance(result.iloc[0].Quantity, int)
    assert result.Quantity.dtype == object
    fractions = build_rfm(sales([('a', '1', '2011-01-01', 100, .5),
                                 ('a', '2', '2011-01-02', 100, .25)]))
    assert fractions.iloc[0].Quantity == .75
