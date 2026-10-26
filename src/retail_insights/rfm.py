"""有明确时间截点的客户 RFM 指标、分数与固定规则画像。"""

import re
from datetime import date, datetime, timedelta
from fractions import Fraction
from numbers import Integral

import numpy as np
import pandas as pd

RFM_COLUMNS = ['CustomerID', 'LastPurchase', 'Recency', 'Frequency', 'MonetaryCents', 'Monetary',
               'Quantity', 'AverageOrderValue', 'RScore', 'FScore', 'MScore', 'Segment']
SEGMENTS = ['高价值客户', '流失风险客户', '沉睡客户', '潜力客户', '一般客户']
SCORE_METHOD = ('使用 Python 整数与 Fraction 精确线性插值的总体分位数 20%、40%、60%、80%；'
                '基础分数=1+严格大于边界的数量；R 分数=6-基础分数，F/M 使用基础分数；'
                '并列值同分，重复边界可跳分，固定 1—5 分。浮点 thresholds 仅用于展示（M 为英镑）；'
                'thresholds_exact 为精确分数字符串，R 单位为日、F 为订单数、M 为整数分。')


def _reference(value):
    if isinstance(value, str):
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) is None:
            raise ValueError('参考日期 reference_date 必须是完整 YYYY-MM-DD 或无时区日期对象')
    elif not isinstance(value, (date, datetime, pd.Timestamp)):
        raise ValueError('参考日期 reference_date 必须是完整 YYYY-MM-DD 或无时区日期对象')
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is not None:
            raise ValueError
        # Force nanosecond bounds so comparisons against cleaned sales are valid.
        timestamp = timestamp.as_unit('ns')
        return timestamp.normalize()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('参考日期 reference_date 无效或包含时区') from exc


def _empty():
    frame = pd.DataFrame({column: pd.Series(dtype='object') for column in RFM_COLUMNS})
    frame['CustomerID'] = frame.CustomerID.astype('string')
    frame['LastPurchase'] = pd.Series(dtype='datetime64[ns]')
    for column in ['Recency', 'Frequency', 'RScore', 'FScore', 'MScore']:
        frame[column] = pd.Series(dtype='int64')
    for column in ['Monetary', 'AverageOrderValue']:
        frame[column] = pd.Series(dtype='float64')
    return frame


def _metadata(frame, reference, window_days, thresholds=None, thresholds_exact=None):
    frame.attrs.update({
        'reference_date': reference.date().isoformat() if reference is not None else None,
        'window_days': window_days,
        'thresholds': thresholds if thresholds is not None else {'R': [], 'F': [], 'M': []},
        'thresholds_exact': thresholds_exact if thresholds_exact is not None else {'R': [], 'F': [], 'M': []},
        'score_method': SCORE_METHOD,
    })
    return frame


def _score(values, reverse=False):
    integers = [int(value) for value in values]
    ordered = sorted(integers)
    thresholds = []
    for numerator in range(1, 5):
        position = Fraction((len(ordered) - 1) * numerator, 5)
        lower = position.numerator // position.denominator
        upper = min(lower + 1, len(ordered) - 1)
        thresholds.append(Fraction(ordered[lower]) + (position - lower) * (ordered[upper] - ordered[lower]))
    score = np.array([1 + sum(value > boundary for boundary in thresholds) for value in integers])
    return (6 - score if reverse else score).astype('int64'), thresholds


def build_rfm(sales, reference_date=None, window_days=None):
    """在 [T-window_days, T) 内构造已知客户 RFM；None 表示全部 T 之前历史。

    默认 T 为包括匿名销售在内的最后销售日次日零点。显式截点排除未来数据；
    Recency 按日历日期计算，Frequency 统计不同发票，金额先按 Python 整数分求和。
    输入应来自 clean_transactions 的 sales，不会被修改。
    """
    if window_days is not None:
        if isinstance(window_days, bool) or not isinstance(window_days, Integral) or window_days <= 0:
            raise ValueError('窗口 window_days 必须是正整数或 None')
        window_days = int(window_days)
    reference = _reference(reference_date) if reference_date is not None else None
    if sales.empty:
        return _metadata(_empty(), reference, window_days)
    required = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'AmountCents', 'Quantity']
    missing = [column for column in required if column not in sales]
    if missing:
        raise ValueError('RFM 销售数据缺少列：' + ', '.join(missing))
    if reference is None:
        latest = sales.InvoiceDate.max()
        if pd.isna(latest):
            return _metadata(_empty(), None, window_days)
        reference = _reference(latest) + timedelta(days=1)
    mask = sales.CustomerID.notna() & sales.CustomerID.astype('string').str.strip().ne('')
    mask &= sales.InvoiceDate.lt(reference)
    if window_days is not None:
        try:
            lower = reference - timedelta(days=window_days)
        except (OverflowError, ValueError) as exc:
            raise ValueError('窗口 window_days 超出可支持日期范围') from exc
        mask &= sales.InvoiceDate.ge(lower)
    selected = sales.loc[mask, required].copy()
    if selected.empty:
        return _metadata(_empty(), reference, window_days)
    selected['CustomerID'] = selected.CustomerID.astype('string')
    grouped = selected.groupby('CustomerID', sort=True)
    result = grouped.agg(LastPurchase=('InvoiceDate', 'max'), Frequency=('InvoiceNo', 'nunique')).reset_index()
    quantities = {
        customer: sum(value.item() if isinstance(value, np.generic) else value for value in rows.Quantity)
        for customer, rows in grouped
    }
    result['Quantity'] = pd.Series([quantities[customer] for customer in result.CustomerID], dtype=object)
    result['Recency'] = (reference - result.LastPurchase.dt.normalize()).dt.days
    # Both per-customer and cross-customer totals may exceed signed int64.
    cents = grouped.AmountCents.apply(lambda values: sum(int(value) for value in values))
    result['MonetaryCents'] = pd.Series([int(cents.loc[customer]) for customer in result.CustomerID], dtype=object)
    result['Monetary'] = result.MonetaryCents.map(lambda amount: amount / 100).astype(float)
    result['AverageOrderValue'] = result.Monetary / result.Frequency
    thresholds, thresholds_exact = {}, {}
    for key, column, reverse in [('R', 'Recency', True), ('F', 'Frequency', False), ('M', 'MonetaryCents', False)]:
        result[key + 'Score'], exact = _score(result[column], reverse)
        thresholds_exact[key] = [str(value) for value in exact]
        thresholds[key] = [float(value / 100 if key == 'M' else value) for value in exact]
    r, f, m = result.RScore, result.FScore, result.MScore
    result['Segment'] = np.select(
        [(r >= 4) & (f >= 4) & (m >= 4), (r <= 2) & (f >= 3), r <= 2, r >= 4],
        SEGMENTS[:4], default=SEGMENTS[4],
    )
    return _metadata(result[RFM_COLUMNS], reference, window_days, thresholds, thresholds_exact)


def rfm_profile(frame):
    """返回固定顺序五类画像；占比使用 0—1，分母为零时为 None。"""
    required = ['Segment', 'MonetaryCents']
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError('RFM 画像缺少列：' + ', '.join(missing))
    if not frame.Segment.isin(SEGMENTS).all():
        raise ValueError('RFM 画像包含未知客户类型')
    total_count = len(frame)
    total_cents = sum(int(value) for value in frame.MonetaryCents)
    records = []
    for segment in SEGMENTS:
        subset = frame.loc[frame.Segment.eq(segment)]
        count = len(subset)
        cents = sum(int(value) for value in subset.MonetaryCents)
        records.append({
            'Segment': segment, 'Count': count, 'MonetaryCents': cents, 'Monetary': cents / 100,
            'CustomerShare': count / total_count if total_count else None,
            'RevenueShare': cents / total_cents if total_cents else None,
        })
    result = pd.DataFrame(records)
    result['MonetaryCents'] = pd.Series([row['MonetaryCents'] for row in records], dtype=object)
    for column in ['CustomerShare', 'RevenueShare']:
        result[column] = pd.Series([row[column] for row in records], dtype=object)
    return result


def rfm_rules():
    """分群规则按优先级从上到下命中，R/F/M 均指 1—5 的分数。"""
    conditions = [
        '优先级1：R、F、M 均为分数；R≥4 且 F≥4 且 M≥4',
        '优先级2：R、F、M 均为分数；未命中上项且 R≤2 且 F≥3',
        '优先级3：R、F、M 均为分数；未命中上项且 R≤2',
        '优先级4：R、F、M 均为分数；未命中上项且 R≥4',
        '优先级5：R、F、M 均为分数；其余客户',
    ]
    recommendations = ['提供会员关怀与专属权益，维护长期关系', '分析购买间隔并测试针对性召回活动',
                       '以低成本触达确认需求，控制召回投入', '推荐相关商品，测试第二次购买激励',
                       '定期沟通并观察购买习惯变化']
    return pd.DataFrame({'客户类型': SEGMENTS, '判定条件': conditions, '运营建议': recommendations})
