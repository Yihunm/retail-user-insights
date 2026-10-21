"""Sales analysis of cleaned transaction frames.

Revenue is GBP, summed from row-rounded integer cents before conversion.
Orders are distinct InvoiceNo values within the current frame or group. An invoice
that spans countries, months or hours can count in several groups; these group
order counts need not add up to the overall order count.
"""

import re
from datetime import date, datetime, timedelta

import pandas as pd


def _calendar_day(value):
    if value is None:
        return None
    if isinstance(value, str):
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            raise ValueError('日期必须是完整的 YYYY-MM-DD')
    elif not isinstance(value, (date, datetime, pd.Timestamp)):
        raise ValueError('日期必须是完整的 YYYY-MM-DD')
    try:
        parsed = pd.Timestamp(value)
        if pd.isna(parsed) or parsed.tzinfo is not None:
            raise ValueError('日期不可为空或包含时区')
        return parsed.normalize()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('无效日期') from exc


def filter_transactions(frame, start=None, end=None, countries=None):
    """Return an independent selection, including both calendar day bounds.

    Bounds accept ISO full-date strings or naive date/datetime objects. A country
    selection of None includes all countries; an empty list selects no rows.
    """
    start_day, end_day = _calendar_day(start), _calendar_day(end)
    if start_day is not None and end_day is not None and start_day > end_day:
        raise ValueError('开始日期不能晚于结束日期')
    mask = pd.Series(True, index=frame.index)
    if start_day is not None:
        mask &= frame['InvoiceDate'] >= start_day
    if end_day is not None:
        mask &= frame['InvoiceDate'] < end_day + timedelta(days=1)
    if countries is not None:
        mask &= frame['Country'].isin(countries)
    return frame.loc[mask].copy()


def sales_summary(sales):
    """Compute sales KPIs; repeat buyers have at least two selected orders.

    Anonymous revenue and orders are included. Customer count and repeat rate
    include only identified customers. Undefined ratios are None.
    """
    revenue = sum(int(value) for value in sales['AmountCents']) / 100
    orders = int(sales['InvoiceNo'].nunique())
    customer_orders = sales.dropna(subset=['CustomerID']).groupby('CustomerID')['InvoiceNo'].nunique()
    customers = len(customer_orders)
    return {
        'revenue': revenue,
        'orders': orders,
        'customers': customers,
        'items': sum(sales['Quantity'].tolist()),
        'aov': revenue / orders if orders else None,
        'repeat_rate': int(customer_orders.ge(2).sum()) / customers if customers else None,
    }


def _money_groups(frame, keys, *, quantity=False):
    """Aggregate cents with Python integers, sorting before float conversion."""
    records = []
    for group_key, group in frame.groupby(keys, sort=True, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        record = dict(zip(keys, group_key))
        record['_cents'] = sum(int(value) for value in group['AmountCents'])
        metric = 'Quantity' if quantity else 'Orders'
        record[metric] = (sum(group['Quantity'].tolist()) if quantity
                          else int(group['InvoiceNo'].nunique()))
        records.append(record)
    # groupby sorts keys, and the stable revenue sort preserves deterministic ties.
    records.sort(key=lambda record: record['_cents'], reverse=True)
    for record in records:
        record['Revenue'] = record.pop('_cents') / 100
    return pd.DataFrame(records, columns=[*keys, 'Revenue', 'Quantity' if quantity else 'Orders'])


def sales_tables(sales):
    """Return five display tables; monthly and time bins are chronological.

    Product, country and customer tables are ordered by descending revenue with
    lexical key ties. Product labels use the most common description, breaking
    frequency ties lexically. Monthly gaps within the frame are filled with zero.
    """
    frame = sales.copy()
    frame['Month'] = frame['InvoiceDate'].dt.strftime('%Y-%m')
    monthly = _money_groups(frame, ['Month'])
    if not frame.empty:
        months = pd.period_range(frame['Month'].min(), frame['Month'].max(), freq='M').astype(str)
        monthly = monthly.set_index('Month').reindex(months, fill_value=0).rename_axis('Month').reset_index()

    products = _money_groups(frame, ['StockCode'], quantity=True)
    descriptions = {}
    for code, group in frame.groupby('StockCode', sort=True, observed=True):
        counts = group['Description'].dropna().value_counts()
        descriptions[code] = (min(counts[counts.eq(counts.max())].index) if not counts.empty else '')
    products.insert(1, 'Description', products['StockCode'].map(descriptions))

    countries = _money_groups(frame, ['Country'])
    customers = _money_groups(frame.dropna(subset=['CustomerID']), ['CustomerID'])
    frame['Weekday'] = frame['InvoiceDate'].dt.dayofweek
    frame['Hour'] = frame['InvoiceDate'].dt.hour
    hour_week = (frame.groupby(['Weekday', 'Hour'], sort=True, observed=True)['InvoiceNo']
                 .nunique().rename('Orders').reset_index())
    return dict(monthly=monthly, products=products, countries=countries,
                hour_week=hour_week, customer_spend=customers)
