import io
import json
import zipfile

import pandas as pd


def test_export_preserves_selected_customers_and_scoring_basis():
    from retail_insights.rfm_presentation import export_rfm

    selected = pd.DataFrame({'CustomerID': ['a'], 'LastPurchase': pd.to_datetime(['2011-01-01']),
                             'Recency': [2], 'Frequency': [1], 'MonetaryCents': [510], 'Monetary': [5.1],
                             'Quantity': [2], 'AverageOrderValue': [5.1], 'RScore': [5], 'FScore': [1],
                             'MScore': [1], 'Segment': ['潜力客户']})
    metadata = {'dataset_id': 'version-a', 'reference_date': '2011-01-03', 'window_days': 90,
                'cohort_customers': 20, 'thresholds': {'R': [1, 2, 3, 4], 'F': [1, 2, 3, 4], 'M': [1, 2, 3, 4]},
                'selected_segments': ['潜力客户'], 'customer_search': 'a'}
    content = export_rfm(selected, metadata)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        customers = pd.read_csv(io.BytesIO(archive.read('RFM客户明细.csv')), dtype={'客户编号': str})
        assert customers['客户编号'].tolist() == ['a']
        assert customers['消费金额（英镑）'].sum() == 5.1
        description = json.loads(archive.read('评分与筛选说明.json'))
        assert description['评分基准客户数'] == 20
        assert description['导出客户数'] == 1
        assert description['客户编号搜索'] == 'a'


def test_export_retains_exact_integer_cents_for_large_amounts():
    from retail_insights.rfm_presentation import export_rfm
    from retail_insights.rfm import build_rfm

    sales = pd.DataFrame({'CustomerID': ['a'], 'InvoiceNo': ['1'],
                          'InvoiceDate': pd.to_datetime(['2011-01-01']),
                          'AmountCents': pd.Series([9_000_000_000_000_000_001], dtype=object),
                          'Quantity': [1]})
    rfm = build_rfm(sales)
    metadata = {'dataset_id': 'v', 'reference_date': '2011-01-02', 'window_days': None,
                'cohort_customers': 1, 'thresholds': rfm.attrs['thresholds'],
                'selected_segments': ['潜力客户'], 'customer_search': ''}
    with zipfile.ZipFile(io.BytesIO(export_rfm(rfm, metadata))) as archive:
        frame = pd.read_csv(io.BytesIO(archive.read('RFM客户明细.csv')), dtype=str)
        assert frame.iloc[0]['消费金额（整数分）'] == '9000000000000000001'
