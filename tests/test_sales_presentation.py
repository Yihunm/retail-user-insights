import io
import json
import zipfile

import pandas as pd


def test_country_translation_is_display_only_and_known_variants_are_covered():
    from retail_insights.sales_presentation import country_name, display_table

    assert country_name('UK') == country_name('United Kingdom') == '英国'
    assert country_name('EIRE') == '爱尔兰'
    assert country_name('RSA') == '南非'
    assert country_name('Unspecified') == '未指定'
    original = pd.DataFrame({'Country': ['France'], 'Revenue': [1.5], 'Orders': [1]})
    shown = display_table(original)
    assert shown.iloc[0]['国家或地区'] == '法国'
    assert original.iloc[0].Country == 'France'


def test_export_has_chinese_headers_country_names_and_selected_scope():
    from retail_insights.sales_presentation import export_analysis

    tables = {'countries': pd.DataFrame({'Country': ['UK'], 'Revenue': [5.1], 'Orders': [1]})}
    summary = {'revenue': 5.1, 'orders': 1, 'customers': 1, 'items': 2, 'aov': 5.1, 'repeat_rate': 0}
    content = export_analysis(tables, summary, 'dataset1', '2011-01-01', '2011-01-01', ['UK'])
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        frame = pd.read_csv(io.BytesIO(archive.read('国家销售.csv')))
        assert list(frame.columns) == ['国家或地区', '销售额（英镑）', '订单数']
        assert frame.iloc[0]['国家或地区'] == '英国'
        scope = json.loads(archive.read('筛选说明.json'))
        assert scope['国家或地区'] == ['英国']
        assert scope['开始日期'] == scope['结束日期'] == '2011-01-01'
