"""中文展示及导出映射，不改变原始字段和筛选值。"""
import io
import json
import zipfile

COUNTRY_NAMES = {
    'Australia': '澳大利亚', 'Austria': '奥地利', 'Bahrain': '巴林', 'Belgium': '比利时',
    'Brazil': '巴西', 'Canada': '加拿大', 'Channel Islands': '海峡群岛', 'Cyprus': '塞浦路斯',
    'Czech Republic': '捷克', 'Denmark': '丹麦', 'EIRE': '爱尔兰', 'European Community': '欧洲共同体',
    'Finland': '芬兰', 'France': '法国', 'Germany': '德国', 'Greece': '希腊',
    'Hong Kong': '中国香港', 'Iceland': '冰岛', 'Israel': '以色列', 'Italy': '意大利',
    'Japan': '日本', 'Lebanon': '黎巴嫩', 'Lithuania': '立陶宛', 'Malta': '马耳他',
    'Netherlands': '荷兰', 'Norway': '挪威', 'Poland': '波兰', 'Portugal': '葡萄牙',
    'RSA': '南非', 'Saudi Arabia': '沙特阿拉伯', 'Singapore': '新加坡', 'Spain': '西班牙',
    'Sweden': '瑞典', 'Switzerland': '瑞士', 'USA': '美国', 'United Arab Emirates': '阿联酋',
    'United Kingdom': '英国', 'UK': '英国', 'Unspecified': '未指定', '未知': '未知',
}
COLUMN_NAMES = {
    'Month': '月份', 'Revenue': '销售额（英镑）', 'Orders': '订单数', 'StockCode': '商品编码',
    'Description': '商品名称（原始）', 'Quantity': '销售数量', 'Country': '国家或地区',
    'Weekday': '星期', 'Hour': '小时', 'CustomerID': '客户编号',
}
TABLE_NAMES = {'monthly': '月度销售', 'products': '商品销售', 'countries': '国家销售',
               'hour_week': '购买时段', 'customer_spend': '客户消费'}
WEEKDAYS = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']


def country_name(value):
    # Unknown future data keeps its source name rather than inventing a translation.
    return COUNTRY_NAMES.get(value, value)


def display_table(frame):
    shown = frame.copy()
    if 'Country' in shown:
        shown['Country'] = shown['Country'].map(country_name)
    if 'Weekday' in shown:
        shown['Weekday'] = shown['Weekday'].map(dict(enumerate(WEEKDAYS)))
    if 'Month' in shown:
        shown['Month'] = shown['Month'].str.replace('-', '年', regex=False) + '月'
    return shown.rename(columns=COLUMN_NAMES)


def export_analysis(tables, summary, dataset_id, start, end, countries):
    content = io.BytesIO()
    labels = {'revenue': '商品销售额（英镑）', 'orders': '订单数', 'customers': '可识别客户数',
              'items': '销售数量', 'aov': '平均订单金额（英镑）', 'repeat_rate': '复购客户占比'}
    description = {labels.get(key, key): value for key, value in summary.items()}
    description.update({'数据版本': dataset_id, '开始日期': str(start), '结束日期': str(end),
                        '国家或地区': [country_name(country) for country in countries]})
    with zipfile.ZipFile(content, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, table in tables.items():
            archive.writestr(TABLE_NAMES[name] + '.csv',
                             display_table(table).to_csv(index=False).encode('utf-8-sig'))
        archive.writestr('筛选说明.json', json.dumps(description, ensure_ascii=False, allow_nan=False, indent=2))
    return content.getvalue()
