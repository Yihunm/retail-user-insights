"""RFM 中文展示与可追溯结果导出。"""
import io
import json
import zipfile

from .rfm import rfm_profile, rfm_rules

CUSTOMER_COLUMNS = {
    'CustomerID': '客户编号', 'LastPurchase': '最近购买时间', 'Recency': '最近购买间隔（天）',
    'Frequency': '购买订单数', 'Monetary': '消费金额（英镑）', 'Quantity': '购买数量',
    'AverageOrderValue': '平均订单金额（英镑）', 'RScore': 'R评分', 'FScore': 'F评分',
    'MScore': 'M评分', 'Segment': '客户类型',
}
PROFILE_COLUMNS = {'Segment': '客户类型', 'Count': '客户数', 'Monetary': '消费金额（英镑）',
                   'CustomerShare': '客户占比', 'RevenueShare': '金额占比'}


def display_rfm(frame, include_cents=False):
    shown = frame[list(CUSTOMER_COLUMNS)].copy()
    shown['LastPurchase'] = shown['LastPurchase'].dt.strftime('%Y-%m-%d %H:%M:%S')
    if include_cents:
        shown['MonetaryCents'] = frame['MonetaryCents'].map(str)
    return shown.rename(columns={**CUSTOMER_COLUMNS, 'MonetaryCents': '消费金额（整数分）'})


def display_profile(frame):
    return frame[list(PROFILE_COLUMNS)].rename(columns=PROFILE_COLUMNS)


def export_rfm(selected, metadata):
    description = {
        '数据版本': metadata['dataset_id'], '分析参考日': metadata['reference_date'],
        '观察窗口天数': metadata['window_days'], '评分基准客户数': metadata['cohort_customers'],
        '导出客户数': len(selected), '客户类型筛选': metadata['selected_segments'],
        '客户编号搜索': metadata['customer_search'], '评分分位点': metadata['thresholds'],
        '精确评分分位点': metadata.get('thresholds_exact', {}),
        '精确分位点单位': 'R为天，F为订单数，M为整数分；分数形式表示精确线性插值。',
        '评分说明': '以观察窗口内全部可识别客户计算20%、40%、60%、80%分位点；同值同分，R反向。展示筛选不重新评分。',
        '金额口径': '有效商品销售，不含匿名客户、取消/退货候选及隔离记录。',
    }
    content = io.BytesIO()
    with zipfile.ZipFile(content, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, table in [('RFM客户明细.csv', display_rfm(selected, include_cents=True)),
                            ('客户分层汇总.csv', display_profile(rfm_profile(selected))),
                            ('分层规则.csv', rfm_rules())]:
            archive.writestr(name, table.to_csv(index=False).encode('utf-8-sig'))
        archive.writestr('评分与筛选说明.json', json.dumps(description, ensure_ascii=False, allow_nan=False, indent=2))
    return content.getvalue()
