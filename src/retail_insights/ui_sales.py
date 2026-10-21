"""交易分析页面：所有图表与导出使用同一筛选范围。"""
import io
import json
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st

from .analytics import filter_transactions, sales_summary, sales_tables
from .preparation import load_dataset


@st.cache_data(show_spinner=False, max_entries=2)
def _dataset(root, pointer):
    return load_dataset(root, json.loads(pointer)['dataset_id'])


def render(root):
    st.title('交易统计与可视化')
    st.caption('以有效商品销售为统计口径；取消/退货候选独立展示。')
    pointer = root / 'data' / 'active.json'
    if not pointer.exists():
        st.info('请先在“数据导入与清洗”中导入数据。')
        return
    try:
        data = _dataset(str(root), pointer.read_text(encoding='utf-8'))
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    sales, returns = data['sales'], data['returns']
    if sales.empty:
        st.info('当前数据没有有效商品销售，无法生成交易看板。')
        return
    st.caption(f"数据版本 {data['metadata']['dataset_id'][:12]}")
    dates = pd.concat([sales.InvoiceDate, returns.InvoiceDate]).dropna()
    country_options = sorted(set(sales.Country) | set(returns.Country))
    lo, hi = dates.min().date(), dates.max().date()
    if (st.session_state.get('sales_version') != data['metadata']['dataset_id']
            or 'sales_dates' not in st.session_state or 'sales_countries' not in st.session_state):
        st.session_state['sales_dates'] = (lo, hi)
        st.session_state['sales_countries'] = country_options
        st.session_state['sales_version'] = data['metadata']['dataset_id']
    left, right = st.columns([2, 3])
    selected = left.date_input('交易日期（包含起止日）', min_value=lo, max_value=hi, key='sales_dates')
    countries = right.multiselect('国家或地区', country_options, key='sales_countries')
    if len(selected) != 2:
        st.info('请选择完整的开始与结束日期。')
        return
    try:
        filtered = filter_transactions(sales, selected[0], selected[1], countries)
        canceled = filter_transactions(returns, selected[0], selected[1], countries)
    except ValueError as error:
        st.error(str(error))
        return
    if filtered.empty:
        st.info('当前筛选范围没有有效销售，请调整日期或国家。')
        refund = sum(int(value) for value in canceled.AmountCents) / 100
        st.caption(f'同范围取消/退货候选 {len(canceled):,} 条，带符号金额 £{refund:,.2f}。')
        return
    stats, tables = sales_summary(filtered), sales_tables(filtered)
    for box, title, value in zip(st.columns(6),
                                ['商品销售额', '销售订单', '可识别客户', '平均订单金额', '复购客户占比', '售出件数'],
                                [f"£{stats['revenue']:,.2f}", f"{stats['orders']:,}",
                                 f"{stats['customers']:,}", f"£{stats['aov']:,.2f}",
                                 '无数据' if stats['repeat_rate'] is None else f"{stats['repeat_rate']:.1%}",
                                 f"{stats['items']:,}"]):
        box.metric(title, value)
    refund = sum(int(value) for value in canceled.AmountCents) / 100
    st.caption(f'同范围取消/退货候选 {len(canceled):,} 条，带符号金额 £{refund:,.2f}。'
               '复购占比仅使用有客户编号的买家，统计期内至少两单视为复购。')
    trend, product, region, timing, customers = st.tabs(['销售趋势', '商品排行', '国家分布', '购买时段', '客户消费'])
    with trend:
        st.plotly_chart(px.line(tables['monthly'], x='Month', y='Revenue', markers=True,
                               labels={'Month': '月份', 'Revenue': '商品销售额（GBP）'}), width='stretch')
        st.dataframe(tables['monthly'], hide_index=True, width='stretch')
    with product:
        top = tables['products'].head(10)
        st.plotly_chart(px.bar(top, x='StockCode', y='Revenue', hover_data=['Description', 'Quantity'],
                              labels={'StockCode': '商品编码', 'Revenue': '商品销售额（GBP）'}), width='stretch')
        st.dataframe(top, hide_index=True, width='stretch')
    with region:
        st.plotly_chart(px.bar(tables['countries'].head(15), x='Country', y='Revenue',
                              labels={'Country': '国家或地区', 'Revenue': '商品销售额（GBP）'}), width='stretch')
    with timing:
        heat = tables['hour_week'].pivot(index='Weekday', columns='Hour', values='Orders')
        heat = heat.reindex(index=range(7), columns=range(24), fill_value=0).fillna(0)
        heat.index = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        st.plotly_chart(px.imshow(heat, labels={'x': '小时', 'y': '星期', 'color': '订单数'},
                                 color_continuous_scale='Blues', aspect='auto'), width='stretch')
        st.caption('同一订单在同一时段只计一次；跨时段或地区的订单分组计数不保证可加。')
    with customers:
        spend = tables['customer_spend']
        if spend.empty:
            st.info('没有可识别客户，匿名交易仍计入整体销售额。')
        else:
            st.plotly_chart(px.histogram(spend, x='Revenue', nbins=40,
                                        labels={'Revenue': '客户消费金额（GBP）'}), width='stretch')
            st.dataframe(spend.head(100), hide_index=True, width='stretch')
    export = io.BytesIO()
    with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            archive.writestr(f'{name}.csv', frame.to_csv(index=False).encode('utf-8-sig'))
        archive.writestr('summary.json', json.dumps({**stats, 'dataset_id': data['metadata']['dataset_id'],
                                                    'start': str(selected[0]), 'end': str(selected[1]),
                                                    'countries': countries}, ensure_ascii=False, allow_nan=False))
    st.download_button('下载当前筛选的分析表 ZIP', export.getvalue(), file_name='sales_analysis.zip',
                       mime='application/zip')
