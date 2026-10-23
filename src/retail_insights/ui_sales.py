"""交易分析页面：所有图表与导出使用同一筛选范围。"""
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from .analytics import filter_transactions, sales_summary, sales_tables
from .preparation import load_dataset
from .sales_presentation import display_table, country_name, export_analysis, WEEKDAYS


@st.cache_data(show_spinner=False, max_entries=2)
def _dataset(root, pointer):
    return load_dataset(root, json.loads(pointer)['dataset_id'])


def _initialize_filters(countries, lo, hi, version):
    state = st.session_state
    if (state.get('sales_version') != version or 'sales_country_selection' not in state
            or 'sales_date_selection' not in state):
        state['sales_country_selection'] = dict.fromkeys(countries, True)
        state['sales_date_selection'] = (lo, hi)
        state['sales_start'], state['sales_end'] = lo, hi
        state['sales_country_search'] = ''
        state['sales_version'] = version
        for key in list(state):
            if key.startswith('sales_country_choice_'):
                del state[key]
    start, end = state.get('sales_date_selection', (lo, hi))
    state.setdefault('sales_start', start)
    state.setdefault('sales_end', end)


def _remember_dates():
    st.session_state['sales_date_selection'] = (st.session_state['sales_start'], st.session_state['sales_end'])


def _select_all(countries, value):
    st.session_state['sales_country_selection'] = dict.fromkeys(countries, value)
    for country in countries:
        st.session_state['sales_country_choice_' + country] = value


def _remember_country(country):
    st.session_state['sales_country_selection'][country] = st.session_state['sales_country_choice_' + country]


def _country_picker(options):
    selection = st.session_state['sales_country_selection']
    count = sum(selection.get(country, False) for country in options)
    st.markdown('**国家或地区**')
    with st.expander('查看全部可选国家或地区并筛选'):
        st.caption(f'已选 {count}/{len(options)} 项')
        query = st.text_input('搜索国家或地区', key='sales_country_search', placeholder='输入名称，按回车搜索')
        all_box, clear_box = st.columns(2)
        all_box.button('全选', key='sales_countries_all', on_click=_select_all, args=(options, True))
        clear_box.button('清空', key='sales_countries_clear', on_click=_select_all, args=(options, False))
        visible = [country for country in sorted(options, key=country_name)
                   if not query or query.strip().casefold() in country_name(country).casefold()
                   or query.strip().casefold() in country.casefold()]
        with st.container(height=260):
            boxes = st.columns(2)
            for index, country in enumerate(visible):
                key = 'sales_country_choice_' + country
                st.session_state.setdefault(key, selection.get(country, False))
                boxes[index % 2].checkbox(country_name(country), key=key,
                                           on_change=_remember_country, args=(country,))
            if not visible:
                st.info('没有匹配的国家或地区。')
        st.caption('按回车搜索，支持中文或原始名称。搜索不改变已选项；全选和清空作用于全部国家或地区。')
    st.caption(f'已选择 {count} 个国家或地区，展开列表即可逐项勾选。')
    return [country for country in options if selection.get(country, False)]


def _chart(figure):
    figure.update_layout(font={'family': 'Microsoft YaHei, PingFang SC, sans-serif'},
                         hoverlabel={'font': {'family': 'Microsoft YaHei, PingFang SC, sans-serif'}})
    figure.update_yaxes(tickformat=',.0f', hoverformat=',.2f')
    st.plotly_chart(figure, width='stretch', config={'displayModeBar': False})


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
    _initialize_filters(country_options, lo, hi, data['metadata']['dataset_id'])
    left, right = st.columns([2, 3])
    with left:
        st.markdown('**交易日期（包含起止日）**')
        start_box, end_box = st.columns(2)
        start = start_box.date_input('开始日期', min_value=lo, max_value=hi,
                                     format='YYYY/MM/DD', key='sales_start', on_change=_remember_dates)
        end = end_box.date_input('结束日期', min_value=lo, max_value=hi,
                                 format='YYYY/MM/DD', key='sales_end', on_change=_remember_dates)
    with right:
        countries = _country_picker(country_options)
    if start is None or end is None:
        st.info('请选择开始日期和结束日期。')
        return
    try:
        filtered = filter_transactions(sales, start, end, countries)
        canceled = filter_transactions(returns, start, end, countries)
    except ValueError as error:
        st.error(str(error))
        return
    if filtered.empty:
        st.info('当前筛选范围没有有效销售，请调整日期或国家。')
        refund = sum(int(value) for value in canceled.AmountCents) / 100
        st.caption(f'同范围取消/退货候选 {len(canceled):,} 条，带符号金额 £{refund:,.2f}。')
        return
    stats, tables = sales_summary(filtered), sales_tables(filtered)
    metric_boxes = st.columns(3) + st.columns(3)
    for box, title, value in zip(metric_boxes,
                                ['商品销售额', '销售订单', '可识别客户', '平均订单金额', '复购客户占比', '售出件数'],
                                [f"£{stats['revenue']:,.2f}", f"{stats['orders']:,}",
                                 f"{stats['customers']:,}", f"£{stats['aov']:,.2f}",
                                 '无数据' if stats['repeat_rate'] is None else f"{stats['repeat_rate']:.1%}",
                                 f"{stats['items']:,}"]):
        box.metric(title, value)
    refund = sum(int(value) for value in canceled.AmountCents) / 100
    st.caption(f'同范围取消/退货候选 {len(canceled):,} 条，带符号金额 £{refund:,.2f}。'
               '复购占比仅使用有客户编号的买家，统计期内至少两单视为复购。')
    shown = {name: display_table(frame) for name, frame in tables.items()}
    trend, product, region, timing, customers = st.tabs(['销售趋势', '商品排行', '国家分布', '购买时段', '客户消费'])
    with trend:
        chart = px.line(shown['monthly'], x='月份', y='销售额（英镑）', markers=True)
        chart.update_xaxes(type='category')
        _chart(chart)
        st.dataframe(shown['monthly'], hide_index=True, width='stretch')
    with product:
        top = shown['products'].head(10)
        _chart(px.bar(top, x='商品编码', y='销售额（英镑）', hover_data=['商品名称（原始）', '销售数量']))
        st.dataframe(top, hide_index=True, width='stretch')
    with region:
        _chart(px.bar(shown['countries'].head(15), x='国家或地区', y='销售额（英镑）'))
    with timing:
        heat = tables['hour_week'].pivot(index='Weekday', columns='Hour', values='Orders')
        heat = heat.reindex(index=range(7), columns=range(24), fill_value=0).fillna(0)
        heat.index = WEEKDAYS
        _chart(px.imshow(heat, labels={'x': '小时', 'y': '星期', 'color': '订单数'},
                         color_continuous_scale='Blues', aspect='auto'))
        st.caption('同一订单在同一时段只计一次；跨时段或地区的订单分组计数不保证可加。')
    with customers:
        spend = shown['customer_spend']
        if spend.empty:
            st.info('没有可识别客户，匿名交易仍计入整体销售额。')
        else:
            chart = px.histogram(spend, x='销售额（英镑）', nbins=40)
            chart.update_xaxes(tickformat=',.0f', hoverformat=',.2f')
            chart.update_yaxes(title_text='客户数')
            chart.update_traces(hovertemplate='消费金额：%{x} 英镑<br>客户数：%{y}<extra></extra>')
            _chart(chart)
            st.dataframe(spend.head(100), hide_index=True, width='stretch')
    st.download_button('下载当前筛选的分析表',
                       export_analysis(tables, stats, data['metadata']['dataset_id'], start, end, countries),
                       file_name='交易分析.zip', mime='application/zip')
