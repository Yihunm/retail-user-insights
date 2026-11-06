"""以真实交易摘要和客户经营入口作为默认首页。"""
import json
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from .analytics import sales_summary
from .preparation import load_dataset
from .rfm import build_rfm, rfm_profile
from .ui_shared import chart, go, page_header
from .ui_theme import PRIMARY, SEGMENT_COLORS, icon


@st.cache_data(show_spinner=False, max_entries=2)
def _overview(root, pointer):
    data = load_dataset(root, json.loads(pointer)['dataset_id'])
    sales = data['sales']
    if sales.empty:
        return None
    reference = (sales.InvoiceDate.max().normalize() + timedelta(days=1)).date()
    cohort = build_rfm(sales, reference_date=reference)
    trend = pd.DataFrame([{'月份': month, '销售额（英镑）': sum(int(x) for x in group.AmountCents) / 100}
        for month, group in sales.groupby(sales.InvoiceDate.dt.strftime('%Y-%m'), sort=True)])
    months = pd.period_range(sales.InvoiceDate.min(), sales.InvoiceDate.max(), freq='M').astype(str)
    trend = trend.set_index('月份').reindex(months, fill_value=0).rename_axis('月份').reset_index()
    return {'stats': sales_summary(sales), 'profile': rfm_profile(cohort), 'cohort': cohort,
            'trend': trend, 'start': sales.InvoiceDate.min().date(), 'end': sales.InvoiceDate.max().date(),
            'reference': reference, 'version': data['metadata']['dataset_id'],
            'anonymous': int(sales.CustomerID.isna().sum())}


def _value_customers(segment, overview):
    reference = overview['reference']
    segments = [segment]
    memory = {'rfm_reference': reference, 'rfm_window': '全部历史',
              'rfm_segments': segments, 'rfm_customer_search': ''}
    go('客户价值', rfm_version=overview['version'], rfm_memory=memory, **memory)


def render(root):
    page_header('工作空间 / 经营总览', '经营工作台', '总览经营表现，发现值得关注的客户。')
    pointer = root / 'data' / 'active.json'
    if not pointer.exists():
        with st.container(border=True):
            st.subheader('从一份交易文件开始')
            st.write('导入 Excel 或 CSV，自动检查数据，然后查看销售表现、客户价值与购买特征。')
            st.button('导入交易数据', key='home_import', type='primary', on_click=go, args=('数据管理',))
        st.info('尚未导入数据。导入后，这里会显示真实的经营摘要和客户分析入口。')
        return
    try:
        overview = _overview(str(root), pointer.read_text(encoding='utf-8'))
    except (ValueError, OSError, KeyError) as error:
        st.error(f'暂时无法读取经营数据：{error}')
        st.button('检查数据', key='home_import', on_click=go, args=('数据管理',))
        return
    if overview is None:
        st.info('当前文件没有有效商品销售，请到数据管理检查文件和质量报告。')
        st.button('检查数据', key='home_import', on_click=go, args=('数据管理',))
        return
    stats = overview['stats']
    st.caption(f"{overview['start']} — {overview['end']} · 全部历史 · 金额单位：英镑")
    values = [f"£{stats['revenue']:,.2f}", f"{stats['orders']:,}", f"{stats['customers']:,}",
              '无可识别客户' if stats['repeat_rate'] is None else f"{stats['repeat_rate']:.1%}"]
    with st.container(key='home_kpis'):
        for box, label, value in zip(st.columns(4), ['商品销售额', '成交订单数', '购买客户数', '复购客户占比'], values):
            box.metric(label, value)
    st.caption('销售额包含匿名购买，客户数与复购占比仅统计有编号的客户；复购指本数据期间至少两笔订单。')
    cohort = overview['cohort']
    trend_box, distribution_box = st.columns([3, 2])
    with trend_box.container(border=False, key='home_trend_card'):
        st.markdown('<div class="panel-heading"><span class="panel-title">销售走势</span>'
                    '<span class="panel-subtitle">按月汇总 · 英镑</span></div>', unsafe_allow_html=True)
        figure = px.area(overview['trend'], x='月份', y='销售额（英镑）', color_discrete_sequence=[PRIMARY])
        figure.update_traces(line={'width': 2.5}, fillcolor='rgba(88,101,217,.08)',
                             hovertemplate='%{x}<br>销售额 £%{y:,.2f}<extra></extra>')
        figure.update_layout(height=280, margin={'l': 4, 'r': 8, 't': 12, 'b': 4})
        figure.update_xaxes(type='category', title=None, showgrid=False, nticks=7)
        figure.update_yaxes(title=None, tickformat=',.0f', nticks=5)
        chart(figure)
        st.caption('首尾月份可能不完整，不据此直接判断环比增长。')
        st.button('查看经营分析', key='home_sales', on_click=go, args=('经营分析',))
    with distribution_box.container(border=False, key='home_distribution_card'):
        st.markdown('<div class="panel-heading"><span class="panel-title">客户价值分布</span>'
                    '<span class="panel-subtitle">历史购买客户</span></div>', unsafe_allow_html=True)
        profile = overview['profile'].rename(columns={'Segment': '客户类型', 'Count': '客户数'})
        if cohort.empty:
            st.info('当前交易没有可识别客户，仍可查看整体经营分析。')
        else:
            figure = px.pie(profile[profile['客户数'].gt(0)], values='客户数', names='客户类型', hole=.76,
                            color='客户类型', color_discrete_map=SEGMENT_COLORS)
            figure.update_traces(textinfo='none', marker={'line': {'color': '#FFFFFF', 'width': 3}},
                                 hovertemplate='%{label}<br>%{value:,} 位 · %{percent}<extra></extra>')
            figure.update_layout(height=280, margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
                                 legend={'orientation': 'h', 'y': -.1, 'x': .5, 'xanchor': 'center', 'font': {'size': 10}})
            figure.add_annotation(x=.5, y=.55, text=f'{len(cohort):,}', showarrow=False,
                                  font={'size': 30, 'color': '#27334F', 'family': 'Arial, sans-serif'})
            figure.add_annotation(x=.5, y=.40, text='购买客户', showarrow=False, font={'size': 11, 'color': '#929CB0'})
            chart(figure)
        st.caption('基于历史购买表现的规则分层，非未来预测。')
        st.button('查看复购与流失风险', key='home_predictions', on_click=go, args=('复购与流失',))
    st.markdown('<div class="section-heading">客户经营</div>'
                '<div class="section-description">选择一个方向，查看客户特征或导出名单。</div>', unsafe_allow_html=True)
    high = int(cohort.Segment.eq('高价值客户').sum())
    risk = int(cohort.Segment.eq('流失风险客户').sum())
    value_box, risk_box, group_box = st.columns(3)
    with value_box.container(border=False, key='home_value_card'):
        st.markdown('<div class="action-top"><span class="action-icon value">' + icon('value') + '</span>'
                    '<span class="action-title">维护高价值客户</span></div>'
                    f'<div class="action-number">{high:,}<small>位高价值客户</small></div>'
                    '<div class="action-copy">根据购买频次与消费金额，<br>安排更有针对性的客户维护。</div>', unsafe_allow_html=True)
        st.button('查看重点客户', key='home_value', on_click=_value_customers, args=('高价值客户', overview),
                  width='stretch')
    with risk_box.container(border=False, key='home_risk_card'):
        st.markdown('<div class="action-top"><span class="action-icon risk">' + icon('risk') + '</span>'
                    '<span class="action-title">关注久未回购客户</span></div>'
                    f'<div class="action-number">{risk:,}<small>位规则风险客户</small></div>'
                    '<div class="action-copy">先核对客户的购买间隔，<br>再考虑合适的回访与沟通。</div>', unsafe_allow_html=True)
        st.button('查看待关注客户', key='home_risk', on_click=_value_customers, args=('流失风险客户', overview),
                  width='stretch')
    with group_box.container(border=False, key='home_group_card'):
        st.markdown('<div class="action-top"><span class="action-icon groups">' + icon('groups') + '</span>'
                    '<span class="action-title">了解不同购买特征</span></div>'
                    '<div class="action-number">差异化经营<small>群体画像</small></div>'
                    '<div class="action-copy">发现客户群体之间的购买差异，<br>为不同群体制定沟通方式。</div>', unsafe_allow_html=True)
        st.button('查看客户分群', key='home_groups', on_click=go, args=('客户分群',), width='stretch')
    st.caption(f"客户规则以 {overview['reference']} 为参考日、使用全部历史；“流失风险”是规则分层，不是未来预测。")
