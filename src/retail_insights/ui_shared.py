"""面向经营人员的导航、页面标题与统一图表样式。"""
import json
from html import escape

import streamlit as st

from .ui_theme import INK, PALETTE, styles, icon

PAGES = ['工作台', '经营分析', '客户价值', '客户分群', '复购与流失', '数据管理', '模型实验']
LEGACY_PAGES = {'数据导入与清洗': '数据管理', '交易统计与可视化': '经营分析',
                'RFM用户价值分析': '客户价值', '用户聚类分析': '客户分群', '复购与流失预测': '复购与流失'}


def go(page, **updates):
    st.session_state['module'] = page
    st.session_state.update(updates)


def page_header(section, title, description):
    with st.container(key='page_heading'):
        st.markdown(f'<div class="header-meta"><div class="page-section">{escape(section)}</div>'
                    '<span class="page-badge">历史交易分析</span></div>', unsafe_allow_html=True)
        st.title(title)
        st.caption(description)


def import_guidance():
    st.info('还没有可用的交易数据。先导入文件，就可以查看经营情况和客户分析。')
    st.button('前往导入数据', key='page_import', type='primary', on_click=go, args=('数据管理',))


def chart(figure, **kwargs):
    original = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    recolor = dict(zip(original, PALETTE))
    for trace in figure.data:
        for attribute in ['line', 'marker']:
            part = getattr(trace, attribute, None)
            color = getattr(part, 'color', None)
            if isinstance(color, str) and color.upper() in recolor:
                part.color = recolor[color.upper()]
    figure.update_layout(template='plotly_white',
                         font={'family': 'Segoe UI, Microsoft YaHei, PingFang SC, sans-serif', 'color': INK, 'size': 11},
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         margin={'l': 12, 'r': 16, 't': 38, 'b': 24},
                         colorway=PALETTE, height=figure.layout.height or 380, bargap=.35,
                         legend_title_text='', legend={'font': {'size': 10}},
                         hoverlabel={'bgcolor': '#FFFFFF', 'font': {'color': INK, 'size': 12}, 'bordercolor': '#E6EAF3'})
    figure.update_xaxes(gridcolor='#EEF0F6', zeroline=False, tickfont={'color': '#909AAF', 'size': 10},
                       title_font={'size': 11, 'color': '#909AAF'})
    figure.update_yaxes(gridcolor='#EEF0F6', zeroline=False, tickfont={'color': '#909AAF', 'size': 10},
                       title_font={'size': 11, 'color': '#909AAF'})
    figure.update_traces(marker_cornerradius=4, selector={'type': 'bar'})
    st.plotly_chart(figure, width='stretch', config={'displayModeBar': False}, **kwargs)


def shell(root):
    st.markdown(styles(), unsafe_allow_html=True)
    current = st.session_state.get('module', '工作台')
    current = LEGACY_PAGES.get(current, current)
    if current not in PAGES:
        current = '工作台'
    st.session_state['module'] = current
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">' + icon('trend') + '</div>'
                    '<div><div class="brand-name">Retail Insights</div>'
                    '<div class="brand-sub">零售客户经营助手</div></div></div>'
                    '<div class="nav-section">工作空间</div>', unsafe_allow_html=True)
        selected = st.radio('工作空间', PAGES, key='module', label_visibility='collapsed', width='stretch')
        st.divider()
        st.caption('当前数据')
        pointer = root / 'data' / 'active.json'
        if pointer.exists():
            try:
                active = json.loads(pointer.read_text(encoding='utf-8'))
                st.markdown('<div class="source-status">交易数据已连接</div>', unsafe_allow_html=True)
                st.caption('历史数据演示 · 金额为英镑')
                with st.expander('数据来源与版本'):
                    st.caption(f"数据版本 {active['dataset_id'][:12]}")
                    st.caption('数据管理可查看来源文件和质量报告。')
            except (ValueError, KeyError, OSError):
                st.caption('数据连接异常，请到数据管理重新导入。')
        else:
            st.caption('尚未导入交易数据')
        st.markdown('<div class="sidebar-hint">从每一笔交易，了解每一位客户。</div>', unsafe_allow_html=True)
    return selected
