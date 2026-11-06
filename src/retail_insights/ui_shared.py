"""面向经营人员的导航、页面标题与统一图表样式。"""
import json
from html import escape

import streamlit as st

PAGES = ['工作台', '经营分析', '客户价值', '客户分群', '复购与流失', '数据管理', '模型实验']
LEGACY_PAGES = {'数据导入与清洗': '数据管理', '交易统计与可视化': '经营分析',
                'RFM用户价值分析': '客户价值', '用户聚类分析': '客户分群', '复购与流失预测': '复购与流失'}
TEAL = '#007F78'
NAVY = '#173B52'


def go(page, **updates):
    st.session_state['module'] = page
    st.session_state.update(updates)


def page_header(section, title, description):
    st.markdown(f'<div class="page-section">{escape(section)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def import_guidance():
    st.info('还没有可用的交易数据。先导入文件，就可以查看经营情况和客户分析。')
    st.button('前往导入数据', key='page_import', type='primary', on_click=go, args=('数据管理',))


def chart(figure, **kwargs):
    figure.update_layout(font={'family': 'Microsoft YaHei, PingFang SC, sans-serif', 'color': NAVY},
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         margin={'l': 12, 'r': 16, 't': 44, 'b': 24},
                         colorway=[TEAL, '#487BA4', '#C58542', '#8587B2', '#697F91'],
                         legend_title_text='')
    figure.update_xaxes(gridcolor='#E8EEF2', zeroline=False)
    figure.update_yaxes(gridcolor='#E8EEF2', zeroline=False)
    st.plotly_chart(figure, width='stretch', config={'displayModeBar': False}, **kwargs)


def shell(root):
    st.markdown('''<style>
    .stApp {background:#F3F6F8; color:#172F40;}
    .stMainBlockContainer {max-width:1380px; padding-top:4rem; padding-bottom:3rem;}
    [data-testid="stAppDeployButton"] {display:none;}
    h1,h2,h3 {font-family:"Microsoft YaHei","PingFang SC",sans-serif; letter-spacing:-.025em;}
    h1 {font-size:2.1rem!important; font-weight:700!important; margin-bottom:.25rem!important;}
    h2 {font-size:1.4rem!important;} h3 {font-size:1.12rem!important;}
    .page-section {color:#007F78;font-size:.76rem;font-weight:700;letter-spacing:.12em;margin-bottom:.3rem;}
    [data-testid="stSidebar"] {background:#FFF; border-right:1px solid #DFE7EC;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {padding:.38rem .2rem;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {background:#E8F5F1;border-radius:8px;}
    [data-testid="stMetric"] {background:#FFF;border:1px solid #E2E9EE;border-radius:12px;padding:1rem 1.15rem;}
    [data-testid="stMetricValue"] {font-family:"Segoe UI",sans-serif;font-variant-numeric:tabular-nums;font-size:1.85rem;}
    [data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#E0E8ED!important;border-radius:12px!important;}
    [data-testid="stForm"], [data-testid="stExpander"] {background:#FFF;}
    .stButton>button,.stDownloadButton>button {border-radius:8px;font-weight:600;}
    button:focus-visible {outline:3px solid #49A99F!important;outline-offset:2px;}
    [data-testid="stTabs"] [role="tablist"] {gap:1.25rem;border-bottom:1px solid #DCE5EB;}
    .brand {font-size:1.28rem;font-weight:750;color:#173B52;margin-bottom:.15rem;}
    .brand span {color:#007F78;} .brand-sub {font-size:.78rem;color:#6B8190;margin-bottom:1.6rem;}
    .workflow {border-left:4px solid #007F78;padding:.45rem 1rem;margin:.6rem 0 1.2rem;background:#E8F5F1;}
    .st-key-home_value_card,.st-key-home_risk_card,.st-key-home_group_card,
    .st-key-home_trend_card,.st-key-home_distribution_card {background:#FFF;}
    @media(max-width:700px) {
      .stMainBlockContainer {padding:3.7rem 1rem 2rem;} h1 {font-size:1.65rem!important;}
      [data-testid="stMetricValue"] {font-size:1.5rem;}
    }
    @media(prefers-reduced-motion:reduce) {* {scroll-behavior:auto!important;}}
    </style>''', unsafe_allow_html=True)
    current = st.session_state.get('module', '工作台')
    current = LEGACY_PAGES.get(current, current)
    if current not in PAGES:
        current = '工作台'
    st.session_state['module'] = current
    with st.sidebar:
        st.markdown('<div class="brand">Retail <span>Insights</span></div>'
                    '<div class="brand-sub">零售客户经营助手</div>', unsafe_allow_html=True)
        selected = st.radio('工作空间', PAGES, key='module', label_visibility='collapsed')
        st.divider()
        st.caption('当前数据')
        pointer = root / 'data' / 'active.json'
        if pointer.exists():
            try:
                active = json.loads(pointer.read_text(encoding='utf-8'))
                st.write('已连接交易数据')
                st.caption('历史数据演示 · 金额为英镑')
                with st.expander('数据来源与版本'):
                    st.caption(f"数据版本 {active['dataset_id'][:12]}")
                    st.caption('数据管理可查看来源文件和质量报告。')
            except (ValueError, KeyError, OSError):
                st.caption('数据连接异常，请到数据管理重新导入。')
        else:
            st.caption('尚未导入交易数据')
        st.caption('先看经营情况，再找到值得关注的客户。')
    return selected
