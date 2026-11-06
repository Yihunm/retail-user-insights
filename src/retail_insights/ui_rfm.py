"""以完整观察窗口评分，筛选仅作用于展示与导出。"""
import json
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from .preparation import load_dataset
from .rfm import build_rfm, rfm_profile, rfm_rules
from .rfm_presentation import display_rfm, display_profile, export_rfm
from .ui_shared import page_header, import_guidance, chart
from .ui_theme import SEGMENT_COLORS

WINDOWS = {'全部历史': None, '最近90天': 90, '最近180天': 180}
COLORS = SEGMENT_COLORS


@st.cache_data(show_spinner=False, max_entries=2)
def _sales(root, pointer):
    return load_dataset(root, json.loads(pointer)['dataset_id'])['sales']


@st.cache_data(show_spinner=False, max_entries=4)
def _calculate(root, pointer, reference, window):
    return build_rfm(_sales(root, pointer), reference_date=reference, window_days=window)


def _initialize(version, reference, segments):
    state = st.session_state
    defaults = {'rfm_reference': reference, 'rfm_window': '全部历史',
                'rfm_segments': segments, 'rfm_customer_search': ''}
    if state.get('rfm_version') != version:
        state['rfm_memory'] = defaults.copy()
        state['rfm_version'] = version
        for key, value in defaults.items():
            state[key] = value
    for key, value in defaults.items():
        state.setdefault(key, state.get('rfm_memory', {}).get(key, value))


def _remember(*keys):
    memory = dict(st.session_state.get('rfm_memory', {}))
    memory.update({key: st.session_state[key] for key in keys})
    st.session_state['rfm_memory'] = memory


def _chart(figure):
    figure.update_yaxes(tickformat=',.0f')
    chart(figure)


def _focus(segments):
    st.session_state['rfm_segments'] = segments
    st.session_state['rfm_customer_search'] = ''
    _remember('rfm_segments', 'rfm_customer_search')


def render(root):
    page_header('客户经营', '客户价值', '找到值得维护、可以培育和需要关注的客户。按购买间隔、订单数和消费金额进行规则分层。')
    pointer_file = root / 'data' / 'active.json'
    if not pointer_file.exists():
        import_guidance()
        return
    try:
        pointer = pointer_file.read_text(encoding='utf-8')
        sales = _sales(str(root), pointer)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    if sales.empty:
        st.info('当前没有有效销售，无法计算RFM。')
        return
    version = json.loads(pointer)['dataset_id']
    lo = sales.InvoiceDate.min().date()
    hi = (sales.InvoiceDate.max().normalize() + timedelta(days=1)).date()
    rules = rfm_rules()
    segments = rules['客户类型'].tolist()
    _initialize(version, hi, segments)
    with st.expander('调整分析范围'):
        date_box, window_box = st.columns(2)
        reference = date_box.date_input('分析参考日', min_value=lo, max_value=hi,
                                        format='YYYY/MM/DD', key='rfm_reference')
        window_name = window_box.selectbox('观察窗口', list(WINDOWS), key='rfm_window')
        st.caption('使用参考日之前的购买记录；更改日期和窗口会重新计算全体客户的分层。')
    _remember('rfm_reference', 'rfm_window')
    if reference is None:
        st.info('请选择分析参考日。')
        return
    window = WINDOWS[window_name]
    st.caption(f'分析参考日 {reference} · {window_name} · 仅分析有客户编号的有效购买')
    try:
        cohort = _calculate(str(root), pointer, reference.isoformat(), window)
    except ValueError as error:
        st.error(str(error))
        return
    if cohort.empty:
        st.info('当前观察窗口内没有可识别客户，请调整参考日或窗口。匿名交易不参与客户价值评分。')
        return
    all_box, value_box, risk_box = st.columns(3)
    all_box.button('查看全部客户', on_click=_focus, args=(segments,), width='stretch')
    value_box.button('重点维护', on_click=_focus, args=(['高价值客户'],), width='stretch')
    risk_box.button('关注流失', on_click=_focus, args=(['流失风险客户'],), width='stretch')
    group_box, search_box = st.columns([3, 2])
    selected_segments = group_box.multiselect('客户类型', segments, key='rfm_segments',
                                               placeholder='选择需要展示的客户类型')
    query = search_box.text_input('客户编号搜索（支持部分匹配）', key='rfm_customer_search',
                                  placeholder='输入编号，按回车搜索').strip()
    _remember('rfm_segments', 'rfm_customer_search')
    st.caption(f'分层基准为 {len(cohort):,} 名客户；以下筛选只改变展示和导出。')
    selected = cohort[cohort.Segment.isin(selected_segments)].copy()
    if query:
        selected = selected[selected.CustomerID.str.contains(query, regex=False, case=False)].copy()
    if selected.empty:
        st.info('当前客户筛选没有匹配结果，请调整客户类型或编号。')
        return
    boxes = st.columns(3)
    boxes[0].metric('当前筛选客户数', f'{len(selected):,}')
    amount = sum(int(value) for value in selected.MonetaryCents) / 100
    boxes[1].metric('当前筛选消费总额', f'£{amount:,.2f}')
    boxes[2].metric('其中高价值客户', f"{selected.Segment.eq('高价值客户').sum():,}")
    profile = display_profile(rfm_profile(selected))
    detail = display_rfm(selected)
    metadata = {'dataset_id': version, 'reference_date': reference.isoformat(), 'window_days': window,
                'cohort_customers': len(cohort), 'thresholds': cohort.attrs['thresholds'],
                'thresholds_exact': cohort.attrs.get('thresholds_exact', {}),
                'selected_segments': selected_segments, 'customer_search': query}
    suggestions = {'高价值客户': '可优先了解偏好并安排重点维护。', '流失风险客户': '曾有购买频次，但近期购买间隔较长，可核对后考虑回访。',
                   '沉睡客户': '近期购买较少，可结合实际业务判断是否需要重新触达。',
                   '潜力客户': '最近有购买，可关注后续复购和消费变化。', '一般客户': '可继续观察购买频次和消费金额的变化。'}
    if len(selected_segments) == 1:
        st.info(suggestions[selected_segments[0]] + ' 这是基于历史交易的规则判断，需结合实际客户情况。')
    distribution, customers, portrait, explanation = st.tabs(['客户分布', '客户名单', '消费特征', '分层规则'])
    with distribution:
        _chart(px.bar(profile, x='客户类型', y='客户数', color='客户类型',
                      color_discrete_map=COLORS, category_orders={'客户类型': segments}))
        st.dataframe(profile.style.format({'消费金额（英镑）': '{:,.2f}', '客户占比': '{:.1%}',
                                          '金额占比': '{:.1%}'}, na_rep='无数据'),
                     hide_index=True, width='stretch')
    with portrait:
        _chart(px.bar(profile, x='客户类型', y='消费金额（英镑）', color='客户类型',
                      color_discrete_map=COLORS, category_orders={'客户类型': segments}))
        _chart(px.scatter(detail, x='购买订单数', y='消费金额（英镑）', color='客户类型',
                          hover_data=['客户编号', '最近购买间隔（天）', 'R评分', 'F评分', 'M评分'],
                          color_discrete_map=COLORS, category_orders={'客户类型': segments}))
    with customers:
        st.download_button(f'导出客户名单（{len(selected):,}人）', export_rfm(selected, metadata),
                           file_name='RFM用户价值分析.zip', mime='application/zip', type='primary')
        columns = ['客户编号', '客户类型', '最近购买时间', '最近购买间隔（天）', '购买订单数', '消费金额（英镑）']
        st.dataframe(detail[columns].head(200), hide_index=True, width='stretch')
        st.caption('预览最多200名客户；完整筛选结果可在下方下载。')
    with explanation:
        st.caption(f'数据版本 {version[:12]} · 分层方法：RFM · 全体基准客户 {len(cohort):,} 人')
        st.markdown('''
**R（最近购买间隔）**：参考日与最近购买日期相差的自然天数，越小越好。

**F（购买频次）**：观察窗口内去重后的订单数，越大越好。

**M（消费金额）**：观察窗口内有效销售金额，越大越好。

以全体基准客户的20%、40%、60%、80%分位点计算1–5分，R反向计分。
同值同分；重复分位点可能跳过部分分值，不会随机拆分相同客户。
以下规则按顺序匹配，每位客户只归入一种类型。
''')
        thresholds = cohort.attrs['thresholds']
        score_table = pd.DataFrame([
            ['最近购买间隔（天）', *thresholds['R']], ['购买订单数', *thresholds['F']],
            ['消费金额（英镑）', *thresholds['M']],
        ], columns=['指标', '20%分位点', '40%分位点', '60%分位点', '80%分位点'])
        st.dataframe(score_table, hide_index=True, width='stretch')
        st.dataframe(rules, hide_index=True, width='stretch')
        with st.expander('查看客户评分明细'):
            st.dataframe(detail[['客户编号', 'R评分', 'F评分', 'M评分']].head(200),
                         hide_index=True, width='stretch')
        st.caption('“流失风险客户”是RFM规则标签，不代表已经预测出未来流失概率。分位点会随参考日和窗口变化。')
