"""显式训练的用户聚类分析页面。"""
import json
from datetime import date, timedelta
from time import perf_counter

import plotly.express as px
import streamlit as st

from .clustering import cluster_customers
from .clustering_presentation import (display_members, display_profile, display_scores,
                                      select_members, persist_clustering, export_clustering, group_name, latest_clustering)
from .preparation import load_dataset
from .rfm import build_rfm
from .ui_shared import page_header, import_guidance, chart

WINDOWS = {'全部历史': None, '最近90天': 90, '最近180天': 180}
K_OPTIONS = {'自动推荐': None, **{f'{k}组': k for k in range(2, 9)}}


@st.cache_data(show_spinner=False, max_entries=2)
def _sales(root, pointer):
    return load_dataset(root, json.loads(pointer)['dataset_id'])['sales']


def _initialize(version, reference):
    defaults = {'cluster_reference': reference, 'cluster_window': '全部历史', 'cluster_k_mode': '自动推荐'}
    state = st.session_state
    if state.get('cluster_version') != version:
        state['cluster_version'] = version
        state['cluster_parameter_memory'] = defaults.copy()
        for key in ['cluster_result', 'cluster_context', 'cluster_filter_memory', 'cluster_saved_checked',
                    'cluster_groups', 'cluster_customer_search']:
            state.pop(key, None)
        for key, value in defaults.items():
            state[key] = value
    for key, value in defaults.items():
        state.setdefault(key, state.get('cluster_parameter_memory', {}).get(key, value))


def _chart(figure, y_format=',.2f'):
    figure.update_yaxes(tickformat=y_format)
    chart(figure)


def render(root):
    page_header('客户经营', '客户分群', '了解不同客户的购买特征，为不同群体安排沟通内容。先看群体画像，再筛选和导出客户名单。')
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
        st.info('当前没有有效销售，无法进行用户聚类。')
        return
    version = json.loads(pointer)['dataset_id']
    lo = sales.InvoiceDate.min().date()
    hi = (sales.InvoiceDate.max().normalize() + timedelta(days=1)).date()
    _initialize(version, hi)
    if not st.session_state.get('cluster_saved_checked'):
        saved = latest_clustering(root, version)
        if saved is not None:
            result, context = saved
            st.session_state['cluster_result'], st.session_state['cluster_context'] = result, context
            defaults = {'cluster_reference': date.fromisoformat(context['reference_date'][:10]),
                        'cluster_window': next(name for name, days in WINDOWS.items() if days == context['window_days']),
                        'cluster_k_mode': '自动推荐' if context['requested_k'] is None else f"{context['requested_k']}组"}
            st.session_state.update(defaults)
            st.session_state['cluster_parameter_memory'] = defaults
        st.session_state['cluster_saved_checked'] = True
    has_result = 'cluster_result' in st.session_state
    if not has_result:
        st.info('还没有客户分群结果。使用默认设置生成一次，之后可直接查看已保存的画像和名单。')
    with st.expander('调整分群设置', expanded=not has_result):
        with st.form('cluster_parameters'):
            date_box, window_box, k_box = st.columns(3)
            reference = date_box.date_input('分析参考日', min_value=lo, max_value=hi,
                                            format='YYYY/MM/DD', key='cluster_reference')
            window_name = window_box.selectbox('观察窗口', list(WINDOWS), key='cluster_window')
            k_mode = k_box.selectbox('客户分组数量', list(K_OPTIONS), key='cluster_k_mode')
            st.caption('默认自动选择分组数量。只有点击生成或更新时，才会重新分析客户。')
            submitted = st.form_submit_button('更新客户分群' if has_result else '生成客户分群', type='primary')
    training_status = st.empty()
    if submitted:
        st.session_state['cluster_parameter_memory'] = {
            'cluster_reference': reference, 'cluster_window': window_name, 'cluster_k_mode': k_mode}
        if reference is None:
            training_status.error('请选择分析参考日。')
            return
        try:
            started = perf_counter()
            with training_status.container():
                with st.spinner('正在提取客户特征、比较聚类数量并训练模型。'):
                    rfm = build_rfm(sales, reference_date=reference, window_days=WINDOWS[window_name])
                    result = cluster_customers(rfm, k=K_OPTIONS[k_mode])
                    context = persist_clustering(result, {'dataset_id': version,
                        'reference_date': reference.isoformat(), 'window_days': WINDOWS[window_name],
                        'requested_k': K_OPTIONS[k_mode], 'elapsed_seconds': perf_counter() - started}, root)
            st.session_state['cluster_result'] = result
            st.session_state['cluster_context'] = context
            st.session_state['cluster_groups'] = sorted(result['customers'].Cluster.unique().tolist())
            st.session_state['cluster_customer_search'] = ''
            st.session_state['cluster_filter_memory'] = {'cluster_groups': st.session_state['cluster_groups'],
                                                         'cluster_customer_search': ''}
            training_status.success('客户分群已更新，画像和名单已保存。')
        except (ValueError, OSError) as error:
            training_status.error(f'分群未完成：{error}')
    if 'cluster_result' not in st.session_state:
        return
    result, context = st.session_state['cluster_result'], st.session_state['cluster_context']
    metadata = result['metadata']
    window_label = f"{context['window_days']}天" if context['window_days'] else '全部历史'
    st.caption(f"当前结果：参考日 {context['reference_date']} · {window_label} · 使用已保存分析；修改设置后需点击更新。")
    selected_score = result['scores'].loc[result['scores'].k.eq(metadata['selected_k']), 'Silhouette'].iloc[0]
    for box, label, value in zip(st.columns(4), ['分析客户数', '客户群体数', '人均购买订单', '人均消费金额'],
                                 [f"{len(result['customers']):,}", str(metadata['selected_k']),
                                  f"{result['customers'].Frequency.mean():.1f}",
                                  f"£{result['customers'].Monetary.mean():,.2f}"]):
        box.metric(label, value)
    profile_raw = result['profile']
    top = profile_raw.loc[profile_raw.MeanMonetary.idxmax()]
    st.write(f"**{group_name(top.Cluster)}** 的人均消费金额最高，为 **£{top.MeanMonetary:,.2f}**。"
             '可先查看该群体的消费频次，再决定维护方式。群体编号只作区分，不表示价值排名。')
    portrait, details, projection, comparison, methods = st.tabs(
        ['群体画像', '客户名单', '特征分布', '分群效果', '方法与版本'])
    profile = display_profile(result['profile'])
    scores = display_scores(result['scores'])
    colors = {group_name(group): px.colors.qualitative.Safe[index]
              for index, group in enumerate(sorted(result['customers'].Cluster.unique()))}
    with comparison:
        st.caption(f"当前轮廓系数：{'无法评价' if selected_score is None else f'{selected_score:.3f}'}；"
                   f"推荐分组数：{metadata['recommended_k']}。该指标评价群体分离程度，不是预测准确率。")
        _chart(px.line(scores, x='聚类数量', y='轮廓系数', markers=True), y_format='.3f')
        _chart(px.line(scores, x='聚类数量', y='簇内平方和', markers=True))
        st.dataframe(scores, hide_index=True, width='stretch')
        st.caption('轮廓系数越大通常表示群体分离越清晰；推荐值仍需结合业务解释。各窗口之间的分数不可直接等同。')
    with portrait:
        _chart(px.bar(profile, x='聚类群体', y='客户数', color='聚类群体', color_discrete_map=colors), y_format=',.0f')
        _chart(px.bar(profile, x='聚类群体', y='人均消费金额（英镑）', color='聚类群体', color_discrete_map=colors))
        st.dataframe(profile.style.format({'客户占比': '{:.1%}', '消费总额（英镑）': '{:,.2f}',
                                          '平均最近购买间隔（天）': '{:.1f}', '平均购买订单数': '{:.2f}',
                                          '人均消费金额（英镑）': '{:,.2f}', '消费金额中位数（英镑）': '{:,.2f}',
                                          '平均购买数量': '{:,.2f}', '平均客单金额（英镑）': '{:,.2f}'}),
                     hide_index=True, width='stretch')
    with projection:
        members = display_members(result['customers'])
        ratios = metadata['pca_explained_variance_ratio']
        figure = px.scatter(members, x='主成分1', y='主成分2', color='聚类群体',
                            hover_data=['客户编号', '最近购买间隔（天）', '购买订单数', '消费金额（英镑）'],
                            color_discrete_map=colors, category_orders={'聚类群体': list(colors)})
        figure.update_xaxes(title_text=f'主成分1（解释方差 {ratios[0]:.1%}）', tickformat=',.2f')
        figure.update_yaxes(title_text=f'主成分2（解释方差 {ratios[1]:.1%}）')
        _chart(figure)
        st.caption('二维位置仅用于展示。群体编号不代表价值高低，改变窗口或聚类数量可能改变编号。')
    with details:
        groups = sorted(result['customers'].Cluster.unique().tolist())
        memory = st.session_state.get('cluster_filter_memory', {})
        st.session_state.setdefault('cluster_groups', memory.get('cluster_groups', groups))
        st.session_state.setdefault('cluster_customer_search', memory.get('cluster_customer_search', ''))
        selected_groups = st.multiselect('客户群体', groups, format_func=group_name,
                                         key='cluster_groups', placeholder='选择群体')
        query = st.text_input('客户编号搜索（支持部分匹配）', key='cluster_customer_search',
                              placeholder='输入编号，按回车搜索').strip()
        st.session_state['cluster_filter_memory'] = {'cluster_groups': selected_groups,
                                                    'cluster_customer_search': query}
        selected = select_members(result, selected_groups, query)
        st.download_button(f'导出客户名单（{len(selected):,}人）',
                           export_clustering(result, context, selected_groups, query),
                           file_name='用户聚类分析.zip', mime='application/zip', type='primary')
        st.caption('筛选只影响明细与导出客户清单；上方画像和数量评价始终基于模型全体客户，不会重新训练。')
        if selected.empty:
            st.info('当前筛选没有匹配客户，请调整群体或编号。')
        else:
            columns = ['客户编号', '聚类群体', '最近购买间隔（天）', '购买订单数', '消费金额（英镑）', '平均订单金额（英镑）']
            st.dataframe(display_members(selected)[columns].head(200), hide_index=True, width='stretch')
        st.caption(f'筛选匹配 {len(selected):,} 人，明细预览最多200人。')
    with methods:
        st.caption(f"训练编号 {context['run_id'][:12]} · 数据版本 {version[:12]}")
        st.markdown('''
使用最近购买间隔、订单数、消费金额、购买数量和平均订单金额五项原始指标。
先进行对数变换（log1p），删除无方差特征，再标准化后训练K-Means。
固定随机种子42，每个候选初始化20次。没有浏览、收藏或加购特征。

PCA仅用于展示，聚类并未只使用二维坐标，也没有使用RFM规则标签作为输入。
''')
        used = {'Recency': '最近购买间隔', 'Frequency': '订单数', 'Monetary': '消费金额',
                'Quantity': '购买数量', 'AverageOrderValue': '平均订单金额'}
        st.write('有效特征：' + '、'.join(used[name] for name in metadata['effective_features']))
        st.caption(f"轮廓系数最多评价 {metadata['silhouette_sample_size']:,} 名客户，抽样时确保每个群体至少有一人。"
                   '这是无监督分群，没有客户真实类别标签，不以分类准确率评价。')
