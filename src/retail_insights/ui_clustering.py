"""显式训练的用户聚类分析页面。"""
import json
from datetime import timedelta
from time import perf_counter

import plotly.express as px
import streamlit as st

from .clustering import cluster_customers
from .clustering_presentation import (display_members, display_profile, display_scores,
                                      select_members, persist_clustering, export_clustering, group_name)
from .preparation import load_dataset
from .rfm import build_rfm

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
        for key in ['cluster_result', 'cluster_context', 'cluster_filter_memory']:
            state.pop(key, None)
        for key, value in defaults.items():
            state[key] = value
    for key, value in defaults.items():
        state.setdefault(key, state.get('cluster_parameter_memory', {}).get(key, value))


def _chart(figure, y_format=',.2f'):
    figure.update_layout(font={'family': 'Microsoft YaHei, PingFang SC, sans-serif'})
    figure.update_yaxes(tickformat=y_format)
    st.plotly_chart(figure, width='stretch', config={'displayModeBar': False})


def render(root):
    st.title('用户聚类分析')
    st.caption('根据交易特征自动发现客户群体；仅在点击“训练并分析”时更新模型。')
    pointer_file = root / 'data' / 'active.json'
    if not pointer_file.exists():
        st.info('请先在“数据导入与清洗”中导入数据。')
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
    with st.form('cluster_parameters'):
        date_box, window_box, k_box = st.columns(3)
        reference = date_box.date_input('分析参考日', min_value=lo, max_value=hi,
                                        format='YYYY/MM/DD', key='cluster_reference')
        window_name = window_box.selectbox('观察窗口', list(WINDOWS), key='cluster_window')
        k_mode = k_box.selectbox('聚类数量', list(K_OPTIONS), key='cluster_k_mode')
        st.caption('自动推荐比较2至8组内的可用候选，按轮廓系数选择；样本少或特征相同时候选数会减少。')
        submitted = st.form_submit_button('训练并分析', type='primary')
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
            training_status.success('训练完成，模型与训练信息已保存到本地。')
        except (ValueError, OSError) as error:
            training_status.error(f'训练未完成：{error}')
    if 'cluster_result' not in st.session_state:
        st.info('设置参考日、窗口和聚类数量后，点击“训练并分析”。')
        return
    result, context = st.session_state['cluster_result'], st.session_state['cluster_context']
    metadata = result['metadata']
    window_label = f"{context['window_days']}天" if context['window_days'] else '全部历史'
    st.caption(f"当前结果：参考日 {context['reference_date']}，窗口 "
               f"{window_label}，训练编号 {context['run_id'][:12]}。修改表单后需再次点击训练。")
    selected_score = result['scores'].loc[result['scores'].k.eq(metadata['selected_k']), 'Silhouette'].iloc[0]
    for box, label, value in zip(st.columns(4), ['模型客户数', '当前聚类数', '推荐聚类数', '当前轮廓系数'],
                                 [f"{len(result['customers']):,}", str(metadata['selected_k']),
                                  str(metadata['recommended_k']) if metadata['recommended_k'] else '无法推荐',
                                  '无法评价' if selected_score is None else f'{selected_score:.3f}']):
        box.metric(label, value)
    comparison, portrait, projection, details, methods = st.tabs(
        ['聚类数量比较', '群体画像', '二维分布', '客户明细', '方法说明'])
    profile = display_profile(result['profile'])
    scores = display_scores(result['scores'])
    colors = {group_name(group): px.colors.qualitative.Safe[index]
              for index, group in enumerate(sorted(result['customers'].Cluster.unique()))}
    with comparison:
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
        selected_groups = st.multiselect('聚类群体', groups, format_func=group_name,
                                         key='cluster_groups', placeholder='选择群体')
        query = st.text_input('客户编号搜索（支持部分匹配）', key='cluster_customer_search',
                              placeholder='输入编号，按回车搜索').strip()
        st.session_state['cluster_filter_memory'] = {'cluster_groups': selected_groups,
                                                    'cluster_customer_search': query}
        selected = select_members(result, selected_groups, query)
        st.caption('筛选只影响明细与导出客户清单；上方画像和数量评价始终基于模型全体客户，不会重新训练。')
        if selected.empty:
            st.info('当前筛选没有匹配客户，请调整群体或编号。')
        else:
            st.dataframe(display_members(selected).head(200), hide_index=True, width='stretch')
        st.caption(f'筛选匹配 {len(selected):,} 人，明细预览最多200人。')
    with methods:
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
    st.download_button('下载当前聚类分析', export_clustering(result, context, selected_groups, query),
                       file_name='用户聚类分析.zip', mime='application/zip')
