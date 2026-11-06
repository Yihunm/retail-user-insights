"""经营人员直接读取已有模型，筛选复购机会或待回访客户。"""
import json
from datetime import timedelta

import pandas as pd
import streamlit as st

from .prediction import infer
from .prediction_presentation import (TASK_LABELS, MODEL_LABELS, saved_records, load_experiment,
                                      display_predictions, export_customer_predictions)
from .ui_shared import go, page_header, import_guidance

TASK_CHOICES = {'复购机会': 'purchase', '未购买风险': 'churn'}


def render(root):
    from .ui_prediction import _sales

    page_header('客户经营', '复购与流失', '查看已有客户的复购机会或未购买风险，筛选后导出名单，用于人工核对和回访。')
    pointer_path = root / 'data' / 'active.json'
    if not pointer_path.exists():
        import_guidance()
        return
    try:
        pointer = pointer_path.read_text(encoding='utf-8')
        version = json.loads(pointer)['dataset_id']
        sales = _sales(str(root), pointer)
    except (ValueError, OSError, KeyError) as error:
        st.error(f'暂时无法读取客户数据：{error}')
        return
    if sales.empty:
        st.info('当前没有有效销售，请到数据管理检查交易文件。')
        return
    latest = (sales.InvoiceDate.max().normalize() + timedelta(days=1)).date()
    if st.session_state.get('prediction_version') != version:
        st.session_state['prediction_version'] = version
        for key in ['prediction_result', 'prediction_context', 'prediction_inference', 'opportunity_inference_key']:
            st.session_state.pop(key, None)
    st.session_state.setdefault('opportunity_task', st.session_state.get('opportunity_task_memory', '复购机会'))
    label = st.radio('查看方向', list(TASK_CHOICES), horizontal=True, key='opportunity_task')
    st.session_state['opportunity_task_memory'] = label
    task = TASK_CHOICES[label]
    results_slot = st.empty()
    marker = (version, task)
    if st.session_state.get('opportunity_render_task') != marker:
        results_slot.empty()
        st.session_state['opportunity_render_task'] = marker
    with results_slot.container():
        _render_task(root, sales, version, latest, task)


def _render_task(root, sales, version, latest, task):
    from .ui_prediction import _set_current

    records = [record for record in saved_records(root, version) if record['task'] == task]
    status = st.empty()
    if not records:
        with st.container(border=True):
            st.subheader('先准备一次预测分析')
            st.write(f'当前数据还没有“{TASK_LABELS[task]}”的模型。到模型实验选择对应任务，完成一次训练后，返回这里直接查看客户名单。')
            st.button('前往准备预测分析', key='prediction_setup', type='primary', on_click=go, args=('模型实验',),
                      kwargs={'prediction_task': TASK_LABELS[task], 'prediction_task_memory': TASK_LABELS[task],
                              'prediction_version': version})
        st.caption('首次训练需要完整的历史与未来记录；当前实验窗口适用于原始 Online Retail 数据。')
        return
    current = st.session_state.get('prediction_context', {})
    if current.get('run_id') != records[0]['run_id'] or current.get('dataset_id') != version:
        try:
            with status.container():
                with st.spinner('正在读取已保存的分析结果…'):
                    result, context = load_experiment(records[0]['artifact_dir'], version)
            _set_current(result, context, latest)
        except (ValueError, OSError, KeyError) as error:
            status.error(f'已有分析暂时无法读取：{error}。请到模型实验重新生成。')
            st.button('前往模型实验', key='prediction_setup', on_click=go, args=('模型实验',))
            return
    context = st.session_state['prediction_context']
    st.subheader(TASK_LABELS[task])
    st.caption('使用已保存模型直接生成名单，切换方向和筛选不会重新训练。')
    st.session_state.setdefault('prediction_infer_date', st.session_state.get('prediction_infer_date_memory', latest))
    with st.expander('调整预测日期'):
        with st.form('opportunity_date_form'):
            cutoff = st.date_input('预测截点', min_value=pd.Timestamp(context['available_from']).date(),
                                  max_value=latest, format='YYYY/MM/DD', key='prediction_infer_date')
            update = st.form_submit_button('更新客户名单')
        st.caption('只允许使用模型完成选择之后的日期；使用截点之前90天内购买过的客户。')
    if cutoff is None:
        st.info('请选择完整的预测日期。')
        return
    signature = (context['run_id'], cutoff.isoformat())
    if update or st.session_state.get('opportunity_inference_key') != signature:
        try:
            with status.container():
                with st.spinner('正在整理客户名单…'):
                    frame = infer(sales, context['artifact_dir'], cutoff=cutoff, expected_dataset_id=version)
            st.session_state['prediction_inference'] = frame
            st.session_state['prediction_infer_date_memory'] = cutoff
            st.session_state['opportunity_inference_key'] = signature
        except (ValueError, OSError) as error:
            status.error(f'客户名单暂时无法生成：{error}')
            return
    frame = st.session_state['prediction_inference']
    positive_count = int(frame.Prediction.eq(1).sum())
    horizon = context['horizon_days']
    st.caption(f'预测日期 {cutoff} · 查看未来{horizon}天，至 {cutoff + timedelta(days=horizon)}（不含当天）')
    positive_label = '预计复购客户' if task == 'purchase' else '待核对高风险客户'
    for box, title, value in zip(st.columns(3), ['分析客户数', positive_label, '其余客户'],
                                 [len(frame), positive_count, len(frame) - positive_count]):
        box.metric(title, f'{value:,}')
    st.info('可优先人工核对名单中的购买间隔与历史消费，再安排沟通。预测分数用于排序，不代表精确概率；未购买不等于永久流失。')
    st.subheader('客户名单')
    choices = ['全部客户', '预计复购', '预计不复购'] if task == 'purchase' else ['全部客户', '高风险', '一般风险']
    memory = st.session_state.get('opportunity_filter_memory', {}).get(task, '全部客户')
    if st.session_state.get('opportunity_result_filter') not in choices:
        st.session_state['opportunity_result_filter'] = memory
    st.session_state.setdefault('opportunity_result_filter', memory)
    st.session_state.setdefault('prediction_customer_search', st.session_state.get('prediction_search_memory', ''))
    filter_box, search_box = st.columns(2)
    result_filter = filter_box.selectbox('预测结果', choices, key='opportunity_result_filter')
    query = search_box.text_input('客户编号搜索（支持部分匹配）', key='prediction_customer_search',
                                 placeholder='输入客户编号后按回车').strip()
    st.session_state['prediction_search_memory'] = query
    st.session_state.setdefault('opportunity_filter_memory', {})[task] = result_filter
    selected = frame
    if result_filter != '全部客户':
        selected = selected[selected.Prediction.eq(int(result_filter == choices[1]))]
    if query:
        selected = selected[selected.CustomerID.str.contains(query, regex=False, case=False)]
    selected = selected.sort_values(['Probability', 'CustomerID'], ascending=[False, True])
    export_label = '导出复购名单' if task == 'purchase' else '导出风险名单'
    st.download_button(f'{export_label}（{len(selected):,}人）',
                       export_customer_predictions(selected, context, query, result_filter),
                       file_name='客户预测.zip', mime='application/zip', type='primary')
    st.caption(f'共 {len(frame):,} 名客户，当前筛选 {len(selected):,} 人 · 按预测分数从高到低排列 · 预览最多200人')
    if selected.empty:
        st.info('没有匹配客户，请调整预测结果或客户编号。')
    else:
        shown = display_predictions(selected, task)
        columns = ['客户编号', '预测判定', '预测分数', '距最近购买天数', '近90天订单数', '近90天金额（英镑）']
        st.dataframe(shown[columns].head(200), hide_index=True, width='stretch',
                     column_config={'预测分数': st.column_config.NumberColumn(format='%.3f'),
                                    '近90天金额（英镑）': st.column_config.NumberColumn(format='£%.2f')})
    with st.expander('预测依据与历史效果'):
        st.write('分析任务：' + TASK_LABELS[task])
        st.write('使用近90天订单数、消费金额、购买间隔等15项交易特征，不包含浏览或加购行为。')
        scores = context['test_scores']
        st.write(f"历史测试精确率 {scores['Precision']:.1%}，召回率 {scores['Recall']:.1%}。"
                 '前者表示预测为正例的客户中实际正例比例，后者表示实际正例中被识别的比例。')
        st.caption('历史测试日期2011-10-01；最新名单的未来结果尚不可验证。')
        st.caption(f"模型 {MODEL_LABELS[context['best_model']]} · 判定阈值 {context['threshold']:.4f} · "
                   f"训练编号 {context['run_id'][:12]} · 数据版本 {version[:12]}")
        st.button('查看完整模型实验', key='opportunity_experiments', on_click=go, args=('模型实验',))
