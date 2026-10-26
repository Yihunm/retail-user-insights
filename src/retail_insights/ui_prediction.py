"""时间隔离的实验评价与显式客户推断。"""
import json
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from .prediction import infer
from .preparation import load_dataset
from .prediction_presentation import (TASK_LABELS, MODEL_LABELS, STRATEGY_LABELS, FEATURE_LABELS,
    display_metrics, display_predictions, run_prediction, saved_records, load_experiment,
    export_customer_predictions)


@st.cache_data(show_spinner=False, max_entries=2)
def _sales(root, pointer):
    return load_dataset(root, json.loads(pointer)['dataset_id'])['sales']


def _set_current(result, context, latest):
    st.session_state['prediction_result'] = result
    st.session_state['prediction_context'] = context
    st.session_state.pop('prediction_inference', None)
    st.session_state['prediction_infer_date'] = latest
    st.session_state['prediction_infer_date_memory'] = latest
    st.session_state['prediction_customer_search'] = ''
    st.session_state['prediction_search_memory'] = ''


def _chart(figure):
    figure.update_layout(font={'family': 'Microsoft YaHei, PingFang SC, sans-serif'})
    st.plotly_chart(figure, width='stretch', config={'displayModeBar': False})


def render(root):
    st.title('复购与流失预测')
    st.caption('过去90天交易特征 → 未来30天复购 / 未来60天未购买风险。仅预测已有购买记录的客户。')
    pointer_path = root / 'data' / 'active.json'
    if not pointer_path.exists():
        st.info('请先在“数据导入与清洗”中导入数据。')
        return
    try:
        pointer = pointer_path.read_text(encoding='utf-8')
        sales = _sales(str(root), pointer)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    if sales.empty:
        st.info('当前没有有效销售，无法构建预测样本。')
        return
    version = json.loads(pointer)['dataset_id']
    latest = (sales.InvoiceDate.max().normalize() + timedelta(days=1)).date()
    if st.session_state.get('prediction_version') != version:
        st.session_state['prediction_version'] = version
        for key in ['prediction_result', 'prediction_context', 'prediction_inference', 'prediction_saved_run']:
            st.session_state.pop(key, None)
        st.session_state['prediction_task'] = TASK_LABELS['purchase']
        st.session_state['prediction_task_memory'] = TASK_LABELS['purchase']
        st.session_state['prediction_search_memory'] = ''
    st.session_state.setdefault('prediction_task', st.session_state.get('prediction_task_memory', TASK_LABELS['purchase']))
    with st.form('prediction_training'):
        task_label = st.selectbox('训练任务', list(TASK_LABELS.values()), key='prediction_task')
        st.caption('比较逻辑回归、决策树、随机森林，以及不处理、欠采样、随机过采样、SMOTE和类别权重，另含先验基线。')
        submitted = st.form_submit_button('构建样本并训练', type='primary')
    training_status = st.empty()
    if submitted:
        task = next(key for key, value in TASK_LABELS.items() if value == task_label)
        st.session_state['prediction_task_memory'] = task_label
        try:
            with training_status.container():
                with st.spinner('正在构建时间样本并进行前向验证、模型对比。全量数据可能需要一至数分钟。'):
                    result, context = run_prediction(sales, task, root, version)
            _set_current(result, context, latest)
            training_status.success('训练完成，测试结果与模型已保存。')
        except (ValueError, OSError) as error:
            training_status.error(f'训练未完成：{error}')
    with st.expander('加载当前数据的已保存模型'):
        records = saved_records(root, version)
        if records:
            choices = {row['run_id']: row for row in records}
            if st.session_state.get('prediction_saved_run') not in choices:
                st.session_state['prediction_saved_run'] = records[0]['run_id']
            selected_run = st.selectbox('已保存训练记录', list(choices), key='prediction_saved_run',
                format_func=lambda value: TASK_LABELS[choices[value]['task']] + ' · ' +
                MODEL_LABELS.get(choices[value]['best_model'], '模型') + ' · ' + value[:12])
            if st.button('加载已保存模型'):
                try:
                    result, context = load_experiment(choices[selected_run]['artifact_dir'], version)
                    _set_current(result, context, latest)
                    training_status.success('已加载本地模型，不重新训练。')
                except (ValueError, OSError, KeyError) as error:
                    training_status.error(f'加载失败：{error}')
        else:
            st.info('当前数据还没有完整的预测模型记录。')
    if 'prediction_result' not in st.session_state:
        st.info('请选择任务并点击“构建样本并训练”，或加载已保存模型。需要数据覆盖本项目固定的历史训练和测试窗口。')
        return
    result, context = st.session_state['prediction_result'], st.session_state['prediction_context']
    task = context['task']
    st.subheader('当前结果：' + TASK_LABELS[task])
    st.caption(f"训练编号 {context['run_id'][:12]} · 数据版本 {version[:12]} · 当前结果以此处任务为准，训练表单用于下一次训练。")
    st.info('模型输出分数未校准，用于排序和阈值判定，不代表精确发生概率。未购买也不等于永久流失。')
    model_box, strategy_box, threshold_box = st.columns(3)
    model_box.metric('最佳模型', MODEL_LABELS[context['best_model']])
    strategy_box.metric('不平衡处理', STRATEGY_LABELS[context['selected_strategy']])
    threshold_box.metric('验证集选定阈值', f"{context['threshold']:.4f}")
    experiments, evaluation, customers, methods = st.tabs(['模型与不平衡对比', '历史测试结果', '客户预测', '时间窗口与方法'])
    with experiments:
        validation = result['metrics'][result['metrics'].Split.eq('validation')]
        shown = display_metrics(validation)
        _chart(px.bar(shown[shown['状态'].eq('已完成')], x='模型', y='平均精确率（AP）', color='不平衡处理',
                      barmode='group'))
        st.dataframe(shown, hide_index=True, width='stretch')
        st.caption('模型及处理方法以验证集AP优先、F1次之选择；每个候选先在较早训练截点内部进行有限前向调参。')
        st.download_button('下载完整模型对比表', display_metrics(result['metrics']).to_csv(index=False).encode('utf-8-sig'),
                           file_name='模型对比.csv', mime='text/csv')
    with evaluation:
        scores = context['test_scores']
        metric_boxes = st.columns(3) + st.columns(3)
        for box, name, label in zip(metric_boxes, ['AveragePrecision', 'F1', 'ROC_AUC', 'Accuracy', 'Precision', 'Recall'],
                                    ['平均精确率（AP）', 'F1值', 'ROC曲线下面积', '准确率', '精确率', '召回率']):
            value = scores[name]
            box.metric(label, '无法计算' if value is None else f'{value:.4f}')
        positive = '未来30天有复购' if task == 'purchase' else '未来60天未购买'
        st.caption(f'最终测试截点为2011-10-01，正例表示“{positive}”。测试集保持原始类别分布，仅用于冻结模型后的评价。')
        roc, pr = result['curves']['roc'], result['curves']['pr']
        if not roc.empty:
            _chart(px.line(roc.rename(columns={'FalsePositiveRate': '假阳性率', 'TruePositiveRate': '真阳性率'}),
                           x='假阳性率', y='真阳性率', title='ROC曲线'))
        if not pr.empty:
            _chart(px.line(pr.rename(columns={'Recall': '召回率', 'Precision': '精确率'}),
                           x='召回率', y='精确率', title='精确率与召回率曲线'))
        _chart(px.imshow(context['confusion_matrix'], x=['预测负例', '预测正例'], y=['实际负例', '实际正例'],
                         text_auto=True, labels={'x': '预测标签', 'y': '实际标签', 'color': '客户数'},
                         color_continuous_scale='Blues', aspect='auto'))
        importance = result['importance'].copy()
        importance['Feature'] = importance.Feature.map(FEATURE_LABELS)
        importance = importance.rename(columns={'Feature': '特征', 'Importance': '重要性'}).sort_values('重要性')
        _chart(px.bar(importance, x='重要性', y='特征', orientation='h', title='全局特征重要性'))
        st.caption('逻辑回归展示标准化系数绝对值，树模型展示纯度重要性；这是全局关联，不是单个客户的因果解释。')
        st.dataframe(display_predictions(result['predictions'], task).head(200), hide_index=True, width='stretch')
    with customers:
        lower = pd.Timestamp(context['available_from']).date()
        st.session_state.setdefault('prediction_infer_date', st.session_state.get('prediction_infer_date_memory', latest))
        with st.form('prediction_inference_form'):
            cutoff = st.date_input('预测截点', min_value=lower, max_value=latest,
                                    key='prediction_infer_date', format='YYYY/MM/DD')
            infer_submitted = st.form_submit_button('生成客户预测')
        st.caption('仅允许模型选择所需标签已完整可见之后的截点，避免倒用未来信息。')
        inference_status = st.empty()
        if infer_submitted:
            try:
                if cutoff is None:
                    raise ValueError('请选择预测截点。')
                with inference_status.container():
                    with st.spinner('正在生成客户特征并使用已保存模型预测。'):
                        prediction = infer(sales, context['artifact_dir'], cutoff=cutoff, expected_dataset_id=version)
                st.session_state['prediction_inference'] = prediction
                st.session_state['prediction_infer_date_memory'] = cutoff
                st.session_state['prediction_customer_search'] = ''
                st.session_state['prediction_search_memory'] = ''
            except (ValueError, OSError) as error:
                inference_status.error(f'预测未完成：{error}')
        if 'prediction_inference' in st.session_state:
            frame = st.session_state['prediction_inference']
            st.session_state.setdefault('prediction_customer_search', st.session_state.get('prediction_search_memory', ''))
            query = st.text_input('客户编号搜索（支持部分匹配）', key='prediction_customer_search',
                                  placeholder='输入编号，按回车搜索').strip()
            st.session_state['prediction_search_memory'] = query
            selected = frame if not query else frame[frame.CustomerID.str.contains(query, regex=False, case=False)]
            date = pd.Timestamp(frame.Cutoff.iloc[0]).date()
            st.caption(f"预测截点 {date}，目标窗口至 {date + timedelta(days=context['horizon_days'])}（不含右端）。"
                       f'共 {len(frame):,} 名客户，当前筛选 {len(selected):,} 人；筛选不重新训练。')
            if selected.empty:
                st.info('没有匹配客户。')
            else:
                st.dataframe(display_predictions(selected, task).head(200), hide_index=True, width='stretch')
            st.download_button('下载当前客户预测', export_customer_predictions(selected, context, query),
                               file_name='客户预测.zip', mime='application/zip')
            st.caption('最新截点之后没有完整未来记录时，这些预测尚无法核对，不包含真实标签。')
        else:
            st.info('选择截点并点击“生成客户预测”。')
    with methods:
        summary = result['sample_summary'].copy()
        if not summary.empty:
            summary['Split'] = summary.Split.map({'train': '训练集', 'validation': '验证集', 'test': '最终测试集'})
            st.dataframe(summary.rename(columns={'Cutoff': '历史截点', 'Rows': '样本数', 'PositiveCount': '正例数',
                                                 'PositiveRate': '正例比例', 'Split': '数据划分'}),
                         hide_index=True, width='stretch')
        st.markdown('''
观察历史为截点之前90天，标签为截点之后30天复购或60天未购买。
历史与未来均使用左闭右开区间；开头和末尾不完整日保守排除。

训练截点为2011年3月2日、4月1日、5月1日、6月1日；8月1日用于模型和阈值选择；10月1日为最终测试。
训练内部前向验证使用“3月→5月”“3/4月→6月”，并检查未来标签已在验证截点前可见。

缺失填补、标准化、欠采样、随机过采样与SMOTE都只在各训练折内拟合。
验证/测试不采样。类别权重只调整训练损失。两种任务按实际标签比例比较，不人为制造不平衡。

阈值按验证集F1最大化选择，同分选较高阈值。最终测试仅评价已选模型；选择后不合并验证集重新训练。
同一客户可出现在不同时间截点，客户编号不参与训练。本实验不代表对从未购买的新用户的预测效果。
''')
        st.dataframe(pd.DataFrame({'特征': list(FEATURE_LABELS.values())}), hide_index=True, width='stretch')
