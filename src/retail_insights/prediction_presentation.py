"""中文预测报告、项目自产模型记录和可追溯导出。"""
import io
import hashlib
import json
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd

from .prediction import train_experiment, read_artifact, TRAIN_CUTOFFS, FEATURES, FEATURE_VERSION
from .prediction_features import make_samples

TASK_LABELS = {'purchase': '未来30天复购', 'churn': '未来60天未购买风险'}
MODEL_LABELS = {'baseline': '先验基线', 'logistic': '逻辑回归', 'tree': '决策树', 'forest': '随机森林'}
STRATEGY_LABELS = {'none': '不处理', 'under': '欠采样', 'over': '随机过采样', 'smote': 'SMOTE过采样',
                   'weight': '类别权重', 'baseline': '基线'}
FEATURE_LABELS = {
    'recency_days': '距最近购买天数', 'orders_7': '近7天订单数', 'orders_30': '近30天订单数',
    'orders_90': '近90天订单数', 'revenue_7': '近7天金额（英镑）', 'revenue_30': '近30天金额（英镑）',
    'revenue_90': '近90天金额（英镑）', 'quantity_90': '近90天购买数量',
    'unique_products_90': '近90天不同商品数', 'active_days_90': '近90天购买活跃天数',
    'average_order_value_90': '近90天平均订单金额', 'max_order_value_90': '近90天最大订单金额',
    'interval_mean_days': '平均购买间隔（天）', 'interval_missing': '购买间隔缺失标记',
    'history_span_days': '窗口内购买跨度（天）',
}
METRIC_LABELS = {'Candidate': '候选方案', 'Model': '模型', 'Strategy': '不平衡处理', 'Split': '数据划分',
                 'Threshold': '判定阈值', 'AveragePrecision': '平均精确率（AP）', 'Accuracy': '准确率',
                 'Precision': '精确率', 'Recall': '召回率', 'F1': 'F1值', 'ROC_AUC': 'ROC曲线下面积',
                 'TrainRows': '训练样本数', 'TrainPositiveRate': '训练正例比例', 'InnerCV_AP': '前向验证平均AP',
                 'Params': '模型参数', 'Status': '状态', 'Reason': '说明', 'InnerFolds': '有效前向折数'}


def sample_summary(samples):
    table = samples.groupby('Cutoff', sort=True).Target.agg(Rows='size', PositiveCount='sum', PositiveRate='mean').reset_index()
    table['Split'] = table.Cutoff.map(lambda value: 'train' if str(value)[:10] in TRAIN_CUTOFFS
                                     else ('validation' if str(value)[:10] == '2011-08-01' else 'test'))
    return table


def display_metrics(frame):
    shown = frame.copy()
    shown['Candidate'] = shown.apply(lambda row: MODEL_LABELS[row.Model] + ' / ' + STRATEGY_LABELS[row.Strategy], axis=1)
    shown['Model'] = shown.Model.map(MODEL_LABELS)
    shown['Strategy'] = shown.Strategy.map(STRATEGY_LABELS)
    shown['Split'] = shown.Split.map({'validation': '验证集', 'test': '最终测试集'})
    shown['Status'] = shown.Status.map({'ok': '已完成', 'skipped': '已跳过'})
    shown['Reason'] = shown.Reason.fillna('').replace({
        'SMOTE needs at least two minority training samples': '训练少数类不足2条，跳过SMOTE',
        'No valid forward folds; first configuration used without tuning': '没有可用前向折，使用首组配置，未调参'})
    return shown.rename(columns=METRIC_LABELS)


def display_predictions(frame, task):
    shown = frame.copy()
    if 'Cutoff' in shown:
        shown['Cutoff'] = pd.to_datetime(shown.Cutoff).dt.strftime('%Y-%m-%d')
    if 'Prediction' in shown:
        shown['Prediction'] = shown.Prediction.map(
            {0: '预计不复购', 1: '预计复购'} if task == 'purchase' else {0: '一般风险', 1: '高风险'})
    if 'Target' in shown:
        shown['Target'] = shown.Target.map({0: '无复购', 1: '有复购'} if task == 'purchase'
                                           else {0: '有购买', 1: '未购买'})
    return shown.rename(columns={**FEATURE_LABELS, 'CustomerID': '客户编号', 'Cutoff': '预测截点',
                                  'Probability': '预测分数', 'Prediction': '预测判定', 'Target': '实际结果'})


def run_prediction(sales, task, root, dataset_id):
    started = perf_counter()
    samples = make_samples(sales, task=task)
    run_id = uuid4().hex
    folder = Path(root).resolve() / 'artifacts' / 'prediction' / run_id
    result = train_experiment(samples, task, folder, dataset_id=dataset_id)
    summary = sample_summary(samples)
    result['metadata'].update(run_id=run_id, elapsed_seconds=perf_counter() - started,
                              sample_summary=summary.to_dict('records'))
    samples.to_parquet(folder / 'samples.parquet', index=False)
    encoded = json.dumps(result['metadata'], ensure_ascii=False, indent=2, allow_nan=False)
    (folder / 'metadata.json').write_text(encoded, encoding='utf-8')
    with closing(sqlite3.connect(Path(root) / 'data' / 'catalog.sqlite3')) as db:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS prediction_runs '
                       '(id TEXT PRIMARY KEY, dataset_id TEXT, task TEXT, created_at TEXT, metadata TEXT)')
            db.execute('INSERT INTO prediction_runs VALUES (?,?,?,?,?)',
                       (run_id, dataset_id, task, result['metadata']['created_at'], encoded))
    _write_complete_marker(folder, run_id)
    result['sample_summary'] = summary
    return result, {**result['metadata'], 'artifact_dir': str(folder)}


def _write_complete_marker(folder, run_id):
    digest = hashlib.sha256((folder / 'metadata.json').read_bytes()).hexdigest()
    (folder / 'COMPLETE.json').write_text(json.dumps({'run_id': run_id, 'metadata_sha256': digest}), encoding='utf-8')


def _completed_metadata(folder):
    required = ['pipeline.joblib', 'metrics.csv', 'predictions.parquet', 'importance.csv',
                'curves_roc.csv', 'curves_pr.csv', 'samples.parquet']
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    marker = json.loads((folder / 'COMPLETE.json').read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or not isinstance(marker, dict):
        raise ValueError('模型记录格式无效，请重新训练。')
    fields = ['run_id', 'created_at', 'best_model', 'selected_strategy', 'threshold', 'sample_summary',
              'available_from', 'horizon_days', 'splits', 'test_scores', 'validation_scores', 'confusion_matrix']
    if (any(key not in metadata for key in fields) or not all((folder / name).is_file() for name in required)
            or metadata.get('features') != FEATURES or metadata.get('feature_version') != FEATURE_VERSION
            or metadata.get('task') not in TASK_LABELS or metadata.get('best_model') not in MODEL_LABELS
            or metadata.get('selected_strategy') not in STRATEGY_LABELS
            or metadata.get('run_id') != folder.name or marker.get('run_id') != folder.name
            or marker.get('metadata_sha256') != hashlib.sha256((folder / 'metadata.json').read_bytes()).hexdigest()
            or not isinstance(metadata.get('created_at'), str)):
        raise ValueError('模型记录不完整或元数据已损坏，请重新训练。')
    return metadata


def saved_records(root, dataset_id):
    records = []
    for path in (Path(root) / 'artifacts' / 'prediction').glob('*/metadata.json'):
        try:
            metadata = _completed_metadata(path.parent)
            if metadata.get('dataset_id') == dataset_id:
                records.append({**metadata, 'artifact_dir': str(path.parent), 'run_id': path.parent.name})
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return sorted(records, key=lambda item: item['created_at'], reverse=True)


def load_experiment(folder, dataset_id):
    _completed_metadata(Path(folder))
    _, metadata = read_artifact(folder, expected_dataset_id=dataset_id)
    folder = Path(folder)
    result = {'metadata': metadata, 'metrics': pd.read_csv(folder / 'metrics.csv'),
              'predictions': pd.read_parquet(folder / 'predictions.parquet'),
              'importance': pd.read_csv(folder / 'importance.csv'),
              'curves': {'roc': pd.read_csv(folder / 'curves_roc.csv'), 'pr': pd.read_csv(folder / 'curves_pr.csv')},
              'sample_summary': pd.DataFrame(metadata.get('sample_summary', [])),
              'artifact_dir': str(folder), 'best_model': metadata['best_model'], 'threshold': metadata['threshold']}
    return result, {**metadata, 'artifact_dir': str(folder)}


def export_customer_predictions(selected, context, customer_search=''):
    description = {'任务': TASK_LABELS[context['task']], '训练编号': context['run_id'],
                   '数据版本': context['dataset_id'], '判定阈值': context['threshold'],
                   '未来观察天数': context['horizon_days'], '模型可用日期': context['available_from'],
                   '已校准': context['probabilities_calibrated'], '客户编号搜索': customer_search,
                   '导出客户数': len(selected), '说明': '预测分数未校准，不代表精确发生概率。'
                   '数据末尾之后的未来结果尚不可验证；未购买不代表永久流失。'}
    content = io.BytesIO()
    with zipfile.ZipFile(content, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('客户预测.csv', display_predictions(selected, context['task']).to_csv(index=False).encode('utf-8-sig'))
        archive.writestr('预测说明.json', json.dumps(description, ensure_ascii=False, indent=2, allow_nan=False))
    return content.getvalue()
