"""聚类结果的中文展示、可信本地模型保存与筛选导出。"""
import io
import json
import sqlite3
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib

from .rfm_presentation import display_rfm

PROFILE_COLUMNS = {'Cluster': '聚类群体', 'Count': '客户数', 'CustomerShare': '客户占比',
                   'Monetary': '消费总额（英镑）', 'MeanRecency': '平均最近购买间隔（天）',
                   'MeanFrequency': '平均购买订单数', 'MeanMonetary': '人均消费金额（英镑）',
                   'MedianMonetary': '消费金额中位数（英镑）', 'MeanQuantity': '平均购买数量',
                   'MeanAverageOrderValue': '平均客单金额（英镑）'}
SCORE_COLUMNS = {'k': '聚类数量', 'Inertia': '簇内平方和', 'Silhouette': '轮廓系数',
                 'EvaluationCustomers': '评价客户数', 'Sampled': '是否抽样', 'ActualClusters': '实际群体数'}


def group_name(value):
    return f'群体{int(value) + 1}'


def display_members(frame, exact_cents=False):
    shown = display_rfm(frame, include_cents=exact_cents).rename(columns={'客户类型': 'RFM规则类型'})
    shown['聚类群体'] = frame.Cluster.map(group_name)
    shown['主成分1'] = frame.PC1
    shown['主成分2'] = frame.PC2
    return shown


def display_profile(frame):
    shown = frame[list(PROFILE_COLUMNS)].rename(columns=PROFILE_COLUMNS).copy()
    shown['聚类群体'] = frame.Cluster.map(group_name)
    return shown


def display_scores(frame):
    shown = frame.rename(columns=SCORE_COLUMNS).copy()
    shown['是否抽样'] = frame.Sampled.map({True: '是', False: '否'})
    return shown


def select_members(result, groups=None, customer_search=''):
    members = result['customers']
    selected = members if groups is None else members[members.Cluster.isin(groups)]
    query = customer_search.strip()
    if query:
        selected = selected[selected.CustomerID.str.contains(query, regex=False, case=False)]
    return selected.copy()


def persist_clustering(result, context, root):
    run_id = uuid4().hex
    folder = Path(root).resolve() / 'artifacts' / 'clustering' / run_id
    metadata = {**result['metadata'], **context, 'run_id': run_id,
                'trained_at': datetime.now(timezone.utc).isoformat()}
    encoded = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    folder.mkdir(parents=True)
    joblib.dump(result['model'], folder / 'model.joblib')
    for name in ['customers', 'profile', 'scores']:
        result[name].to_csv(folder / f'{name}.csv', index=False, encoding='utf-8-sig')
    (folder / 'metadata.json').write_text(encoded, encoding='utf-8')
    data_dir = Path(root).resolve() / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(data_dir / 'catalog.sqlite3')) as db:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS cluster_runs '
                       '(id TEXT PRIMARY KEY, dataset_id TEXT, created_at TEXT, metadata TEXT)')
            db.execute('INSERT INTO cluster_runs VALUES (?,?,?,?)',
                       (run_id, metadata['dataset_id'], metadata['trained_at'], encoded))
    return {**metadata, 'artifact_dir': str(folder)}


def export_clustering(result, context, groups=None, customer_search=''):
    members = select_members(result, groups, customer_search)
    explanation = {
        '训练编号': context['run_id'], '数据版本': context['dataset_id'],
        '分析参考日': context['reference_date'], '观察窗口天数': context['window_days'],
        '模型客户数': len(result['customers']), '导出客户数': len(members),
        '模型特征': result['metadata']['features'], '有效特征': result['metadata']['effective_features'],
        '聚类数量': result['metadata']['selected_k'], '推荐聚类数量': result['metadata']['recommended_k'],
        '随机种子': result['metadata']['random_state'], '初始化次数': result['metadata']['n_init'],
        '评价抽样上限': result['metadata']['silhouette_sample_size'],
        '降维解释方差比例': result['metadata']['pca_explained_variance_ratio'],
        '客户群体筛选': None if groups is None else [group_name(value) for value in groups],
        '客户编号搜索': customer_search.strip(),
        '说明': 'PCA仅用于展示，分群使用全部有效变换特征。群体编号不代表价值排名；'
                '画像与聚类数量评价基于模型全体客户，筛选只影响导出客户明细。',
    }
    content = io.BytesIO()
    with zipfile.ZipFile(content, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, frame in [('聚类客户明细.csv', display_members(members, exact_cents=True)),
                            ('模型总体群体画像.csv', display_profile(result['profile'])),
                            ('聚类数量评价.csv', display_scores(result['scores']))]:
            archive.writestr(name, frame.to_csv(index=False).encode('utf-8-sig'))
        archive.writestr('聚类与筛选说明.json', json.dumps(explanation, ensure_ascii=False, indent=2,
                                                        allow_nan=False))
    return content.getvalue()
