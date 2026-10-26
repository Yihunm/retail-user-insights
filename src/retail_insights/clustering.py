"""全量 RFM 特征上的确定性 KMeans；PCA 仅用于展示。"""

import warnings
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = ['Recency', 'Frequency', 'Monetary', 'Quantity', 'AverageOrderValue']


def _validate(rfm, k, random_state, sample_size):
    missing = [column for column in ['CustomerID', 'MonetaryCents', *FEATURES] if column not in rfm]
    if missing:
        raise ValueError('聚类数据缺少列：' + ', '.join(missing))
    if len(rfm) < 3:
        raise ValueError('聚类至少需要 3 个客户样本')
    ids = rfm.CustomerID
    if not ids.map(lambda value: isinstance(value, str) and bool(value.strip())).all() or ids.duplicated().any():
        raise ValueError('CustomerID 必须是唯一、非空的客户字符串')
    for column in FEATURES:
        if not rfm[column].map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f'特征 {column} 必须是非负有限数值')
        try:
            values = rfm[column].to_numpy(dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'特征 {column} 必须是非负有限数值') from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'特征 {column} 必须是非负有限数值')
    if not rfm.MonetaryCents.map(lambda value: isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0).all():
        raise ValueError('MonetaryCents 必须是非负整数分')
    if k is not None and (isinstance(k, bool) or not isinstance(k, Integral) or not 2 <= k < len(rfm)):
        raise ValueError('k 必须是满足 2≤k<客户数 的整数，或 None 自动选择')
    if isinstance(random_state, bool) or not isinstance(random_state, Integral) or not 0 <= random_state <= 2**32-1:
        raise ValueError('random_state 必须是 0 到 2**32-1 的整数')
    if isinstance(sample_size, bool) or not isinstance(sample_size, Integral) or sample_size < 1:
        raise ValueError('silhouette_sample_size 抽样数量必须是正整数')


def _stratified_indices(labels, sample_size, random_state):
    """每簇先选一人，再从剩余客户中无放回抽取，确保稀有簇存在。"""
    if len(labels) <= sample_size:
        return np.arange(len(labels))
    generator = np.random.default_rng(random_state)
    mandatory = [int(generator.choice(np.flatnonzero(labels == label))) for label in np.unique(labels)]
    remaining = np.setdiff1d(np.arange(len(labels)), mandatory)
    extra = generator.choice(remaining, size=sample_size-len(mandatory), replace=False)
    return np.sort(np.concatenate([mandatory, extra]).astype(int))


def _python_sum(values):
    return sum(value.item() if isinstance(value, np.generic) else value for value in values)


def _profile(customers):
    records = []
    for cluster, rows in customers.groupby('Cluster', sort=True):
        count = len(rows)
        cents = sum(int(value) for value in rows.MonetaryCents)
        records.append({
            'Cluster': int(cluster), 'Count': count, 'CustomerShare': count / len(customers),
            'MonetaryCents': cents, 'Monetary': cents / 100,
            'MeanRecency': _python_sum(rows.Recency) / count,
            'MeanFrequency': _python_sum(rows.Frequency) / count,
            'MeanMonetary': _python_sum(rows.Monetary) / count,
            'MedianMonetary': float(np.median(rows.Monetary.to_numpy(dtype=float))),
            'MeanQuantity': _python_sum(rows.Quantity) / count,
            'MeanAverageOrderValue': _python_sum(rows.AverageOrderValue) / count,
        })
    profile = pd.DataFrame(records)
    profile['MonetaryCents'] = pd.Series([record['MonetaryCents'] for record in records], dtype=object)
    return profile


def cluster_customers(rfm, k=4, random_state=42, silhouette_sample_size=2000):
    """在 log1p→VarianceThreshold→StandardScaler 的完整特征空间拟合 KMeans。

    计算所有候选 k 的轮廓系数；抽样时至少包含每个簇一人。当抽样人数等于
    簇数时轮廓系数无定义，该行返回 None。自动选择忽略无效行，相同时选较小 k。
    profile 金额总计使用精确整数分；数值 Cluster 不表示客户价值等级。
    """
    _validate(rfm, k, random_state, silhouette_sample_size)
    random_state, silhouette_sample_size = int(random_state), int(silhouette_sample_size)
    customers = rfm.sort_values('CustomerID', kind='stable').reset_index(drop=True).copy(deep=True)
    with threadpool_limits(limits=1):
        preprocessing = Pipeline([
            ('log1p', FunctionTransformer(np.log1p, validate=True, feature_names_out='one-to-one')),
            ('variance', VarianceThreshold()), ('scale', StandardScaler()),
        ])
        try:
            transformed = preprocessing.fit_transform(customers[FEATURES])
        except ValueError as exc:
            raise ValueError('客户特征没有可用于聚类的方差或差异') from exc
        distinct = len(np.unique(transformed, axis=0))
        maximum = min(8, len(customers)-1, distinct)
        if maximum < 2:
            raise ValueError('客户变换后的特征没有足够不同向量或方差')
        if k is not None and k > distinct:
            raise ValueError('k 不能超过变换后不同特征向量的数量')
        if len(customers) > silhouette_sample_size and silhouette_sample_size < maximum:
            raise ValueError('轮廓系数抽样 silhouette_sample_size 必须至少等于最大候选 k')
        requested = int(k) if k is not None else None
        candidates = list(range(2, maximum+1))
        # A user may explicitly request k>8; evaluate it alongside the standard search.
        fitting = sorted(set(candidates + ([requested] if requested is not None else [])))
        models, records = {}, []
        for candidate in fitting:
            model = KMeans(n_clusters=candidate, random_state=random_state, n_init=20, algorithm='lloyd')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                labels = model.fit_predict(transformed)
            models[candidate] = model
            actual = len(np.unique(labels))
            if len(customers) > silhouette_sample_size and silhouette_sample_size < actual:
                raise ValueError('轮廓系数抽样数量必须至少覆盖每个簇')
            indices = _stratified_indices(labels, silhouette_sample_size, random_state)
            value = None
            if actual == candidate and 1 < len(np.unique(labels[indices])) < len(indices):
                value = float(silhouette_score(transformed[indices], labels[indices]))
            if candidate in candidates:
                records.append({'k': candidate, 'Inertia': float(model.inertia_), 'Silhouette': value,
                                'EvaluationCustomers': len(indices), 'Sampled': len(indices) < len(customers),
                                'ActualClusters': actual})
        valid = [row for row in records if row['Silhouette'] is not None]
        recommended = min(valid, key=lambda row: (-row['Silhouette'], row['k']))['k'] if valid else None
        selected = requested if requested is not None else recommended
        if selected is None:
            raise ValueError('没有可比较的轮廓系数；请增加抽样人数或明确选择 k')
        selected_model = models[selected]
        if len(np.unique(selected_model.labels_)) != selected:
            raise ValueError('所选 k 未形成要求数量的实际簇，请减少 k')
        customers['Cluster'] = selected_model.labels_.astype(int)
        dimensions = min(2, transformed.shape[1])
        pca = PCA(n_components=dimensions, svd_solver='full')
        coordinates = pca.fit_transform(transformed)
        customers['PC1'] = coordinates[:, 0]
        customers['PC2'] = coordinates[:, 1] if dimensions == 2 else 0.
        ratios = pca.explained_variance_ratio_.tolist() + ([0.] if dimensions == 1 else [])
        pipeline = Pipeline([*preprocessing.steps, ('kmeans', selected_model)])
    scores = pd.DataFrame(records)
    scores['Silhouette'] = pd.Series([row['Silhouette'] for row in records], dtype=object)
    metadata = {
        'features': FEATURES.copy(),
        'effective_features': [name for name, keep in zip(FEATURES, preprocessing.named_steps['variance'].get_support()) if keep],
        'random_state': random_state, 'n_init': 20, 'selected_k': selected, 'recommended_k': recommended,
        'silhouette_sample_size': silhouette_sample_size, 'cohort_customers': len(customers),
        'pca_explained_variance_ratio': ratios, 'pca_dimensions': dimensions,
        'preprocessing': 'log1p → VarianceThreshold → StandardScaler → KMeans (Lloyd, n_init=20)；PCA 仅用于展示',
    }
    return {'customers': customers, 'profile': _profile(customers), 'scores': scores,
            'model': pipeline, 'metadata': metadata}
