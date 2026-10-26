import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from retail_insights.clustering import FEATURES, cluster_customers
from retail_insights.clustering import _stratified_indices


def cohort(n=18):
    groups = np.arange(n) // 6
    values = np.arange(n) % 6
    return pd.DataFrame({
        'CustomerID': [f'{i:04}' for i in range(n)], 'Recency': 1 + groups*30 + values,
        'Frequency': 1 + groups*5 + values, 'Monetary': 10 + groups*100 + values,
        'Quantity': 2 + groups*20 + values/2, 'AverageOrderValue': 5 + groups*10 + values,
        'MonetaryCents': [int(value) for value in (10 + groups*100 + values)*100],
        'Segment': ['一般客户']*n,
    })


def test_reproducibility_permutation_nonmutation_and_pipeline_roundtrip():
    frame = cohort()
    before = frame.copy(deep=True)
    result = cluster_customers(frame, k=3)
    again = cluster_customers(frame.sample(frac=1, random_state=11), k=3)
    pd.testing.assert_frame_equal(result['customers'], again['customers'])
    pd.testing.assert_frame_equal(frame, before)
    assert result['customers'].CustomerID.tolist() == sorted(frame.CustomerID)
    assert result['customers'].Segment.tolist() == frame.Segment.tolist()
    assert np.isfinite(result['customers'][['PC1', 'PC2']]).all().all()
    assert np.array_equal(result['model'].predict(result['customers'][FEATURES]), result['customers'].Cluster)
    stream = io.BytesIO()
    joblib.dump(result['model'], stream)
    stream.seek(0)
    assert np.array_equal(joblib.load(stream).predict(frame[FEATURES]), result['customers'].Cluster)
    json.dumps(result['metadata'], allow_nan=False)
    assert result['metadata']['n_init'] == 20
    assert result['metadata']['selected_k'] == 3
    assert result['metadata']['cohort_customers'] == len(frame)
    assert len(result['metadata']['pca_explained_variance_ratio']) == 2


def test_candidates_and_auto_select_best_silhouette():
    result = cluster_customers(cohort(), k=None)
    scores = result['scores']
    assert scores.columns.tolist() == ['k', 'Inertia', 'Silhouette', 'EvaluationCustomers', 'Sampled', 'ActualClusters']
    assert scores.k.tolist() == list(range(2, 9))
    best = scores.sort_values(['Silhouette', 'k'], ascending=[False, True]).iloc[0]['k']
    assert result['metadata']['selected_k'] == best
    assert result['metadata']['recommended_k'] == best
    assert scores.EvaluationCustomers.tolist() == [18]*7
    assert not scores.Sampled.any()
    assert scores.ActualClusters.tolist() == scores.k.tolist()


@pytest.mark.parametrize('n', [0, 1, 2])
def test_tiny_input_is_rejected(n):
    with pytest.raises(ValueError, match='至少|样本|客户'):
        cluster_customers(cohort(n), k=2)


@pytest.mark.parametrize('k', [True, False, 1, 1.5, '3', 18, 19])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match='k|K'):
        cluster_customers(cohort(), k=k)


@pytest.mark.parametrize('value', [-1, np.nan, np.inf, 'bad', True])
def test_invalid_feature_values_rejected(value):
    frame = cohort()
    frame['Recency'] = frame.Recency.astype(object)
    frame.loc[0, 'Recency'] = value
    with pytest.raises(ValueError, match='Recency|特征'):
        cluster_customers(frame)


def test_missing_features_duplicate_or_missing_customer_ids():
    with pytest.raises(ValueError, match='Monetary'):
        cluster_customers(cohort().drop(columns='Monetary'))
    for bad in [None, '', '   ', '0001']:
        frame = cohort()
        frame.loc[0, 'CustomerID'] = bad
        with pytest.raises(ValueError, match='CustomerID|客户'):
            cluster_customers(frame)


def test_constant_and_transformed_identical_vectors_are_rejected():
    constant = cohort()
    constant[FEATURES] = 1
    with pytest.raises(ValueError, match='差异|方差|variance'):
        cluster_customers(constant)
    frame = cohort(6)
    frame[FEATURES] = [[1, 2, 3, 4, 5]]*3 + [[2, 3, 4, 5, 6]]*3
    with pytest.raises(ValueError, match='不同|唯一|distinct|k'):
        cluster_customers(frame, k=3)
    result = cluster_customers(frame, k=2)
    assert result['scores'].k.tolist() == [2]


def test_one_effective_feature_has_zero_second_pca_coordinate():
    frame = cohort(6)
    frame[FEATURES] = 1
    frame['Recency'] = [1, 2, 3, 10, 11, 12]
    result = cluster_customers(frame, k=2)
    assert result['metadata']['effective_features'] == ['Recency']
    assert result['metadata']['pca_dimensions'] == 1
    assert result['customers'].PC2.tolist() == [0.]*6
    assert result['metadata']['pca_explained_variance_ratio'] == [1., 0.]


def test_stratified_sampling_includes_rare_cluster_and_respects_size():
    frame = cohort(30)
    frame[FEATURES] = 1
    frame['Recency'] = [1]*14 + [10]*15 + [100000]
    result = cluster_customers(frame, k=3, silhouette_sample_size=10)
    scores = result['scores']
    assert scores.k.tolist() == [2, 3]
    assert scores.EvaluationCustomers.tolist() == [10, 10]
    assert scores.Sampled.tolist() == [True, True]
    assert scores.Silhouette.notna().all()
    assert result['customers'].Cluster.nunique() == 3
    again = cluster_customers(frame, k=3, silhouette_sample_size=10)
    pd.testing.assert_frame_equal(scores, again['scores'])


def test_sample_size_must_allow_each_candidate_cluster_and_one_extra_customer():
    with pytest.raises(ValueError, match='sample|抽样|采样'):
        cluster_customers(cohort(), k=3, silhouette_sample_size=4)


def test_profile_exact_cents_counts_means_and_fractional_quantities():
    frame = cohort()
    frame['MonetaryCents'] = pd.Series([9_000_000_000_000_000_000+i for i in range(len(frame))], dtype=object)
    frame['Quantity'] = pd.Series([6_000_000_000_000_000_000]*17 + [.5], dtype=object)
    profile = cluster_customers(frame, k=3)['profile']
    assert profile.columns.tolist() == ['Cluster', 'Count', 'CustomerShare', 'MonetaryCents', 'Monetary',
                                       'MeanRecency', 'MeanFrequency', 'MeanMonetary', 'MedianMonetary',
                                       'MeanQuantity', 'MeanAverageOrderValue']
    assert sum(profile.Count) == len(frame)
    assert sum(profile.MonetaryCents) == sum(frame.MonetaryCents)
    assert all(isinstance(value, int) for value in profile.MonetaryCents)
    assert profile.CustomerShare.sum() == pytest.approx(1)
    assert profile.MeanQuantity.ge(0).all()


def test_three_samples_have_only_two_candidate_clusters():
    result = cluster_customers(cohort(3), k=2)
    assert result['scores'].k.tolist() == [2]


def test_sampling_always_preserves_singleton_cluster():
    labels = np.array([0]*90 + [1]*9 + [2])
    indices = _stratified_indices(labels, 12, 42)
    assert len(indices) == len(set(indices)) == 12
    assert 99 in indices
    assert set(labels[indices]) == {0, 1, 2}
    assert np.array_equal(indices, _stratified_indices(labels, 12, 42))


def test_default_large_cohort_samples_at_most_two_thousand():
    frame = cohort(2100)
    frame[FEATURES] = 1
    frame['Recency'] = [1]*1000 + [10]*1099 + [100000]
    result = cluster_customers(frame, k=3)
    assert result['scores'].EvaluationCustomers.tolist() == [2000, 2000]
    assert result['scores'].Sampled.tolist() == [True, True]


def test_undefined_silhouette_returns_none_and_auto_explains_failure():
    result = cluster_customers(cohort(3), k=2, silhouette_sample_size=2)
    assert result['scores'].Silhouette.tolist() == [None]
    assert result['metadata']['recommended_k'] is None
    with pytest.raises(ValueError, match='轮廓系数'):
        cluster_customers(cohort(3), k=None, silhouette_sample_size=2)


def test_auto_breaks_silhouette_ties_with_lower_k(monkeypatch):
    monkeypatch.setattr('retail_insights.clustering.silhouette_score', lambda *_: .25)
    assert cluster_customers(cohort(), k=None)['metadata']['selected_k'] == 2
