import io
import json
import zipfile

import joblib


def test_cluster_artifacts_and_export_keep_group_labels_and_model_schema(tmp_path):
    from retail_insights.data import clean_transactions, read_transactions
    from retail_insights.rfm import build_rfm
    from retail_insights.clustering import cluster_customers, FEATURES
    from retail_insights.clustering_presentation import persist_clustering, export_clustering
    from test_clustering_page import cluster_source

    rfm = build_rfm(clean_transactions(read_transactions(cluster_source(tmp_path)))['sales'])
    result = cluster_customers(rfm, k=3)
    context = {'dataset_id': 'dataset-a', 'reference_date': rfm.attrs['reference_date'], 'window_days': None,
               'requested_k': 3}
    saved = persist_clustering(result, context, root=tmp_path)
    model = joblib.load(saved['artifact_dir'] + '/model.joblib')
    assert (model.predict(result['customers'][FEATURES]) == result['customers'].Cluster).all()
    content = export_clustering(result, saved, groups=[0], customer_search='')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        import pandas as pd
        members = pd.read_csv(io.BytesIO(archive.read('聚类客户明细.csv')))
        assert members['聚类群体'].unique().tolist() == ['群体1']
        assert len(members) == result['customers'].Cluster.eq(0).sum()
        metadata = json.loads(archive.read('聚类与筛选说明.json'))
        assert metadata['模型客户数'] == 12
        assert metadata['导出客户数'] == len(members)
        assert metadata['模型特征'] == FEATURES
        assert 'PCA仅用于展示' in metadata['说明']
