import json
from pathlib import Path

from test_clustering_page import cluster_source


def test_saved_report_restores_identifiers_dates_and_groups_and_skips_broken_newer_run(tmp_path):
    from retail_insights.preparation import prepare_file, load_dataset
    from retail_insights.rfm import build_rfm
    from retail_insights.clustering import cluster_customers
    from retail_insights.clustering_presentation import persist_clustering, latest_clustering

    metadata = prepare_file(cluster_source(tmp_path), root=tmp_path)
    cohort = build_rfm(load_dataset(tmp_path)['sales'])
    result = cluster_customers(cohort, k=3)
    saved = persist_clustering(result, {'dataset_id': metadata['dataset_id'],
        'reference_date': cohort.attrs['reference_date'], 'window_days': None, 'requested_k': 3}, tmp_path)
    broken = tmp_path / 'artifacts' / 'clustering' / 'broken'
    broken.mkdir()
    (broken / 'metadata.json').write_text(json.dumps({**saved, 'trained_at': '2099-01-01T00:00:00Z'}))
    loaded, context = latest_clustering(tmp_path, metadata['dataset_id'])
    assert context['run_id'] == saved['run_id']
    assert loaded['customers'].CustomerID.tolist() == result['customers'].CustomerID.tolist()
    assert loaded['customers'].LastPurchase.tolist() == result['customers'].LastPurchase.tolist()
    assert loaded['customers'].Cluster.tolist() == result['customers'].Cluster.tolist()
    assert latest_clustering(tmp_path, 'other-data') is None
    Path(saved['artifact_dir'], 'customers.csv').write_text('damaged')
    assert latest_clustering(tmp_path, metadata['dataset_id']) is None
