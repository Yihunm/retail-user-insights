"""本地数据质量工作台；当前只开放已验证的数据导入模块。"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from retail_insights.preparation import PROJECT_ROOT, load_dataset, prepare_file  # noqa: E402


st.set_page_config(page_title='Retail Insights · 数据质量', layout='wide')
root = Path(os.environ.get('RETAIL_DATA_ROOT', PROJECT_ROOT)).resolve()
st.sidebar.title('Retail Insights')
st.sidebar.caption('电商用户行为分析系统')
st.sidebar.markdown('**当前模块**\n\n01 · 数据导入与清洗')
st.sidebar.divider()
st.sidebar.caption('后续依次开发：交易分析、RFM、聚类、复购与流失预测。')
st.title('数据导入与质量检查')
st.caption('Online Retail · 历史交易数据 · 金额单位 GBP（英镑）')

with st.expander('导入数据', expanded=not (root / 'data' / 'active.json').exists()):
    mode = st.radio('数据来源', ['选择文件', '本机文件路径'], horizontal=True)
    with st.form('import_form'):
        if mode == '选择文件':
            upload = st.file_uploader('选择 Excel 或 CSV', type=['xlsx', 'csv'])
            local_path = ''
        else:
            upload = None
            local_path = st.text_input('本机文件路径', placeholder='输入完整的 .xlsx 或 .csv 文件路径')
        drop_duplicates = st.checkbox('移除八个原始字段完全相同的重复行', value=True)
        st.caption('完全相同的记录未必都是错误，重复处理策略会写入质量报告。原始文件会保留。')
        submitted = st.form_submit_button('导入并检查', type='primary')
    if submitted:
        try:
            if mode == '选择文件':
                if upload is None:
                    raise ValueError('请先选择文件。')
                content = upload.getvalue()
                if len(content) > 80 * 1024**2:
                    raise ValueError('文件超过 80 MB，请在本机使用命令行导入。')
                folder = root / '.tmp' / 'uploads'
                folder.mkdir(parents=True, exist_ok=True)
                source = folder / (hashlib.sha256(content).hexdigest() + Path(upload.name).suffix.lower())
                source.write_bytes(content)
            else:
                if not local_path.strip():
                    raise ValueError('请输入数据文件路径。')
                source = Path(local_path.strip().strip('"'))
            with st.spinner('正在读取、分类与核对。首次读取全量 Excel 可能需要几十秒。'):
                imported = prepare_file(source, root=root, drop_duplicates=drop_duplicates)
            st.success('已复用相同数据版本。' if imported['reused'] else '导入完成，原始数据已保留。')
        except Exception as error:
            st.error(f'导入失败：{error}')


@st.cache_data(show_spinner=False, max_entries=2)
def cached_dataset(workspace, pointer):
    return load_dataset(workspace, dataset_id=json.loads(pointer)['dataset_id'])


pointer_path = root / 'data' / 'active.json'
if not pointer_path.exists():
    st.info('尚未导入数据。请先选择 Online Retail.xlsx 或兼容的 CSV 文件。')
    st.stop()
try:
    dataset = cached_dataset(str(root), pointer_path.read_text(encoding='utf-8'))
except Exception as error:
    st.error(f'数据无法加载：{error}。请重新导入或检查质量报告。')
    st.stop()

report = dataset['report']
metadata = dataset['metadata']
st.caption(f"数据版本 {metadata['dataset_id'][:12]} · 来源 {metadata['source_name']}")
metrics = st.columns(4)
for box, title, key in zip(metrics, ['原始明细', '有效销售', '取消 / 退货候选', '隔离记录'],
                           ['raw_rows', 'sales_rows', 'return_rows', 'quarantine_rows']):
    box.metric(title, f'{report[key]:,}')

if report['partition_verified']:
    st.success('行数核对通过：'
               f"{report['raw_rows']:,} = 重复移除 {report['duplicate_rows']:,} + "
               f"销售 {report['sales_rows']:,} + 取消/退货候选 {report['return_rows']:,} + "
               f"隔离 {report['quarantine_rows']:,}")
else:
    st.error('行数核对未通过，请停止后续分析并检查清洗结果。')

quality_tab, data_tab, rules_tab = st.tabs(['质量报告', '明细预览', '处理口径'])
with quality_tab:
    left, right = st.columns([3, 2])
    labels = {
        'invalid_numeric': '数量或单价无效', 'invalid_date': '交易时间无效',
        'missing_invoice': '订单编号缺失', 'missing_stock_code': '商品编号缺失',
        'nonpositive_price': '单价小于等于零', 'zero_quantity': '商品数量为零',
        'contradictory_cancellation': '取消订单与正数量冲突', 'special_code': '特殊业务编码',
        'invalid_amount': '金额无效或超出范围',
    }
    with left:
        st.subheader('隔离原因')
        problems = pd.DataFrame([{'问题': labels.get(key, key), '行数': count}
                                 for key, count in report['issue_counts'].items()])
        st.dataframe(problems, hide_index=True, width='stretch')
        st.caption('以上为去重后的问题计数，同一行可能存在多个问题，不能直接相加。')
    with right:
        st.subheader('核对摘要')
        st.write(f"去重后缺少客户编号：{report['missing_customer_rows']:,} 行")
        st.write('有效匿名销售保留在整体统计中，后续不用于客户建模。')
        st.write(f"有效商品销售额：£{report['sales_amount_cents'] / 100:,.2f}")
        st.write(f"取消/退货候选金额：£{report['return_amount_cents'] / 100:,.2f}")
        st.caption('这些金额已排除隔离记录，不等同于原始文件全部交易的金额。')
        st.download_button('下载质量报告 JSON', json.dumps(report, ensure_ascii=False, indent=2),
                           file_name='quality.json', mime='application/json')
with data_tab:
    for key, title in [('sales', '有效销售'), ('returns', '取消 / 退货候选'), ('quarantine', '隔离记录')]:
        with st.expander(f'{title} · {len(dataset[key]):,} 行', expanded=key == 'sales'):
            st.dataframe(dataset[key].head(200), hide_index=True, width='stretch')
            st.caption('预览最多 200 行；完整数据已保存为 Parquet。')
            if key == 'quarantine':
                st.download_button('下载隔离记录 CSV', dataset[key].to_csv(index=False).encode('utf-8-sig'),
                                   file_name='quarantine.csv', mime='text/csv')
with rules_tab:
    st.markdown('''
- 有效销售：正数量、正单价、有效时间与编号，且不属于取消或特殊业务编码。
- 取消/退货候选：负数量、正单价且其他字段有效；不声称已匹配到原始订单。
- 隔离：字段异常、零/负价格、取消订单正数量，或特殊业务编码。
- 客户编号缺失不会删除有效销售；商品名称与国家缺失显示“未知”。
- 金额按逐行四舍五入到整数分计算，先汇总整数分，再转换为英镑。
- 系统没有浏览、收藏或加购日志，不生成这些字段。
''')
    st.write('特殊业务编码（本版固定配置）：', ', '.join(report['special_codes']))
    st.caption('这些记录独立保留在隔离表中，可核查后再调整规则。')
