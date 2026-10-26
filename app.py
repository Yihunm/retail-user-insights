"""电商用户分析本地演示入口。"""
import os
import sys
from pathlib import Path
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from retail_insights.preparation import PROJECT_ROOT  # noqa: E402
from retail_insights.ui_data import render as render_data  # noqa: E402
from retail_insights.ui_sales import render as render_sales  # noqa: E402
from retail_insights.ui_rfm import render as render_rfm  # noqa: E402

st.set_page_config(page_title='Retail Insights · 用户分析', layout='wide')
root = Path(os.environ.get('RETAIL_DATA_ROOT', PROJECT_ROOT)).resolve()
st.sidebar.title('Retail Insights')
st.sidebar.caption('电商用户行为分析系统')
pages = {'数据导入与清洗': render_data, '交易统计与可视化': render_sales, 'RFM用户价值分析': render_rfm}
selected = st.sidebar.radio('功能模块', list(pages), key='module')
st.sidebar.divider()
st.sidebar.caption('历史交易数据 · GBP · 本地演示')
pages[selected](root)
