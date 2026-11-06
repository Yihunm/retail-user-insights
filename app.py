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
from retail_insights.ui_clustering import render as render_clustering  # noqa: E402
from retail_insights.ui_prediction import render as render_prediction  # noqa: E402
from retail_insights.ui_home import render as render_home  # noqa: E402
from retail_insights.ui_shared import shell  # noqa: E402

st.set_page_config(page_title='Retail Insights · 客户经营', layout='wide')
root = Path(os.environ.get('RETAIL_DATA_ROOT', PROJECT_ROOT)).resolve()
pages = {'工作台': render_home, '经营分析': render_sales, '客户价值': render_rfm,
         '客户分群': render_clustering, '复购与流失': render_prediction, '数据管理': render_data,
         '模型实验': lambda root: render_prediction(root, mode='experiments')}
selected = shell(root)
pages[selected](root)
