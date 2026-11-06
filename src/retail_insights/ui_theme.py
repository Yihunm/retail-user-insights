"""全站视觉语言：靛蓝导航、轻量卡片与业务语义配色。"""
from urllib.parse import quote

PRIMARY = '#5865D9'
INK = '#222C46'
PALETTE = [PRIMARY, '#229D92', '#D39A4F', '#8E79C5', '#5E93C5', '#C87689', '#6D9D76', '#9A8C78']
SEGMENT_COLORS = {'高价值客户': PRIMARY, '流失风险客户': '#D39A4F', '沉睡客户': '#A0AEC0',
                  '潜力客户': '#229D92', '一般客户': '#6D96C4'}
ICONS = {
    'overview': '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>'
                '<rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    'trend': '<path d="M4 4v16h16M7 14l4-4 4 2 5-7"/>',
    'value': '<path d="m12 3 3 6 6 1-4.5 4.5 1 6.5-5.5-3-5.5 3 1-6.5L3 10l6-1Z"/>',
    'groups': '<circle cx="9" cy="8" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3M16 5a3 3 0 0 1 0 6M17 15a5 5 0 0 1 4 5"/>',
    'risk': '<path d="M3 12h4l3-7 4 14 3-7h4"/>',
    'data': '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 4 16 4 16 0V5M4 12v7c0 4 16 4 16 0v-7"/>',
    'experiment': '<path d="M9 3h6M10 3v6l-6 10a1.4 1.4 0 0 0 1 2h14a1.4 1.4 0 0 0 1-2L14 9V3M7 15h10"/>',
    'money': '<rect x="3" y="5" width="18" height="14" rx="3"/><circle cx="12" cy="12" r="3"/><path d="M6 12h.01M18 12h.01"/>',
    'orders': '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 2v4M15 2v4M9 11h6M9 15h4"/>',
    'repeat': '<path d="M20 7h-5l2-3M4 17h5l-2 3M20 7a8 8 0 0 0-14-2M4 17a8 8 0 0 0 14 2M20 7v5M4 17v-5"/>',
}


def icon(name, color='currentColor'):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{ICONS[name]}</svg>')


def styles():
    navigation = []
    for index, name in enumerate(['overview', 'trend', 'value', 'groups', 'risk', 'data', 'experiment'], 1):
        url = quote(icon(name, '#A9B4CF'), safe='')
        navigation.append(f'[data-testid="stSidebar"] [role="radiogroup"]>div:nth-child({index}) '
                          f'label::before {{background-image:url("data:image/svg+xml,{url}");}}')
    metrics = []
    for index, name in enumerate(['money', 'orders', 'groups', 'repeat'], 1):
        url = quote(icon(name, PRIMARY if index % 2 else '#229D92'), safe='')
        metrics.append(f'.st-key-home_kpis [data-testid="stColumn"]:nth-child({index}) '
                       f'[data-testid="stMetric"]::after {{background-image:url("data:image/svg+xml,{url}");}}')
    return '<style>' + CSS + ''.join(navigation + metrics) + '</style>'


CSS = '''
:root {--ri-primary:#5865D9;--ri-ink:#222C46;--ri-muted:#78849C;--ri-border:#E9ECF3;}
.stApp {background:#F6F7FB;color:var(--ri-ink);}
.stApp, .stMarkdown {font-family:"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;}
.stMainBlockContainer {max-width:1440px;padding:3.8rem 3.2rem 3rem;}
[data-testid="stHeader"] {background:rgba(246,247,251,.94);}
[data-testid="stAppDeployButton"] {display:none;}
h1,h2,h3 {font-family:"Microsoft YaHei","PingFang SC",sans-serif;letter-spacing:-.035em;}
h1 {font-size:1.85rem!important;font-weight:700!important;line-height:1.4!important;margin-bottom:0!important;padding:.3rem 0 .2rem!important;}
h2 {font-size:1.3rem!important;} h3 {font-size:1rem!important;font-weight:650!important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:#758199;font-size:.78rem;line-height:1.75;}
.st-key-page_heading {gap:.4rem;}
.header-meta {display:flex;justify-content:space-between;align-items:center;gap:1rem;margin:0 0 .15rem;}
.page-section {color:#8691A6;font-size:.7rem;letter-spacing:.08em;display:flex;align-items:center;gap:8px;}
.page-section::before {content:"";height:5px;width:5px;border-radius:50%;background:#7783E6;}
.page-badge {color:#658275;background:#EBF3EF;border:1px solid #DEEAE2;font-size:.68rem;
  border-radius:20px;padding:4px 10px;white-space:nowrap;}
[data-testid="stSidebar"] {background:#19223B;border-right:0;}
[data-testid="stSidebarContent"] {padding-top:.4rem;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding:0 0 1.5rem;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {color:#A9B4CF;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#9AA8C5;}
[data-testid="stSidebar"] button {color:#BFC9DF;}
[data-testid="stSidebar"] hr {border-color:rgba(177,192,226,.12);margin:1.2rem 0;}
.brand {display:flex;align-items:center;gap:12px;margin:.25rem 0 2rem;}
.brand-mark {width:36px;height:40px;display:flex;align-items:center;justify-content:center;
  background:#5865D9;border-radius:11px;color:white;box-shadow:0 4px 16px #10182B;}
.brand-name {font-size:1.05rem;font-weight:700;letter-spacing:-.03em;color:#F8FAFF;}
.brand-sub {font-size:.65rem;letter-spacing:.05em;color:#8F9BB7;margin-top:3px;}
.nav-section {color:#74829F;font-size:.65rem;letter-spacing:.12em;margin:0 0 .6rem .7rem;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:5px;width:100%;}
[data-testid="stSidebar"] [role="radiogroup"]>div {width:100%;}
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
  width:100%;box-sizing:border-box;display:flex;align-items:center;gap:12px;padding:12px 13px;
  border-radius:9px;background:transparent;transition:background .16s ease;}
[data-testid="stSidebar"] [data-testid="stRadioOption"]>div>div:first-child {display:none;}
[data-testid="stSidebar"] [data-testid="stRadioOption"]::before {
  content:"";width:19px;height:19px;flex:0 0 19px;background-size:contain;background-repeat:no-repeat;}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {font-size:.84rem;color:#ABB6D0;}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {background:#26314D;}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {background:#5865D9;box-shadow:0 4px 14px #121B31;}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p {color:#FFF;font-weight:600;}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked)::before {filter:brightness(0) invert(1);}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:focus-visible) {outline:2px solid #BAC4FF;outline-offset:3px;}
[data-testid="stSidebar"] [role="radiogroup"]>div:nth-child(6) {border-top:1px solid #303B54;padding-top:13px;margin-top:15px;}
[data-testid="stSidebar"] [data-testid="stExpander"] {background:transparent;border-color:#35405A;color:#B6C1D8;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {color:#A9B4CF;font-size:.72rem;}
.source-status {display:flex;align-items:center;gap:8px;font-size:.76rem;color:#D2DCF0;margin:.4rem 0;}
.source-status::before {content:"";width:6px;height:6px;border-radius:50%;background:#56C6AB;box-shadow:0 0 0 4px #233E43;}
.sidebar-hint {font-size:.67rem;color:#74829F;line-height:1.8;margin-top:1.1rem;}
[data-testid="stMetric"] {position:relative;background:#FFF;border:1px solid var(--ri-border);
  border-radius:14px;padding:1.15rem 1.25rem;box-shadow:0 3px 12px rgba(32,45,78,.025);}
[data-testid="stMetricLabel"] p {font-size:.76rem;color:#7B859C;}
[data-testid="stMetricValue"] {font-family:"Segoe UI",sans-serif;font-variant-numeric:tabular-nums;
  font-size:1.85rem;font-weight:650;letter-spacing:-.04em;color:#24304D;line-height:1.6;}
.st-key-home_kpis [data-testid="stMetric"] {padding-top:1.2rem;min-height:118px;}
.st-key-home_kpis [data-testid="stMetric"]::after {content:"";position:absolute;top:18px;right:18px;
  width:32px;height:32px;background-size:20px;background-position:center;background-repeat:no-repeat;
  background-color:#F1F3FB;border-radius:9px;opacity:.9;}
[data-testid="stForm"], [data-testid="stExpander"] {background:#FFF;border:1px solid var(--ri-border);border-radius:12px;}
[data-testid="stExpander"] summary {padding:.7rem .9rem;font-size:.82rem;}
.stButton>button,.stDownloadButton>button {border-radius:8px;font-weight:550;min-height:2.5rem;
  font-size:.8rem;border-color:#E2E6F0;transition:box-shadow .16s ease,background .16s ease;}
.stButton>button:hover,.stDownloadButton>button:hover {border-color:#A5AEEA;box-shadow:0 3px 10px #5865D913;}
button[kind="primary"] {box-shadow:0 3px 8px #5865D91F;}
button:focus-visible {outline:3px solid #ADB7FF!important;outline-offset:2px;}
[data-testid="stTabs"] [role="tablist"] {gap:1.6rem;border-bottom:1px solid #E6EAF3;}
[data-testid="stTabs"] [role="tab"] {font-size:.82rem;padding-bottom:.8rem;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {color:#5865D9;font-weight:650;}
[data-testid="stDataFrame"] {border-radius:10px;overflow:hidden;}
[data-testid="stAlert"] {border-radius:10px;font-size:.82rem;}
.panel-heading {display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:.7rem;}
.panel-title {font-size:.94rem;font-weight:650;color:#2A3653;}
.panel-subtitle {font-size:.68rem;color:#929CB0;}
.section-heading {display:flex;align-items:center;gap:10px;margin:.6rem 0 .1rem;font-size:1rem;font-weight:650;}
.section-heading::before {content:"";width:4px;height:16px;border-radius:3px;background:#5865D9;}
.section-description {font-size:.74rem;color:#8B95AA;margin:.3rem 0 1rem;}
.st-key-home_value_card,.st-key-home_risk_card,.st-key-home_group_card,
.st-key-home_trend_card,.st-key-home_distribution_card {
  background:#FFF;border:1px solid #E9ECF3!important;border-radius:15px!important;
  padding:1.25rem!important;box-shadow:0 3px 14px rgba(32,45,78,.025);}
.st-key-home_value_card,.st-key-home_risk_card,.st-key-home_group_card {height:100%;}
.action-top {display:flex;align-items:center;gap:10px;margin-bottom:6px;}
.action-icon {width:38px;height:38px;display:flex;align-items:center;justify-content:center;border-radius:11px;}
.action-icon.value {color:#6570D9;background:#EFF0FC;}
.action-icon.risk {color:#C18A40;background:#FAF3E7;}
.action-icon.groups {color:#27978C;background:#EAF6F2;}
.action-title {font-weight:650;color:#2C3751;font-size:.91rem;}
.action-number {font-family:"Segoe UI",sans-serif;font-size:1.65rem;font-weight:650;color:#293450;letter-spacing:-.04em;}
.action-number small {font-family:"Microsoft YaHei",sans-serif;font-size:.7rem;color:#929BB0;font-weight:400;margin-left:7px;}
.action-copy {font-size:.76rem;line-height:1.85;color:#838DA4;min-height:44px;margin:.3rem 0 .65rem;}
.st-key-home_value_card .stButton>button,.st-key-home_risk_card .stButton>button,
.st-key-home_group_card .stButton>button {background:#F7F8FD;border:0;color:#6570C9;}
.st-key-home_value_card .stButton>button:hover,.st-key-home_risk_card .stButton>button:hover,
.st-key-home_group_card .stButton>button:hover {background:#ECEFFB;color:#4653C5;}
@media(max-width:1100px) {
  .stMainBlockContainer {padding-left:1.8rem;padding-right:1.8rem;}
  [data-testid="stMetricValue"] {font-size:1.5rem;}
  .st-key-home_kpis [data-testid="stMetric"]::after {display:none;}
}
@media(max-width:700px) {
  .stMainBlockContainer {padding:3.5rem 1rem 2rem;} h1 {font-size:1.55rem!important;}
  [data-testid="stMetricValue"] {font-size:1.7rem;}
  .st-key-home_kpis [data-testid="stMetric"] {min-height:0;}
  .st-key-home_kpis [data-testid="stMetric"]::after {display:block;}
  .page-badge {font-size:.6rem;padding:3px 7px;}
  .header-meta {gap:.5rem;}
}
@media(prefers-reduced-motion:reduce) {* {scroll-behavior:auto!important;transition:none!important;}}
'''
