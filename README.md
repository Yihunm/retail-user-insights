# Retail User Insights

基于 Online Retail 历史交易的电商用户分析系统。

仓库：https://github.com/Yihunm/retail-user-insights

## 当前进度

按模块开发、验证并独立提交。已实现模块 01 数据导入与清洗，以及模块 02 交易统计与可视化。

RFM、聚类、复购/流失预测尚未实现。`tests/drafts/` 仅保存后续模块测试草稿，不参与当前测试。没有访问、浏览、收藏、加购日志，不生成这些虚构特征。

交易模块支持日期/国家筛选、商品销售额、订单数、可识别客户、平均订单金额、复购占比，以及月度趋势、商品排行、国家分布、星期/小时热力图和客户消费分布。当前筛选的五类分析表可下载为 ZIP；取消/退货候选按同一范围独立展示。

日期使用独立的“开始日期”和“结束日期”日历框，包含所选起止日。国家或地区可展开完整中文清单逐项勾选，支持中文/原始名称搜索、全选和清空；搜索不会改变隐藏项的选择。图表、悬浮提示、分析表和导出字段采用中文，商品编码与原始商品名保留原始内容。

## 本地启动

Python 3.11 以上；当前已验证 Windows / Python 3.13。首次安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

已经安装好依赖时：

```powershell
.\start.ps1
```

或运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

打开 http://127.0.0.1:8501 。选择 `Online Retail.xlsx` 上传，或在本机文件路径中填写数据位置，点击“导入并检查”。本机路径功能仅用于可信的本地单人演示；上线时需移除或限制。

如 PowerShell 禁止脚本执行，使用上面的 Python 命令即可，无须更改全局执行策略。

项目不附带原始数据。请自行从 UCI Online Retail 数据集获取并遵守其使用条件。已在本机导入的数据、模型和虚拟环境均不提交 Git。

## 命令行导入

```powershell
.\.venv\Scripts\python.exe -m retail_insights.cli prepare --input "你的路径\Online Retail.xlsx"
```

默认按八个原始字段移除完全重复行。需要保留时追加 `--keep-duplicates`。可追加 `--root "输出目录"` 将产物保存在其他指定目录。

输入支持 `.xlsx` 和 UTF-8 / UTF-8-BOM 编码的 `.csv`；默认 Excel 第一张表。Python API `read_transactions(..., sheets=['Sheet1', 'Sheet2'])` 支持指定同结构工作表。当前页面和 CLI 为单文件导入；多文件批量接入留到数据模块的后续增量。

## 清洗规则

- 保留原始八字段，编号使用字符串。客户编号缺失的有效销售仍然保留。
- 销售、取消/退货候选和隔离记录互斥；移除重复行独立计数。
- 取消订单正数量、非正价格、异常时间/数值、缺订单或商品编码进入隔离表，并记录原因。
- 数值型日期没有明确单位时不猜测；带时区时间目前不支持，均进入隔离表。
- 特殊业务编码 `POST,D,DOT,M,BANK CHARGES,C2,AMAZONFEE,CRUK` 暂按配置隔离，原记录可追溯；不是确认已匹配原订单的真实退款。
- 金额使用 Decimal HALF_UP 逐行取整数分，汇总 Python 整数，避免浮点误差和 Int64 汇总溢出。
- 原始行数 = 移除重复行 + 销售行 + 取消/退货候选行 + 隔离行。
- 质量报告的问题计数来自去重后数据，可能重叠，不能直接相加。

## 数据与版本

```text
data/raw/<SHA256>.xlsx              原始副本
data/processed/<版本>/sales.parquet
data/processed/<版本>/returns.parquet
data/processed/<版本>/quarantine.parquet
data/processed/<版本>/quality.json
data/processed/<版本>/metadata.json
data/catalog.sqlite3                导入登记
data/active.json                    当前完整版本
```

版本由文件内容、清洗规则版本及去重策略决定。同版本复用，不重复读取 Excel。导入失败不切换当前版本。页面读取当前版本并缓存，训练功能尚未开放。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
```

受限环境若不允许默认临时目录，可用 `--basetemp=.tmp/pytest-local` 指定仓库内临时目录。页面测试需要允许本地回环通信。

模块验收记录见 [docs/module-01-validation.md](docs/module-01-validation.md)。

交易模块验收见 [docs/module-02-validation.md](docs/module-02-validation.md)。
