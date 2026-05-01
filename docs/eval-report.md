# Eval Report

## 设计目标

首版评测不是替代人工判断，而是验证系统是否满足三个底线：

- 能返回引用
- 回答有帮助
- 证据不足时不编造

## 当前指标

`EvalRunner` 输出：

- `helpfulness`
- `citation_validity`
- `grounded_pass_rate`

## Demo Suite

`demo_suite.json` 默认覆盖：

- 项目定位
- 安装说明
- 文档理解
- Issue 讨论
- PR 变更
- 证据不足时的拒答行为

## 判定逻辑

当前版本使用轻量启发式评测：

- `citation_validity`：回答是否带 citation
- `helpfulness`：回答是否覆盖预期关键词
- `grounded_pass_rate`：回答是否 grounded 或正确拒答

## 后续可扩展方向

- 引入 judge model 做答案质量打分
- 对 citation snippet 做 span-level 对齐
- 统计不同 source type 的召回率
- 对 trace 做 step-level latency 分析

