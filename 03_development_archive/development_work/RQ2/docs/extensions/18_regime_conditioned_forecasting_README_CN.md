# RQ2 延伸：区域专用与平均洋流条件模型

## 研究问题

这段实验代码回答两个彼此独立的问题：

1. 使用各海洋环境自己的 training cases 训练区域专用模型，是否优于使用全部 training cases 的全局模型？
2. 向全局模型提供 forecast origin 的 GLORYS 长期平均流速，或者让模型只预测平均流轨迹的残差，是否改善预测？

为了避免把“算法差异”和“训练策略差异”混在一起，第一轮实验全部使用相同的 multi-output Ridge。确认训练策略有效后，再用 Random Forest 或 FNN 复验。

## 四种策略

- **Global original**：原始 98 维输入，使用全部 3,591 个 training cases。
- **Regional**：Shelf、Kuroshio Core、Transition 和 Other Offshore 分别拟合一个 Ridge。
- **Current-conditioned**：在 98 维输入后加入 forecast origin 的 `u_clim`、`v_clim` 和 `current_speed_clim`。
- **Current-residual**：先用 forecast origin 的长期平均流速生成基础位移，再让 Ridge 预测真实轨迹与该基础位移之间的残差。

每种策略都只在 training partition 内按 drifter 做 grouped 5-fold cross-validation。Calibration partition 未使用；固定 test partition 仅用于最终比较。

## 运行方法

在终端中运行：

```bash
cd "/Users/roger/Documents/data3001-project"
source .venv/bin/activate
mkdir -p /tmp/data3001-rq2-extension-mpl
export MPLCONFIGDIR=/tmp/data3001-rq2-extension-mpl
python RQ2/18_regime_conditioned_forecasting.py
```

主程序：

`RQ2/18_regime_conditioned_forecasting.py`

结果表格：

`RQ2/data/rq2_extension/`

结果图片：

`RQ2/figures/rq2_extension/`

## 第一轮结果

| Regime | Global Ridge RMSE | Regional Ridge RMSE | 区域化改善 | 95% drifter-clustered CI |
|---|---:|---:|---:|---:|
| Shelf | 46.35 km | 40.27 km | 13.11% | 2.52% 至 22.51% |
| Kuroshio Core | 92.08 km | 86.26 km | 6.32% | 2.18% 至 10.46% |
| Transition | 68.75 km | 68.67 km | 0.11% | -2.63% 至 2.98% |
| Other Offshore | 56.53 km | 54.36 km | 3.85% | 2.13% 至 5.45% |

初步结论：区域专用 Ridge 在 Shelf、Kuroshio Core 和 Other Offshore 上的 72-h RMSE 改善区间不包含零；Transition 几乎没有改善。这说明“区域模型一定更好”并不成立，收益取决于环境是否具有足够独特且稳定的动力规律。

加入长期平均流速后的结果：

- Shelf：Current-conditioned Ridge 改善 7.55%，95% CI 为 0.89% 至 12.59%。
- Kuroshio Core：改善 3.85%，但 CI 为 -0.37% 至 7.87%，证据尚不充分。
- Transition 和 Other Offshore：几乎没有明确改善。
- 简单的 origin-current residual 模型没有比直接加入平均流速产生更强证据。

## 解释限制

1. 这是 Ridge 的第一轮控制实验，不能直接推出 Random Forest 或 FNN 也会得到相同结果。
2. `Current-residual` 目前将 forecast origin 的长期平均 `u,v` 保持 72 小时不变；它不是逐步查询空间流场的完整 mean-flow advection。
3. 使用的是 1994–2022 全月份长期平均流场。后续可以建立 12 个月度气候平均流场，使模型获得季节信息。
4. Kuroshio Core 的高 RMSE 与高流速相关，但这里仍然没有证明高流速是唯一原因。
5. 本轮结果是在查看 test 结果后的探索性发现。若作为正式结论，最好预先固定方法后再进行额外空间或时间外推验证。

## 下一步建议

1. 先用 Random Forest 重复 Global、Regional 和 Current-conditioned 三种策略。
2. 把 residual baseline 改为每小时在预测位置查询当地平均 `u,v`，而不是固定 origin 流速。
3. 做 East China Sea 与 South Japan 两段黑潮之间的 leave-one-region-out 测试，以检查空间泛化能力。
4. 同时报告绝对 RMSE、normalized error 和相对于 persistence/mean-flow baseline 的 forecast skill。
