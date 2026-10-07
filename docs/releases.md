# 发行记录

## 0.4.0

新增可选 `performance.compile_scope="training_blocks"`：对完整动力学子步循环、anchor 写入、
Gaussian NLL/KL 的末维均值使用 PyTorch 编译融合。默认 `compile=false`；已有
`compile=true` 配置仍按默认 `compile_scope="predictor"` 执行。

- 四个目标在所选计算精度下通过前向、反向和有限性预检后一起挂接；保留参数名、随机状态、
  持久状态和时间轴契约。未增加包级 CUDA/Triton 依赖，也不编译 decoder。
- 新增 `compile_fallback=false`。只有显式设为 `true` 才允许预检失败时警告并记录参考回退；
  训练启动后的编译错误会停止，不在中途切换策略。
- 检查点记录实际生效的编译范围，拒绝不同范围之间的续训；兼容旧版参考检查点与
  0.3.0 未记录范围的 predictor 编译检查点。
- 静态形状可能因 batch、积分步数或计算精度变化触发重编译。需要支持
  `emulate_precision_casts` 的编译后端；参考路径继续支持原有依赖范围。

配置、数值约定与测量方法见 [性能文档](performance.md)。发行包仅包含通用代码、文档、
示例与测试；实验数据、配置、测量日志和训练权重由调用者在包外管理。

## 0.3.0

加入编码器批处理、主机侧积分规划、训练统计延迟同步、decoder 批量传输与目标关键帧缓存，
以及可选 BF16、SDPA、fused AdamW、predictor 编译和 pinned/非阻塞传输。
