## Summary

修复音频生成断续、缺失问题 (Issue #2)

### 修改内容

1. **随机种子问题修复**
   - 将固定的随机种子 `12345` 改为动态生成（使用系统时间 + 熵源）
   - 支持用户通过 CLI 参数 `--seed` 指定种子以实现可复现性

2. **解码器边界处理修复**
   - 将 codes 截断为 16 的倍数（一帧 = 16 个 codes）
   - 使用 `clamp(0, 2047)` 替代 `min(2047)` 确保值在有效范围内
   - 保留未满一帧的剩余 codes 供下次迭代处理

3. **采样参数暴露**
   - 新增 `SamplerConfig` 结构体
   - CLI 新增参数：`--temperature`, `--top_k`, `--top_p`, `--seed`
   - 支持运行时调整采样策略

4. **代码质量**
   - 修复所有 clippy 警告
   - 消除编译警告

### 测试建议

```bash
# 使用默认参数
.\qwen3_tts.exe --text "测试文本" --output "test.wav" --speaker sohee

# 调整温度（更高温度 = 更随机）
.\qwen3_tts.exe --text "测试文本" --output "test.wav" --speaker sohee --temperature 0.7

# 使用固定种子复现结果
.\qwen3_tts.exe --text "测试文本" --output "test.wav" --speaker sohee --seed 12345
```

Closes #2
