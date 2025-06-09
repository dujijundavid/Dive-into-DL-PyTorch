# Cursor AI 核心实践指南 🎯

## 目录

1. [安装与初始设置](#安装与初始设置)
2. [核心功能掌握](#核心功能掌握)
3. [提示工程最佳实践](#提示工程最佳实践)
4. [工作流程优化](#工作流程优化)
5. [代码规则配置](#代码规则配置)
6. [常见问题解决](#常见问题解决)

---

## 安装与初始设置

### 1. 下载安装

```bash
# 从官网下载 Cursor AI
# https://www.cursor.sh/

# Windows: cursor-setup.exe
# macOS: Cursor.dmg  
# Linux: cursor.appimage
```

### 2. 首次配置

#### 基本设置优化
```json
{
  "editor.quickSuggestions": {
    "other": true,
    "comments": false,
    "strings": true
  },
  "cursor.enableAutoEdit": true,
  "cursor.enableAutoCompletion": true,
  "cursor.chat.alwaysSearchWeb": false
}
```

#### 快捷键配置
- `Ctrl/Cmd + K`: 打开 Command K 编辑模式
- `Ctrl/Cmd + L`: 打开 Chat 聊天面板
- `Ctrl/Cmd + I`: 打开 Inline Chat
- `Tab`: 接受 AI 代码建议
- `Escape`: 拒绝 AI 建议

---

## 核心功能掌握

### 1. Tab 补全功能

#### 最佳实践
```python
# ✅ 好的做法 - 提供上下文
def calculate_user_discount(user_type, purchase_amount):
    # AI 会根据函数名和参数推测逻辑
    if user_type == "premium":  # Tab 这里会智能补全
```

```python
# ❌ 避免的做法 - 缺乏上下文
def func(x, y):
    # AI 难以推测意图
    return  # Tab 补全效果差
```

#### 提升 Tab 补全效果的技巧
1. **命名规范**: 使用描述性的变量名和函数名
2. **注释引导**: 在复杂逻辑前添加注释
3. **类型提示**: 使用 TypeScript 或 Python 类型注解
4. **上下文保持**: 保持相关代码在可见区域

### 2. Chat 模式对话

#### 有效的提示结构
```
角色定义 + 具体任务 + 上下文信息 + 期望输出

示例：
作为一个 React 专家，帮我创建一个用户登录组件。
需要包含：
- 用户名和密码输入框
- 表单验证
- 提交按钮
- 错误信息显示
请使用 TypeScript 和 Tailwind CSS。
```

#### Chat 最佳实践
1. **具体明确**: 避免模糊的描述
2. **分步骤**: 复杂任务分解为小步骤
3. **提供示例**: 给出期望的输入/输出示例
4. **迭代优化**: 基于结果调整后续询问

### 3. Command K 编辑

#### 常用命令模式
- **添加功能**: "添加错误处理逻辑"
- **重构代码**: "将这个函数分解为更小的函数"
- **优化性能**: "优化这个循环的性能"
- **添加测试**: "为这个函数添加单元测试"

#### Command K 技巧
```javascript
// 选中代码块，使用 Ctrl+K
function processUserData(users) {
    // 选中这段代码
    return users.map(user => user.name);
}

// 输入: "添加用户验证和错误处理"
// AI 会智能重构选中的代码
```

---

## 提示工程最佳实践

### 1. 优质提示词模板

#### 代码生成模板
```
作为一个 [技术栈] 专家，请帮我 [具体任务]。

要求：
- [功能需求1]
- [功能需求2] 
- [技术约束]

代码风格：
- [编码规范]
- [命名约定]

请提供完整的代码和简要说明。
```

#### 代码优化模板
```
请优化以下 [语言] 代码：

[代码块]

优化目标：
- 性能提升
- 可读性改善
- 遵循最佳实践

请说明优化的理由。
```

### 2. 上下文管理策略

#### 使用 @docs 功能
```
@docs 请根据 React 官方文档，创建一个符合最新最佳实践的组件
```

#### 使用 @web 功能
```  
@web 查找 2024 年最新的 Next.js 14 性能优化技巧
```

#### 使用 @codebase 功能
```
@codebase 分析当前项目的代码结构，建议如何重构用户认证模块
```

---

## 工作流程优化

### 1. 项目初始化流程

```markdown
## 新项目 Cursor AI 设置清单

### 第一步：项目配置
- [ ] 创建 .cursorrules 文件
- [ ] 配置项目特定的代码风格
- [ ] 设置排除文件列表

### 第二步：工作区设置  
- [ ] 配置快捷键偏好
- [ ] 设置 AI 模型偏好
- [ ] 启用相关扩展

### 第三步：团队同步
- [ ] 分享 .cursorrules 文件
- [ ] 统一代码风格标准
- [ ] 制定 AI 使用规范
```

### 2. 日常开发工作流

#### 功能开发流程
1. **需求分析**: 使用 Chat 模式讨论需求
2. **代码生成**: 利用 Tab 补全快速编码
3. **代码优化**: 使用 Command K 重构
4. **测试编写**: AI 辅助生成测试用例
5. **文档更新**: 自动生成文档注释

#### 调试解决流程
1. **错误识别**: 粘贴错误信息到 Chat
2. **原因分析**: AI 分析可能原因
3. **解决方案**: 获得修复建议
4. **代码修复**: 应用修复方案
5. **验证测试**: 确认问题解决

---

## 代码规则配置

### 1. .cursorrules 文件示例

#### 通用规则
```yaml
# .cursorrules
项目类型: Web应用
主要技术栈: React, TypeScript, Node.js

编码规范:
  - 使用函数式组件和 Hooks
  - 优先使用 TypeScript 严格模式
  - 遵循 ESLint 和 Prettier 配置
  - 使用语义化的变量和函数命名

代码风格:
  - 单引号字符串
  - 尾随逗号
  - 2空格缩进
  - 行末不留分号

注释要求:
  - 复杂逻辑必须添加注释
  - 函数需要 JSDoc 文档
  - 重要的业务逻辑需要说明

测试要求:
  - 新功能需要单元测试
  - 使用 Jest 和 React Testing Library
  - 测试覆盖率不低于 80%
```

#### 框架特定规则
```yaml
# React 项目规则
React最佳实践:
  - 使用 functional components
  - 合理使用 useCallback 和 useMemo
  - 状态提升和组件拆分
  - 避免在 render 中创建新对象

性能优化:
  - 懒加载组件 (React.lazy)
  - 图片优化和懒加载
  - Bundle 拆分和代码分割
  - 使用 React.memo 防止不必要渲染
```

### 2. 项目特定配置

#### API 项目配置
```yaml
API开发规范:
  - RESTful 接口设计
  - 统一错误处理格式
  - 输入验证和数据清理
  - 完整的 API 文档
  - 安全认证实现
```

#### 前端项目配置
```yaml
前端开发规范:
  - 响应式设计原则
  - 无障碍访问支持
  - 浏览器兼容性考虑
  - 性能监控集成
  - SEO 优化实现
```

---

## 常见问题解决

### 1. 性能优化

#### Tab 补全延迟问题
```json
// settings.json 优化配置
{
  "cursor.cpp.disableRealTimeChecking": true,
  "cursor.general.enableLogging": false,
  "cursor.chat.enableContextualMessages": true
}
```

#### 大文件处理优化
- 排除不必要的文件 (node_modules, dist)
- 使用 .cursorignore 文件
- 限制上下文窗口大小

### 2. 功能使用问题

#### Chat 响应不准确
- 提供更多上下文信息
- 使用 @docs/@web 获取最新信息
- 分解复杂问题为简单问题
- 明确指定技术版本

#### 代码生成质量问题
- 完善 .cursorrules 配置
- 提供更好的代码示例
- 使用渐进式的请求方式
- 及时修正和反馈

### 3. 团队协作问题

#### 统一团队标准
```bash
# 团队配置同步脚本
cp team-configs/.cursorrules ./
cp team-configs/cursor-settings.json ./.cursor/
```

#### 版本控制集成
```gitignore
# .gitignore 建议配置
.cursor/
*.cursor-*
.ai-context/
```

---

## 学习进阶路径

### 第一阶段 (1周)
- [ ] 熟练使用 Tab 补全
- [ ] 掌握基本 Chat 对话
- [ ] 配置个人偏好设置

### 第二阶段 (2-3周)  
- [ ] 编写高质量提示词
- [ ] 配置项目 .cursorrules
- [ ] 集成到日常开发流程

### 第三阶段 (1个月)
- [ ] 优化团队协作流程
- [ ] 建立最佳实践模板
- [ ] 探索高级功能特性

---

## 总结

掌握这些核心实践将帮助您：

1. **提升开发效率**: 减少重复劳动，专注核心逻辑
2. **改善代码质量**: AI 辅助重构和优化
3. **加快学习速度**: 快速理解新技术和框架
4. **增强团队协作**: 统一开发标准和最佳实践

继续阅读 `02_ADVANCED_TRICKS.md` 了解更多高级技巧! 🚀 