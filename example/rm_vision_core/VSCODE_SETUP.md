# VSCode C++ 索引配置指南

> 为 rm_vision_core 项目配置 VSCode，实现代码跳转、智能提示、调试功能

---

## 一、已配置的文件

```
/home/duang/vision/example/rm_vision_core/.vscode/
├── c_cpp_properties.json    # C++ IntelliSense 配置
├── settings.json            # 工作区设置
├── tasks.json               # 构建任务
├── launch.json              # 调试配置
└── extensions.json          # 推荐扩展
```

---

## 二、使用步骤

### 步骤 1：打开项目

```bash
# 在 VSCode 中打开项目根目录
code /home/duang/vision/example/rm_vision_core
```

### 步骤 2：安装推荐扩展

VSCode 会自动提示安装推荐扩展，点击"安装"即可。

或手动安装：
- **C/C++** (ms-vscode.cpptools)
- **C/C++ Extension Pack** (ms-vscode.cpptools-extension-pack)
- **CMake Tools** (ms-vscode.cmake-tools)
- **CMake** (twxs.cmake)
- **Better C++ Syntax** (jeff-hykin.better-cpp-syntax)

### 步骤 3：生成编译数据库

按 `Ctrl+Shift+P`，输入 `Tasks: Run Task`，选择 `cmake-configure`

或在终端执行：
```bash
cd /home/duang/vision/example/rm_vision_core
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

这会生成 `build/compile_commands.json`，用于 IntelliSense 索引。

### 步骤 4：等待索引完成

右下角会显示 "Parsing..." 进度，等待完成（首次可能需要 1-2 分钟）。

---

## 三、功能验证

### 3.1 代码跳转

- **跳转到定义**：`F12` 或 `Ctrl+点击`
- **查看定义**：`Alt+F12`（悬浮窗口）
- **跳转到声明**：`Ctrl+F12`
- **返回上一位置**：`Alt+←`

**测试**：
1. 打开 `modules/detector/rune_detector/src/rune_detector_match.cpp`
2. 找到 `RuneTracker::cast(trackers[i])`
3. `Ctrl+点击` `RuneTracker`，应该跳转到 `rune_tracker.h`

### 3.2 智能提示

输入代码时会自动弹出补全建议：
- 类成员
- 函数参数
- 命名空间

**测试**：
1. 打开任意 `.cpp` 文件
2. 输入 `cv::` 应该看到 OpenCV 的函数列表
3. 输入 `RuneFan::` 应该看到类的静态方法

### 3.3 查找引用

- **查找所有引用**：`Shift+F12`
- **查找符号**：`Ctrl+T`（全局搜索类/函数）

**测试**：
1. 打开 `modules/feature/rune_fan/include/vc/feature/rune_fan.h`
2. 右键 `find_active_fans` → "查找所有引用"
3. 应该显示所有调用此函数的位置

### 3.4 大纲视图

- 左侧边栏 → "大纲" 面板
- 显示当前文件的类、函数、变量结构

---

## 四、构建与调试

### 4.1 构建项目

**方法 1：使用任务**
- `Ctrl+Shift+P` → `Tasks: Run Build Task` → `cmake-build`

**方法 2：使用快捷键**
- `Ctrl+Shift+B`（默认构建任务）

**方法 3：终端命令**
```bash
cd /home/duang/vision/example/rm_vision_core
cmake --build build -j 8
```

### 4.2 调试程序

1. 打开 `examples/rune_detect_demo/main.cpp`
2. 在需要的行设置断点（点击行号左侧）
3. 按 `F5` 启动调试
4. 使用调试控制：
   - `F5`：继续
   - `F10`：单步跳过
   - `F11`：单步进入
   - `Shift+F11`：跳出

### 4.3 清理构建

`Ctrl+Shift+P` → `Tasks: Run Task` → `cmake-clean`

---

## 五、常用快捷键

| 功能 | 快捷键 |
|------|--------|
| 跳转到定义 | `F12` |
| 查看定义（悬浮） | `Alt+F12` |
| 查找所有引用 | `Shift+F12` |
| 全局搜索符号 | `Ctrl+T` |
| 文件内搜索符号 | `Ctrl+Shift+O` |
| 返回上一位置 | `Alt+←` |
| 前进到下一位置 | `Alt+→` |
| 构建项目 | `Ctrl+Shift+B` |
| 启动调试 | `F5` |
| 格式化代码 | `Shift+Alt+F` |
| 命令面板 | `Ctrl+Shift+P` |

---

## 六、配置说明

### 6.1 c_cpp_properties.json

配置了所有头文件搜索路径：
```json
"includePath": [
    "${workspaceFolder}/**",
    "${workspaceFolder}/common/core/include",
    "${workspaceFolder}/modules/detector/rune_detector/include",
    "/usr/include/opencv4",
    "/usr/include/eigen3",
    ...
]
```

### 6.2 settings.json

- 关联 `.h` 文件为 C++（而非 C）
- 使用 CMake 生成的 `compile_commands.json`
- 启用 IntelliSense 错误提示

### 6.3 tasks.json

定义了三个任务：
- `cmake-configure`：配置 CMake
- `cmake-build`：构建项目
- `cmake-clean`：清理构建

### 6.4 launch.json

配置了调试器：
- 程序路径：`build/examples/rune_detect_demo/rune_detect_demo`
- 调试器：gdb
- 构建前自动执行 `cmake-build`

---

## 七、常见问题

### Q1：代码无法跳转？

**解决方法**：
1. 检查右下角是否显示 "Parsing..."，等待完成
2. 确认 `build/compile_commands.json` 存在
3. 重新加载窗口：`Ctrl+Shift+P` → `Reload Window`
4. 重建索引：`Ctrl+Shift+P` → `C/C++: Reset IntelliSense Database`

### Q2：找不到头文件？

**解决方法**：
1. 检查 `.vscode/c_cpp_properties.json` 中的 `includePath`
2. 确认 OpenCV 和 Eigen3 已安装：
   ```bash
   pkg-config --cflags opencv4
   pkg-config --cflags eigen3
   ```
3. 如果路径不同，修改 `c_cpp_properties.json`

### Q3：构建失败？

**解决方法**：
1. 检查依赖是否安装：
   ```bash
   sudo apt install build-essential cmake
   sudo apt install libopencv-dev libeigen3-dev
   ```
2. 清理后重新构建：
   ```bash
   rm -rf build
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
   cmake --build build
   ```

### Q4：IntelliSense 提示错误但代码能编译？

**原因**：IntelliSense 和编译器的解析可能不一致

**解决方法**：
1. 确保 `c_cpp_properties.json` 中的 `cppStandard` 为 `c++20`
2. 检查 `compilerPath` 是否正确
3. 使用 `compile_commands.json`：
   ```json
   "compileCommands": "${workspaceFolder}/build/compile_commands.json"
   ```

---

## 八、推荐工作流

### 阅读代码流程

1. **从入口开始**：
   - 打开 `examples/rune_detect_demo/main.cpp`
   - 找到 `main()` 函数

2. **跟踪调用链**：
   - `F12` 跳转到 `RuneDetector::detect()`
   - 继续跟踪 `findFeatures()` → `extractRuneFeatures()` → ...

3. **查看类定义**：
   - `Ctrl+T` 输入类名（如 `RuneFan`）
   - 查看头文件了解接口

4. **查找使用位置**：
   - `Shift+F12` 查看函数在哪里被调用
   - 理解数据流向

### 调试流程

1. **设置断点**：
   - 在关键函数入口设置断点
   - 如 `RuneFanActive::make_feature()`

2. **启动调试**：
   - `F5` 启动
   - 观察变量值（鼠标悬停或"变量"面板）

3. **单步执行**：
   - `F10` 逐行执行
   - `F11` 进入函数内部
   - 理解算法逻辑

4. **条件断点**：
   - 右键断点 → "编辑断点"
   - 设置条件（如 `i == 3`）

---

## 九、进阶技巧

### 9.1 使用书签

安装扩展：**Bookmarks** (alefragnani.Bookmarks)

- `Ctrl+Alt+K`：切换书签
- `Ctrl+Alt+L`：跳转到下一个书签

### 9.2 代码片段

创建 `.vscode/cpp.code-snippets`：
```json
{
    "OpenCV Mat": {
        "prefix": "cvmat",
        "body": [
            "cv::Mat ${1:image} = cv::imread(\"${2:path}\");",
            "if (${1:image}.empty()) {",
            "    std::cerr << \"Failed to load image\" << std::endl;",
            "    return -1;",
            "}"
        ]
    }
}
```

### 9.3 多光标编辑

- `Ctrl+Alt+↓`：向下添加光标
- `Ctrl+D`：选中下一个相同单词
- `Alt+点击`：添加光标

### 9.4 代码折叠

- `Ctrl+Shift+[`：折叠当前区域
- `Ctrl+Shift+]`：展开当前区域
- `Ctrl+K Ctrl+0`：折叠所有
- `Ctrl+K Ctrl+J`：展开所有

---

## 十、学习路径

### 第一天：熟悉环境
1. 安装扩展
2. 生成编译数据库
3. 练习代码跳转

### 第二天：阅读主流程
1. 从 `main.cpp` 开始
2. 跟踪 `detect()` 流程
3. 理解数据流

### 第三天：深入核心算法
1. 阅读 `rune_fan_active.cpp`
2. 设置断点调试
3. 观察变量变化

### 第四天：理解匹配逻辑
1. 阅读 `rune_detector_match.cpp`
2. 理解代价矩阵
3. 跟踪 DFS 搜索

### 第五天：总结与实践
1. 绘制调用关系图
2. 修改参数观察效果
3. 尝试移植算法

---

**配置完成！** 现在你可以在 VSCode 中高效阅读 rm_vision_core 的代码了。

有任何问题随时问我！
