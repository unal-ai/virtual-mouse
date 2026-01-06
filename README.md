# AI Virtual Mouse 🖱️✋

用摄像头手势控制电脑鼠标的跨平台应用。基于 MediaPipe 手部追踪和 PyAutoGUI 鼠标控制。

**适用场景**：公司形象屏、展览互动屏等，让参观者通过手势操作屏幕内容。

## 功能特点

- ✅ **跨平台**：支持 macOS、Windows、Linux
- ✅ **多种视频输入**：内置摄像头、USB 外接摄像头、RTSP 视频流
- ✅ **实时手势识别**：基于 MediaPipe 的高精度手部追踪
- ✅ **光标平滑**：自适应阻尼算法减少抖动
- ✅ **可配置**：通过 YAML 文件自定义所有参数

## 手势操作

| 手势       | 动作         | 说明                        |
| ---------- | ------------ | --------------------------- |
| ✌️ V手势    | 移动光标     | 食指+中指张开               |
| ✌️ 两指并拢 | **左键点击** | 从V手势合拢两指（最简单！） |
| ☝️ 仅食指   | 右键点击     | 从V手势放下中指             |
| 🤏 捏合     | 双击         | 拇指+食指捏合               |
| ✊ 握拳     | 拖拽         | 握拳并移动                  |
| 🤏 左手捏合 | 滚动         | 左手拇指+食指捏合后上下移动 |
| 🖐 手掌张开 | 停止         | 五指全部张开                |

## 安装

### 前置要求

- Python 3.8+
- 摄像头（内置或外接）

### 安装依赖

```bash
cd /Users/david/gitrepos/virtual_mouse

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### macOS 权限设置

首次运行需要授予终端/Python **辅助功能权限**：

1. 打开 **系统偏好设置** → **安全性与隐私** → **隐私**
2. 选择 **辅助功能**
3. 添加 **终端** 或 **iTerm** 或 **Python**

## 使用方法

### 基本使用

```bash
# 使用默认摄像头运行
python -m src.main

# 按 ESC 或 Enter 退出
```

### 命令行参数

```bash
# 使用外接 USB 摄像头（索引 1）
python -m src.main --source 1

# 使用 RTSP 视频流
python -m src.main --source "rtsp://192.168.1.100:554/stream"

# 无预览窗口模式（后台运行）
python -m src.main --headless

# 不显示手部骨架
python -m src.main --no-landmarks

# 使用自定义配置文件
python -m src.main --config /path/to/config.yaml
```

### 配置文件

编辑 `config.yaml` 自定义设置：

```yaml
camera:
  source: 0                    # 摄像头索引或 RTSP URL
  flip_horizontal: true        # 镜像模式

gesture:
  detection_confidence: 0.7    # 检测置信度
  smoothing_factor: 0.3        # 光标平滑度

display:
  show_preview: true           # 显示预览窗口
  show_landmarks: true         # 显示手部骨架
```

## 项目结构

```
virtual_mouse/
├── config.yaml          # 配置文件
├── requirements.txt     # 依赖列表
├── README.md           # 说明文档
└── src/
    ├── __init__.py
    ├── main.py          # 主入口
    ├── config.py        # 配置加载器
    ├── video_input.py   # 视频输入（摄像头/RTSP）
    ├── gestures.py      # 手势识别
    └── controller.py    # 鼠标控制
```

## 故障排除

### 摄像头打不开

```bash
# 检查摄像头是否被占用
lsof | grep -i camera

# macOS: 重置摄像头权限
tccutil reset Camera
```

### 鼠标控制无反应（macOS）

确保已授予辅助功能权限，见上方安装说明。

### RTSP 连接不稳定

程序会自动重连，最多尝试 5 次。可在 `video_input.py` 中调整重连参数。

## 许可证

MIT License
