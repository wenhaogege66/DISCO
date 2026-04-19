#!/bin/bash
# 启动 TensorBoard 查看 TIGER 训练曲线
# 用法: bash TIGER/scripts/start_tensorboard.sh [port]

PORT=${1:-6008}
LOGDIR="/home/sjj/wenhao/TIGER/tensorboard"

echo "启动 TensorBoard..."
echo "日志目录: $LOGDIR"
echo "访问地址: http://localhost:${PORT}"
echo ""
echo "如需远程访问，请在本地运行:"
echo "  ssh -L ${PORT}:localhost:${PORT} sjj@$(hostname)"
echo "  然后在浏览器打开: http://localhost:${PORT}"
echo ""

tensorboard --logdir="$LOGDIR" --port="$PORT" --bind_all
