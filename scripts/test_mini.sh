#!/usr/bin/env bash
# =============================================================================
# SparseDrive —— nuScenes mini 单卡评测 (3D 时序检测)
#
# 配置: projects/configs/sparsedrive_small_det_mini.py
#   eval_mode = with_det=True, with_tracking=True, 其余全 False
#   -> 输出 NDS / mAP; 若结果里带 instance_ids 会顺带算 AMOTA 等跟踪指标
#
# 单卡直接走 tools/test.py (--launcher 默认 none -> single_gpu_test),
# 不用 tools/dist_test.sh, 省掉分布式进程组。
#
# 用法:
#   bash scripts/test_mini.sh                                  # 用训练产出的 latest.pth
#   bash scripts/test_mini.sh ckpt/xxx.pth                     # 指定权重
#   bash scripts/test_mini.sh ckpt/xxx.pth --out res.pkl       # 顺便存原始结果
#   bash scripts/test_mini.sh ckpt/xxx.pth --result_file res.pkl --show_only
#                                                              # 复用已存结果, 只重跑评测
#   bash scripts/test_mini.sh --help                           # 看 tools/test.py 全部参数
#
# 其余参数原样透传给 tools/test.py。
# =============================================================================
set -eo pipefail

cd "$(dirname "$0")/.."

# ---------- 环境检查 ----------
if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "[test_mini] 当前 shell 里没有可用的 torch, 请先激活环境:"
    echo "    conda activate sparsedrive"
    exit 1
fi

CONFIG=projects/configs/sparsedrive_small_det_mini.py
DEFAULT_CKPT=work_dirs/sparsedrive_small_det_mini/latest.pth

# --help 直接转交给 tools/test.py
if [ "${1:-}" = "--help" ]; then
    python3 tools/test.py --help
    exit 0
fi

# 第一个参数若以 .pth 结尾, 就当作权重; 否则用训练产出的 latest.pth
if [ $# -gt 0 ] && [[ "$1" == *.pth ]]; then
    CKPT=$1
    shift
else
    CKPT=$DEFAULT_CKPT
fi

if [ ! -f "$CKPT" ]; then
    echo "[test_mini] 找不到权重: $CKPT"
    echo "  先跑 bash scripts/train_mini.sh, 或把权重路径作为第一个参数传入:"
    echo "    bash scripts/test_mini.sh ckpt/your.pth"
    exit 1
fi
if [ ! -e data/infos/mini/nuscenes_infos_val.pkl ]; then
    echo "[test_mini] 缺少: data/infos/mini/nuscenes_infos_val.pkl"
    echo "  先执行数据转换 (见 scripts/train_mini.sh 头部注释)"
    exit 1
fi

echo "[test_mini] config : $CONFIG"
echo "[test_mini] ckpt   : $CKPT"
echo "[test_mini] 评测集 : mini_val (81 帧)"

python3 tools/test.py "$CONFIG" "$CKPT" --eval bbox "$@"
