#!/usr/bin/env bash
# =============================================================================
# SparseDrive —— 仅 3D 时序检测, nuScenes mini, 单卡 / 低显存
#
# 配置: projects/configs/sparsedrive_small_det_mini.py
#   - with_det=True / with_map=False / with_motion_plan=False
#   - temporal=True, det_head num_temp_instances=600
#   - batch=1, 10 epoch = 3230 iters (GroupInBatchSampler 上限: mini 只有 16 个序列组)
#
# 前置依赖（缺任何一个下面会直接报出来，不会跑到一半才炸）:
#   ckpt/resnet50-19c8e357.pth                 backbone 预训练
#   data/nuscenes/                             nuScenes mini + can_bus + maps
#   data/infos/mini/nuscenes_infos_train.pkl   scripts/create_data.sh
#   data/kmeans/kmeans_det_900.npy             python tools/kmeans/kmeans_det.py
#
# 用法:
#   bash scripts/train_mini.sh                 # 完整训练 3230 iters
#   bash scripts/train_mini.sh --dry-run       # 只跑 50 iters, 验证链路是否通
#   GPUS=1 bash scripts/train_mini.sh          # 卡数(默认 1)
#
# 其余参数原样透传给 tools/train.py, 例如:
#   bash scripts/train_mini.sh --seed 1 --deterministic
#   bash scripts/train_mini.sh --work-dir work_dirs/my_run
#
# 4GB 显存实测(batch=1, 256x704): reserved 2.06 GiB, 可跑;
# batch=2 会 OOM。若还不够, 见配置文件头部注释的调优顺序。
# =============================================================================
set -eo pipefail

cd "$(dirname "$0")/.."

# ---------- 环境检查 ----------
if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "[train_mini] 当前 shell 里没有可用的 torch, 请先激活环境:"
    echo "    conda activate sparsedrive"
    exit 1
fi

CONFIG=projects/configs/sparsedrive_small_det_mini.py
GPUS=${GPUS:-1}

# ---------- 前置文件检查 ----------
missing=0
for f in \
    ckpt/resnet50-19c8e357.pth \
    data/infos/mini/nuscenes_infos_train.pkl \
    data/kmeans/kmeans_det_900.npy
do
    if [ ! -e "$f" ]; then
        echo "[train_mini] 缺少: $f"
        missing=1
    fi
done
if [ ! -d data/nuscenes ]; then
    echo "[train_mini] 缺少: data/nuscenes/  (nuScenes mini 数据)"
    missing=1
fi
if [ "$missing" = 1 ]; then
    echo
    echo "[train_mini] 请先按顺序准备以上文件:"
    echo "  1) 把 v1.0-mini.tgz / can_bus.zip / nuScenes-map-expansion-v1.3.zip"
    echo "     解压到 data/nuscenes/  (注意 map-expansion 要解到 maps/ 下)"
    echo "  2) python tools/data_converter/nuscenes_converter.py nuscenes \\"
    echo "         --root-path ./data/nuscenes --canbus ./data/nuscenes \\"
    echo "         --out-dir ./data/infos/ --extra-tag nuscenes --version v1.0-mini"
    echo "  3) python tools/kmeans/kmeans_det.py"
    exit 1
fi

# ---------- 参数解析 ----------
DRY_RUN=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--dry-run" ]; then
        DRY_RUN=1
    else
        ARGS+=("$a")
    fi
done

if [ "$DRY_RUN" = 1 ]; then
    echo "[train_mini] dry-run: 只跑 50 次迭代, 关闭验证与定期存档"
    ARGS+=(--no-validate
           --cfg-options
           runner.max_iters=50
           checkpoint_config.interval=1000
           evaluation.interval=1000)
fi

echo "[train_mini] config   : $CONFIG"
echo "[train_mini] gpus     : $GPUS"
echo "[train_mini] work_dir : work_dirs/$(basename "$CONFIG" .py)"

bash ./tools/dist_train.sh "$CONFIG" "$GPUS" "${ARGS[@]}"
