# ================ base config ===================
version = 'mini'      # 数据集版本，是下面 length 字典的 key；决定用 mini 还是 trainval
version = 'trainval'  # 后写的生效，会覆盖上一行；影响 anno_root / num_iters_per_epoch
length = {'trainval': 28130, 'mini': 323}  # 两个版本各自的训练样本(关键帧)数量

plugin = True                            # 是否加载外部插件(本仓库的 mmdet3d_plugin)
plugin_dir = "projects/mmdet3d_plugin/"  # 插件目录，其中的模型/数据集会被自动注册
dist_params = dict(backend="nccl")       # 分布式通信后端，NVIDIA GPU 用 nccl
log_level = "INFO"                       # 日志级别
work_dir = None                          # 输出目录；None 时自动用 work_dirs/<配置文件名>

total_batch_size = 64                       # 全局批大小 = 所有卡的样本数之和
num_gpus = 8                                # GPU 数量
batch_size = total_batch_size // num_gpus   # 单卡批大小 = 64 // 8 = 8
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))  # 一个 epoch 的迭代数 = 28130 // 64 = 439
num_epochs = 100                            # 训练总轮数
checkpoint_epoch_interval = 20              # 每隔多少个 epoch 存一次 checkpoint

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval  # 存档间隔，按迭代数计 = 439 * 20
)
log_config = dict(
    interval=51,                                      # 每 51 次迭代往日志/终端写一行
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),  # 文本日志，按 iteration 计数(不按 epoch)
        dict(type="TensorboardLoggerHook"),           # TensorBoard 日志
    ],
)
load_from = None            # 预训练权重路径；不为 None 时训练前加载(微调用，如 stage2 加载 stage1)
resume_from = None          # 断点续训路径；除权重外还会恢复 optimizer / lr / 迭代进度
workflow = [("train", 1)]   # 执行流程：跑 1 个 train 阶段
fp16 = dict(loss_scale=32.0)  # 开启混合精度训练，loss 初始缩放系数 32，约省一半显存
input_shape = (704, 256)      # 输入图像尺寸，含义是 (宽, 高)；下面会反转成 (高, 宽) 给 final_dim


# ================== model ========================
class_names = [              # 检测的 10 个类别；列表下标即 label id
    "car",                   # 0 小汽车
    "truck",                 # 1 卡车
    "construction_vehicle",  # 2 工程车
    "bus",                   # 3 公交车
    "trailer",               # 4 挂车
    "barrier",               # 5 路障
    "motorcycle",            # 6 摩托车
    "bicycle",               # 7 自行车
    "pedestrian",            # 8 行人
    "traffic_cone",          # 9 交通锥
]
map_class_names = [          # 在线建图的 3 个类别；列表下标即 label id
    'ped_crossing',          # 0 人行横道
    'divider',               # 1 车道线 / 道路分隔线
    'boundary',              # 2 可行驶区域边界
]
num_classes = len(class_names)          # = 10，检测分类头的输出维度
num_map_classes = len(map_class_names)  # = 3，建图分类头的输出维度
roi_size = (30, 60)  # BEV 兴趣区域(米)：x 向 30m、y 向 60m；用于裁剪地图 GT，必须与 GT 生成时一致

num_sample = 20    # 每条地图折线重采样成 20 个点(建图 query 的输出点数)
fut_ts = 12        # 运动预测的未来帧数(0.5s/帧 -> 共 6s)
fut_mode = 6       # 运动预测的轨迹模式数，即每个目标预测 6 条候选未来轨迹
ego_fut_ts = 6     # 规划的未来帧数(0.5s/帧 -> 共 3s)
ego_fut_mode = 6   # 规划的轨迹模式数
queue_length = 4   # 实例队列长度(历史 + 当前帧)，供 motion head 做更长的时序建模

embed_dims = 256              # 主干特征 / query 的通道数
num_groups = 8                # 分组数：deformable 注意力的组数，同时也是多头注意力的头数
num_decoder = 6               # decoder 层数(检测头与建图头共用)
num_single_frame_decoder = 1      # 检测头前 N 层 decoder 不做时序(纯单帧)，其余层才做时序融合
num_single_frame_decoder_map = 1  # 建图头的单帧 decoder 层数
use_deformable_func = True  # 使用编译好的 deformable_aggregation CUDA 算子(需先执行 mmdet3d_plugin/ops/setup.py)
strides = [4, 8, 16, 32]    # FPN 各层相对原图的降采样倍率
num_levels = len(strides)   # = 4，FPN 输出层数，也是 deformable 采样用到的特征层数
num_depth_layers = 3        # 辅助深度监督只取前 3 个尺度
drop_out = 0.1              # 通用 dropout 概率
temporal = True             # 检测头是否启用时序(实例记忆队列)
temporal_map = True         # 建图头是否启用时序
decouple_attn = True        # 检测头是否把 query 与 query_pos 拼接后再做注意力(通道翻倍)
decouple_attn_map = False   # 建图头是否解耦注意力
decouple_attn_motion = True # 运动/规划头是否解耦注意力
with_quality_estimation = True  # 检测框是否额外预测质量分(用于排序/重打分)

task_config = dict(
    with_det=True,           # 构建并训练检测头(Sparse4DHead)
    with_map=True,           # 构建并训练建图头(Sparse4DHead)
    with_motion_plan=False,  # 不构建运动预测/规划头(stage1 只学感知，stage2 才打开)
)

model = dict(
    type="SparseDrive",                          # 检测器类名(projects/mmdet3d_plugin/models/sparsedrive.py)
    use_grid_mask=True,                          # 训练时对输入图像做网格掩码(GridMask)增强
    use_deformable_func=use_deformable_func,     # 透传给各 head，决定用 CUDA 算子还是纯 PyTorch 采样
    img_backbone=dict(                           # 图像主干网络
        type="ResNet",                           # 骨干类型
        depth=50,                                # ResNet 深度，50 层
        num_stages=4,                            # 4 个 stage(res1~res4)
        frozen_stages=-1,                        # -1 表示不冻结任何 stage，全部参与训练
        norm_eval=False,                         # False 表示训练时更新 BN 的统计量(不冻结 BN)
        style="pytorch",                         # 卷积/padding 风格
        with_cp=True,                            # 使用梯度检查点(checkpoint)，省显存换计算时间
        out_indices=(0, 1, 2, 3),                # 输出这 4 个 stage 的特征图给 neck
        norm_cfg=dict(type="BN", requires_grad=True),  # 归一化层配置：BatchNorm，参数可学习
        pretrained="ckpt/resnet50-19c8e357.pth",       # 骨干预训练权重路径
    ),
    img_neck=dict(                               # 图像颈部，负责多尺度特征融合
        type="FPN",                              # 特征金字塔
        num_outs=num_levels,                     # 输出层数 = 4
        start_level=0,                           # 从 backbone 的第 0 个输出开始接
        out_channels=embed_dims,                 # 每层输出通道数 = 256
        add_extra_convs="on_output",             # 额外层的卷积接在输出上(用于生成第 4 层)
        relu_before_extra_convs=True,            # 额外卷积前是否先过 ReLU
        in_channels=[256, 512, 1024, 2048],      # backbone 4 个 stage 的输入通道数
    ),
    depth_branch=dict(  # for auxiliary supervision only  仅做辅助监督(不参与推理)
        type="DenseDepthNet",                    # 稠密深度预测分支
        embed_dims=embed_dims,                   # 输入通道数
        num_depth_layers=num_depth_layers,       # 用前 3 个尺度的特征做深度预测
        loss_weight=0.2,                         # 深度损失的权重
    ),
    head=dict(
        type="SparseDriveHead",              # 统一的多任务 head，内部按 task_config 构建子 head
        task_config=task_config,             # 任务开关(检测/建图/运动规划)
        det_head=dict(                       # ---------- 3D 检测头 ----------
            type="Sparse4DHead",             # 稀疏 query 式检测头
            cls_threshold_to_reg=0.05,       # 分类分数超过该阈值的 query 才参与锚框回归
            decouple_attn=decouple_attn,     # 是否解耦注意力(query 与 query_pos 拼接输入)
            instance_bank=dict(              # 实例记忆库：时序建模的核心(跨帧保存实例特征/锚框)
                type="InstanceBank",         # 实例库实现
                num_anchor=900,              # 每帧的检测 query 数(也是锚框数)
                embed_dims=embed_dims,       # 实例特征的通道数
                anchor="data/kmeans/kmeans_det_900.npy",  # K-means 生成的 900 个初始锚框(由 kmeans_det.py 产出)
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),  # 锚框->关键点，用于算采样位置
                num_temp_instances=600 if temporal else -1,  # 每帧保留 600 个历史实例传给下一帧；-1 表示关闭时序
                confidence_decay=0.6,        # 历史实例置信度的衰减系数(每帧乘一次)
                feat_grad=False,             # 缓存的历史特征不参与反向传播(detach)
            ),
            anchor_encoder=dict(              # 锚框编码器：把 3D 框编码成特征并注入 query
                type="SparseBox3DEncoder",    # 3D 框编码器实现
                vel_dims=3,                   # 需要编码的速度/朝向等维度数
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,  # 各分量的通道划分，和 = 256
                mode="cat" if decouple_attn else "add",  # 拼接还是相加注入 query
                output_fc=not decouple_attn,  # 是否在末尾再加一层全连接
                in_loops=1,                   # 内部的重复次数
                out_loops=4 if decouple_attn else 2,  # 输出层数
            ),
            num_single_frame_decoder=num_single_frame_decoder,  # 前 N 层不做时序(纯单帧)
            operation_order=(                 # decoder 每层按顺序执行哪些算子
                [
                    "gnn",         # 实例间自注意力(当前帧 900 个 query 内部做注意力)
                    "norm",        # 层归一化
                    "deformable",  # 可变形特征聚合(按关键点从 6 路相机特征上采样)
                    "ffn",         # 前馈网络
                    "norm",        # 层归一化
                    "refine",      # 锚框回归(更新 3D 框)
                ]
                * num_single_frame_decoder   # 复制成「单帧层」
                + [
                    "temp_gnn",    # 时序注意力(与 InstanceBank 给出的历史实例交互)
                    "gnn",         # 实例间自注意力
                    "norm",        # 层归一化
                    "deformable",  # 可变形特征聚合
                    "ffn",         # 前馈网络
                    "norm",        # 层归一化
                    "refine",      # 锚框回归
                ]
                * (num_decoder - num_single_frame_decoder)  # 复制成「时序层」
            )[2:],  # 丢掉列表最前面的 2 个算子("gnn"、"norm")，让第 1 层直接从 deformable 开始
            temp_graph_model=dict(                       # 时序注意力模块(对应 op "temp_gnn")
                type="MultiheadFlashAttention",          # 注意力实现；无 flash-attn 时自动回退为 PyTorch 原生实现
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,  # 输入通道(解耦时为 2 倍)
                num_heads=num_groups,                    # 注意力头数 = 8
                batch_first=True,                        # 张量布局为 (batch, seq, dim)
                dropout=drop_out,                        # 注意力 dropout
            )
            if temporal                                  # 只有开启时序才构建该模块
            else None,
            graph_model=dict(                            # 帧内注意力模块(对应 op "gnn")
                type="MultiheadFlashAttention",          # 同上
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,  # 输入通道
                num_heads=num_groups,                    # 注意力头数
                batch_first=True,                        # (batch, seq, dim)
                dropout=drop_out,                        # 注意力 dropout
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),  # LayerNorm，按最后一维归一化
            ffn=dict(                                    # 前馈网络(对应 op "ffn")
                type="AsymmetricFFN",                    # 输入输出通道可不同的 FFN
                in_channels=embed_dims * 2,              # 输入通道 = 512(解耦注意力时 query 与 pos 拼接过)
                pre_norm=dict(type="LN"),                # 先做 LayerNorm 再进全连接
                embed_dims=embed_dims,                   # 中间层通道数 = 256
                feedforward_channels=embed_dims * 4,     # 中间隐层通道数 = 1024
                num_fcs=2,                               # 全连接层数
                ffn_drop=drop_out,                       # dropout 概率
                act_cfg=dict(type="ReLU", inplace=True), # 激活函数
            ),
            deformable_model=dict(                  # 可变形特征聚合(对应 op "deformable")，从多相机多尺度特征采样
                type="DeformableFeatureAggregation",  # 实现类
                embed_dims=embed_dims,              # query 通道数
                num_groups=num_groups,              # 通道分组数，同一组内的通道共享一个采样权重
                num_levels=num_levels,              # 在每个相机上采样几层特征(FPN 层数 = 4)
                num_cams=6,                         # 相机数量(nuScenes 前/前左/前右/后/后左/后右)
                attn_drop=0.15,                     # 采样权重的 dropout 概率
                use_deformable_func=use_deformable_func,  # 用 CUDA 算子加速(否则走纯 PyTorch 的 grid_sample)
                use_camera_embed=True,              # 是否把相机内外参编码后加到权重预测里(帮网络判断点在哪个相机)
                residual_mode="cat",                # 输出与输入 query 的融合方式："cat" 拼接(通道翻倍) / "add" 相加
                kps_generator=dict(                 # 关键点生成器：由 3D 框算出要采样的 3D 点
                    type="SparseBox3DKeyPointsGenerator",  # 以 3D 框为基准生成关键点
                    num_learnable_pts=6,            # 额外由网络预测 6 个偏移点
                    fix_scale=[                     # 固定的关键点偏移，单位是「框尺寸的比例」
                        [0, 0, 0],                  # 框中心
                        [0.45, 0, 0],               # 沿 x 正向 0.45 倍
                        [-0.45, 0, 0],              # 沿 x 负向
                        [0, 0.45, 0],               # 沿 y 正向
                        [0, -0.45, 0],              # 沿 y 负向
                        [0, 0, 0.45],               # 沿 z 正向
                        [0, 0, -0.45],              # 沿 z 负向
                    ],                              # 共 7 个固定点 + 6 个可学习点 = 13 个采样点
                ),
            ),
            refine_layer=dict(                      # 锚框回归层(对应 op "refine")，输出最终 3D 框
                type="SparseBox3DRefinementModule", # 实现类
                embed_dims=embed_dims,              # 输入特征通道数
                num_cls=num_classes,                # 分类输出维度 = 10
                refine_yaw=True,                    # 是否把朝向(yaw)也作为回归量
                with_quality_estimation=with_quality_estimation,  # 是否额外预测质量分(centerness/yawness)
            ),
            sampler=dict(                           # 目标分配器：把 GT 框匹配到 query 上，并生成回归目标
                type="SparseBox3DTarget",           # 匈牙利匹配 + denoising 目标
                num_dn_groups=0,                    # DN(denoising) 查询组数；0 = 不使用 DN
                num_temp_dn_groups=0,               # 时序 DN 的组数；0 = 不使用
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,  # DN 加噪幅度：位置 3 维(X,Y,Z) 2.0，其余 7 维 0.5
                max_dn_gt=32,                       # 每帧最多为多少个 GT 生成 DN 样本
                add_neg_dn=True,                    # 是否额外生成「无目标」的负 DN 样本
                cls_weight=2.0,                     # 匹配代价里分类项的权重
                box_weight=0.25,                    # 匹配代价里回归项的权重
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,  # 匹配代价里 3D 框各分量的权重，长度 10
                                                                #   X,Y,Z = 2.0 | W,L,H = 0.5 | SIN_YAW,COS_YAW,VX,VY = 0.0(不参与)
                cls_wise_reg_weights={              # 按类别覆盖上面的分量权重
                    class_names.index("traffic_cone"): [  # 交通锥(目标小、易漏)单独给一套权重
                        2.0,                        # X
                        2.0,                        # Y
                        2.0,                        # Z
                        1.0,                        # W
                        1.0,                        # L
                        1.0,                        # H
                        0.0,                        # SIN_YAW
                        0.0,                        # COS_YAW
                        1.0,                        # VX
                        1.0,                        # VY
                    ],
                },
            ),
            loss_cls=dict(                          # 分类损失
                type="FocalLoss",                   # Focal Loss，缓解正负样本不平衡
                use_sigmoid=True,                   # 用 sigmoid 做二分类(每个类别独立)
                gamma=2.0,                          # 聚焦参数，越大越关注难样本
                alpha=0.25,                         # 正样本的权重系数
                loss_weight=2.0,                    # 该损失在总损失里的权重
            ),
            loss_reg=dict(                          # 回归损失
                type="SparseBox3DLoss",             # 3D 框回归损失组合
                loss_box=dict(type="L1Loss", loss_weight=0.25),  # 框参数 L1 损失及权重
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),  # 中心度分类损失
                loss_yawness=dict(type="GaussianFocalLoss"),       # 朝向可靠性分类损失
                cls_allow_reverse=[class_names.index("barrier")],  # 这些类别允许朝向翻转 180°(barrier 无正反)
            ),
            decoder=dict(type="SparseBox3DDecoder"),  # 解码器：把网络输出还原成带尺度的 3D 框(反归一化/exp)
            reg_weights=[2.0] * 3 + [1.0] * 7,        # 回归损失里各分量的权重，长度 10
                                                      #   X,Y,Z = 2.0 | W,L,H,SIN_YAW,COS_YAW,VX,VY = 1.0
        ),
        map_head=dict(                           # ---------- 在线建图头 ----------
            type="Sparse4DHead",                 # 与检测头共用同一套稀疏 query 框架
            cls_threshold_to_reg=0.05,           # 分类分数超过该阈值的 query 才参与回归
            decouple_attn=decouple_attn_map,     # 是否解耦注意力
            instance_bank=dict(                  # 地图 query 的实例库
                type="InstanceBank",             # 实例库实现
                num_anchor=100,                  # 每帧 100 个地图 query(每条折线一个 query)
                embed_dims=embed_dims,           # 实例特征通道数
                anchor="data/kmeans/kmeans_map_100.npy",  # K-means 生成的地图 query 初始锚点(kmeans_map.py 产出)
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),  # 折线锚点->采样点
                num_temp_instances=0 if temporal_map else -1,  # 保留的历史实例数
                                                               #   注意 0 = 关闭时序(cache 直接 return)，stage2 才改成 33
                confidence_decay=0.6,            # 历史实例置信度衰减系数
                feat_grad=True,                  # 缓存特征是否参与反向传播(建图这里为 True，与检测相反)
            ),
            anchor_encoder=dict(                 # 折线锚点编码器
                type="SparsePoint3DEncoder",     # 针对「一条折线的多个点」的编码器
                embed_dims=embed_dims,           # 输出通道数
                num_sample=num_sample,           # 每条折线采样点数 = 20
            ),
            num_single_frame_decoder=num_single_frame_decoder_map,  # 前 N 层不做时序
            operation_order=(                    # decoder 各层的算子顺序(含义同检测头)
                [
                    "gnn",         # 实例间自注意力(100 个地图 query 内部交互)
                    "norm",        # 层归一化
                    "deformable",  # 可变形特征聚合
                    "ffn",         # 前馈网络
                    "norm",        # 层归一化
                    "refine",      # 折线回归
                ]
                * num_single_frame_decoder_map   # 单帧层
                + [
                    "temp_gnn",    # 时序注意力(当前 stage1 实际未生效，见 num_temp_instances=0)
                    "gnn",         # 实例间自注意力
                    "norm",        # 层归一化
                    "deformable",  # 可变形特征聚合
                    "ffn",         # 前馈网络
                    "norm",        # 层归一化
                    "refine",      # 折线回归
                ]
                * (num_decoder - num_single_frame_decoder_map)  # 时序层
            )[:],  # 这里没有做切片，保留全部算子(检测头那边是 [2:])
            temp_graph_model=dict(                       # 时序注意力(对应 op "temp_gnn")
                type="MultiheadFlashAttention",          # 注意力实现，无 flash-attn 时自动回退 PyTorch 原生
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,  # 输入通道
                num_heads=num_groups,                    # 注意力头数
                batch_first=True,                        # (batch, seq, dim)
                dropout=drop_out,                        # 注意力 dropout
            )
            if temporal_map                              # 只有开启建图时序才构建
            else None,
            graph_model=dict(                            # 帧内注意力(对应 op "gnn")
                type="MultiheadFlashAttention",          # 同上
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,  # 输入通道
                num_heads=num_groups,                    # 注意力头数
                batch_first=True,                        # (batch, seq, dim)
                dropout=drop_out,                        # 注意力 dropout
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),  # LayerNorm
            ffn=dict(                                    # 前馈网络(对应 op "ffn")
                type="AsymmetricFFN",                    # 输入输出通道可不同
                in_channels=embed_dims * 2,              # 输入通道 = 512(residual_mode="cat" 后通道翻倍)
                pre_norm=dict(type="LN"),                # 先归一化再进全连接
                embed_dims=embed_dims,                   # 输出通道 = 256
                feedforward_channels=embed_dims * 4,     # 隐层通道 = 1024
                num_fcs=2,                               # 全连接层数
                ffn_drop=drop_out,                       # dropout 概率
                act_cfg=dict(type="ReLU", inplace=True), # 激活函数
            ),
            deformable_model=dict(                  # 可变形特征聚合(建图版)
                type="DeformableFeatureAggregation",  # 与检测头同一个实现
                embed_dims=embed_dims,              # query 通道数
                num_groups=num_groups,              # 通道分组数
                num_levels=num_levels,              # 使用几层 FPN 特征
                num_cams=6,                         # 相机数量
                attn_drop=0.15,                     # 采样权重 dropout
                use_deformable_func=use_deformable_func,  # 是否用 CUDA 算子
                use_camera_embed=True,              # 是否加相机参数编码
                residual_mode="cat",                # "cat" 拼接输出与输入(通道翻倍)
                kps_generator=dict(                 # 关键点生成器(针对折线)
                    type="SparsePoint3DKeyPointsGenerator",  # 以「一条折线的多个点」为基准生成关键点
                    embed_dims=embed_dims,          # 输入特征通道数
                    num_sample=num_sample,          # 每条折线的采样点数 = 20
                    num_learnable_pts=3,            # 额外由网络预测 3 个可学习高度偏移
                    fix_height=(0, 0.5, -0.5, 1, -1),  # 固定的采样高度(米，相对地面)，共 5 个
                    ground_height=-1.84023, # ground height in lidar frame  雷达坐标系下的地面高度(米)
                ),                                  # 每个采样点生成 5+3=8 个关键点 -> 20*8=160 个采样位置
            ),
            refine_layer=dict(                       # 折线回归层
                type="SparsePoint3DRefinementModule",  # 输出每条折线的 20 个点
                embed_dims=embed_dims,               # 输入通道数
                num_sample=num_sample,               # 输出点数 = 20
                num_cls=num_map_classes,             # 分类输出维度 = 3
            ),
            sampler=dict(                            # 地图目标分配器
                type="SparsePoint3DTarget",          # 针对折线的目标生成
                assigner=dict(                       # 匹配器：把 GT 折线分配给 query
                    type='HungarianLinesAssigner',   # 匈牙利匹配(折线版)
                    cost=dict(                       # 匹配代价
                        type='MapQueriesCost',       # 地图任务的代价函数
                        cls_cost=dict(type='FocalLossCost', weight=1.0),              # 分类代价及权重
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                                                                                      # 折线点 L1 代价：权重 10.0
                                                                                      # beta 是平滑 L1 的转折点
                                                                                      # permute=True 允许折线首尾反向匹配
                    ),
                ),
                num_cls=num_map_classes,             # 类别数 = 3
                num_sample=num_sample,               # 每条折线点数 = 20
                roi_size=roi_size,                   # BEV 兴趣区域，需与 GT 生成时一致
            ),
            loss_cls=dict(                           # 地图分类损失
                type="FocalLoss",                    # Focal Loss
                use_sigmoid=True,                    # sigmoid 二分类
                gamma=2.0,                           # 聚焦参数
                alpha=0.25,                          # 正样本权重
                loss_weight=1.0,                     # 该损失在总损失里的权重
            ),
            loss_reg=dict(                           # 折线回归损失
                type="SparseLineLoss",               # 折线回归损失组合
                loss_line=dict(
                    type='LinesL1Loss',              # 折线点 L1 损失
                    loss_weight=10.0,                # 权重
                    beta=0.01,                       # 平滑 L1 的转折点
                ),
                num_sample=num_sample,               # 每条折线点数
                roi_size=roi_size,                   # BEV 兴趣区域
            ),
            decoder=dict(type="SparsePoint3DDecoder"),  # 折线解码器
            reg_weights=[1.0] * 40,                  # 回归损失的逐维权重；40 = 20 个点 × 2 维(x,y)
            gt_cls_key="gt_map_labels",              # 从数据里取分类 GT 的键名
            gt_reg_key="gt_map_pts",                 # 从数据里取折线点 GT 的键名
            gt_id_key="map_instance_id",             # 取实例 id 的键名(时序跟踪用)
            with_instance_id=False,                  # 是否启用实例 id 监督
            task_prefix='map',                       # 任务前缀，用于给损失命名(如 loss_map_cls)
        ),
        motion_plan_head=dict(                   # ---------- 运动预测 + 规划头 ----------
            type='MotionPlanningHead',           # 实现类；stage1 的 task_config 里未开启，不会被构建
            fut_ts=fut_ts,                       # 运动预测的未来帧数 = 12 (6s)
            fut_mode=fut_mode,                   # 运动预测的轨迹模式数 = 6
            ego_fut_ts=ego_fut_ts,               # 规划的未来帧数 = 6 (3s)
            ego_fut_mode=ego_fut_mode,           # 规划的轨迹模式数 = 6
            motion_anchor=f'data/kmeans/kmeans_motion_{fut_mode}.npy',  # 他车轨迹的 K-means 初始锚点
            plan_anchor=f'data/kmeans/kmeans_plan_{ego_fut_mode}.npy',  # 自车规划的 K-means 初始锚点
            embed_dims=embed_dims,               # 特征通道数
            decouple_attn=decouple_attn_motion,  # 是否解耦注意力
            instance_queue=dict(                 # 实例队列：保存更长的历史(比 InstanceBank 更长)
                type="InstanceQueue",            # 队列实现
                embed_dims=embed_dims,           # 特征通道数
                queue_length=queue_length,       # 队列长度 = 4 (历史 3 帧 + 当前帧)
                tracking_threshold=0.2,          # 置信度阈值，低于该值的历史实例不参与关联
                feature_map_scale=(input_shape[1]/strides[-1], input_shape[0]/strides[-1]),
                                                 # 取最后一层 FPN 特征图的尺寸 (H, W)，用于编码自车特征
            ),
            operation_order=(        # motion/plan decoder 的算子顺序：3 个完整 block + 1 个 refine
                [
                    "temp_gnn",      # 时序注意力(与实例队列里的历史实例交互)
                    "gnn",           # 实例间自注意力
                    "norm",          # 层归一化
                    "cross_gnn",     # 交叉注意力(与检测/地图的所有实例交互)
                    "norm",          # 层归一化
                    "ffn",           # 前馈网络
                    "norm",          # 层归一化
                ] * 3 +              # 上面的 block 重复 3 次
                [
                    "refine",        # 最后一次轨迹回归
                ]
            ),
            temp_graph_model=dict(                       # 时序注意力(对应 op "temp_gnn")
                type="MultiheadAttention",               # mmcv 标准多头注意力
                                                         #   注意：只有这个模块会收到 key_padding_mask(temp_mask)
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,  # 输入通道
                num_heads=num_groups,                    # 注意力头数
                batch_first=True,                        # (batch, seq, dim)
                dropout=drop_out,                        # 注意力 dropout
            ),
            graph_model=dict(                            # 帧内注意力(对应 op "gnn")
                type="MultiheadFlashAttention",          # 稀疏注意力实现，无 flash-attn 时自动回退 PyTorch
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,  # 输入通道
                num_heads=num_groups,                    # 注意力头数
                batch_first=True,                        # (batch, seq, dim)
                dropout=drop_out,                        # 注意力 dropout
            ),
            cross_graph_model=dict(                      # 交叉注意力(对应 op "cross_gnn")
                type="MultiheadFlashAttention",          # 同上
                embed_dims=embed_dims,                   # 输入通道(交叉注意力不做解耦，所以是 256)
                num_heads=num_groups,                    # 注意力头数
                batch_first=True,                        # (batch, seq, dim)
                dropout=drop_out,                        # 注意力 dropout
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),  # LayerNorm
            ffn=dict(                                    # 前馈网络(对应 op "ffn")
                type="AsymmetricFFN",                    # 输入输出通道可不同
                in_channels=embed_dims,                  # 输入通道 = 256
                pre_norm=dict(type="LN"),                # 先归一化再进全连接
                embed_dims=embed_dims,                   # 输出通道 = 256
                feedforward_channels=embed_dims * 2,     # 隐层通道 = 512(比检测头小，省显存)
                num_fcs=2,                               # 全连接层数
                ffn_drop=drop_out,                       # dropout 概率
                act_cfg=dict(type="ReLU", inplace=True), # 激活函数
            ),
            refine_layer=dict(                       # 轨迹回归层(对应 op "refine")
                type="MotionPlanningRefinementModule",  # 同时输出他车轨迹与自车规划
                embed_dims=embed_dims,               # 输入通道数
                fut_ts=fut_ts,                       # 他车未来帧数 = 12
                fut_mode=fut_mode,                   # 他车轨迹模式数 = 6
                ego_fut_ts=ego_fut_ts,               # 自车未来帧数 = 6
                ego_fut_mode=ego_fut_mode,           # 自车轨迹模式数 = 6
            ),
            motion_sampler=dict(                     # 运动预测的目标生成器
                type="MotionTarget",                 # 把 GT 未来轨迹分配给对应的 query
            ),
            motion_loss_cls=dict(                    # 运动预测的分类损失(选哪条模式)
                type='FocalLoss',                    # Focal Loss
                use_sigmoid=True,                    # sigmoid 二分类
                gamma=2.0,                           # 聚焦参数
                alpha=0.25,                          # 正样本权重
                loss_weight=0.2                      # 在总损失里的权重
            ),
            motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),  # 运动轨迹回归损失及权重
            planning_sampler=dict(                   # 规划的目标生成器
                type="PlanningTarget",               # 生成自车 GT 轨迹 + 驾驶指令
                ego_fut_ts=ego_fut_ts,               # 自车未来帧数
                ego_fut_mode=ego_fut_mode,           # 自车轨迹模式数
            ),
            plan_loss_cls=dict(                      # 规划的分类损失
                type='FocalLoss',                    # Focal Loss
                use_sigmoid=True,                    # sigmoid 二分类
                gamma=2.0,                           # 聚焦参数
                alpha=0.25,                          # 正样本权重
                loss_weight=0.5,                     # 权重
            ),
            plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),     # 规划轨迹回归损失及权重
            plan_loss_status=dict(type='L1Loss', loss_weight=1.0),  # 自车运动状态(速度/加速度)损失及权重
            motion_decoder=dict(type="SparseBox3DMotionDecoder"),   # 运动轨迹解码器
            planning_decoder=dict(                   # 规划解码器
                type="HierarchicalPlanningDecoder",  # 分层规划选择 + 碰撞感知重打分
                ego_fut_ts=ego_fut_ts,               # 自车未来帧数
                ego_fut_mode=ego_fut_mode,           # 自车轨迹模式数
                use_rescore=True,                    # 是否启用重打分
            ),
            num_det=50,                              # 从检测结果里取 top-50 个目标送入运动预测
            num_map=10,                              # 从建图结果里取 top-10 条车道线做交叉注意力
        ),
    ),
)

# ================== data ========================
dataset_type = "NuScenes3DDataset"   # 数据集类名(projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset.py)
data_root = "data/nuscenes/"         # nuScenes 原始数据根目录(samples/ sweeps/ can_bus/ maps/ 等)
anno_root = "data/infos/" if version == 'trainval' else "data/infos/mini/"  # infos pkl 所在目录
                                                                             # 注意 mini 版本落在 data/infos/mini/
file_client_args = dict(backend="disk")  # 文件读取后端：本地磁盘(换成 petrel 等可读远程)

img_norm_cfg = dict(                 # 图像归一化配置(ImageNet 统计量)
    mean=[123.675, 116.28, 103.53],  # 每通道均值(0~255 尺度)
    std=[58.395, 57.12, 57.375],     # 每通道标准差
    to_rgb=True                      # 是否把 BGR 转成 RGB
)
train_pipeline = [                  # 训练数据流水线，按顺序执行
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),  # 读取 6 路相机图像并转 float32
    dict(
        type="LoadPointsFromFile",   # 读取当前帧的激光雷达点云(仅用于生成深度 GT，不作模型输入)
        coord_type="LIDAR",          # 点云坐标系
        load_dim=5,                  # 每点原始维度
        use_dim=5,                   # 实际使用的维度
        file_client_args=file_client_args,  # 文件后端
    ),
    dict(type="ResizeCropFlipImage"),        # 图像缩放/裁剪/翻转 + 三维旋转增强，同时更新投影矩阵
    dict(
        type="MultiScaleDepthMapGenerator",  # 生成多尺度深度 GT(用点云投影到图像)
        downsample=strides[:num_depth_layers],  # 使用前 3 个降采样倍率 [4, 8, 16]
    ),
    dict(type="BBoxRotation"),                      # 3D 框全局旋转增强(与图像增强保持一致)
    dict(type="PhotoMetricDistortionMultiViewImage"),  # 光度畸变增强(亮度/对比度/饱和度/色调)
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),  # 图像归一化
    dict(
        type="CircleObjectRangeFilter",  # 过滤掉距离自车过远的 GT 框
        class_dist_thred=[55] * len(class_names),  # 10 个类别各自的距离阈值，都是 55 米
    ),
    dict(type="InstanceNameFilter", classes=class_names),  # 过滤不属于这 10 类的 GT 框
    dict(
        type='VectorizeMap',          # 把地图折线 GT 转成固定点数的向量形式
        roi_size=roi_size,            # BEV 兴趣区域
        simplify=False,               # 是否对折线做简化(训练时保留原始点数)
        normalize=False,              # 是否归一化坐标
        sample_num=num_sample,        # 每条折线重采样成 20 个点
        permute=True,                 # 是否随机交换点的顺序(增强顺序不变性)
    ),
    dict(type="NuScenesSparse4DAdaptor"),  # 把 GT 转成模型需要的格式(类别 id、3D 框 tensor、时序元信息)
    dict(
        type="Collect",               # 收集流水线产物，其余中间结果丢弃
        keys=[                        # 需要保留的数据键
            "img",                    # 6 路图像
            "timestamp",              # 时间戳(时序对齐用)
            "projection_mat",         # 雷达->图像 的投影矩阵
            "image_wh",               # 图像宽高(归一化投影坐标用)
            "gt_depth",               # 多尺度深度 GT
            "focal",                  # 相机焦距(深度监督用)
            "gt_bboxes_3d",           # 3D 检测框 GT
            "gt_labels_3d",           # 检测类别 GT
            'gt_map_labels',          # 地图折线类别 GT
            'gt_map_pts',             # 地图折线点 GT
            'gt_agent_fut_trajs',     # 他车未来轨迹 GT
            'gt_agent_fut_masks',     # 他车未来轨迹有效性
            'gt_ego_fut_trajs',       # 自车未来轨迹 GT
            'gt_ego_fut_masks',       # 自车未来轨迹有效性
            'gt_ego_fut_cmd',         # 驾驶指令 GT(直行/左转/右转 one-hot)
            'ego_status',             # 自车状态(加速度/角速度/速度/转角)
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
                                      # 元信息：自车到世界的位姿及其逆、时间戳、实例 id
                                      #   前三者正是跨帧时序对齐的依据
    ),
]
test_pipeline = [                   # 推理/测试流水线，只做必要的预处理，没有数据增强
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),  # 读取 6 路图像
    dict(type="ResizeCropFlipImage"),       # 固定方式缩放+裁剪(test_mode 下不随机)
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),  # 图像归一化
    dict(type="NuScenesSparse4DAdaptor"),   # 转成模型输入格式
    dict(
        type="Collect",                     # 收集产物
        keys=[
            "img",                          # 6 路图像
            "timestamp",                    # 时间戳(时序对齐)
            "projection_mat",               # 投影矩阵
            "image_wh",                     # 图像宽高
            'ego_status',                   # 自车状态
            'gt_ego_fut_cmd',               # 驾驶指令(规划用)
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],  # 时序对齐所需元信息
    ),
]
eval_pipeline = [                   # 评测流水线：与 test_pipeline 组合使用，提供 GT 供评测
    dict(
        type="CircleObjectRangeFilter", # 与训练一致的 GT 距离过滤
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),  # 与训练一致的类别过滤
    dict(
        type='VectorizeMap',            # 地图折线转向量
        roi_size=roi_size,              # BEV 兴趣区域
        simplify=True,                  # 评测时做折线简化
        normalize=False,                # 不做归一化
    ),
    dict(
        type='Collect',                 # 收集 GT
        keys=[
            'vectors',                  # 地图折线 GT
            "gt_bboxes_3d",             # 3D 检测框 GT
            "gt_labels_3d",             # 检测类别 GT
            'gt_agent_fut_trajs',       # 他车未来轨迹 GT
            'gt_agent_fut_masks',       # 他车未来轨迹有效性
            'gt_ego_fut_trajs',         # 自车未来轨迹 GT
            'gt_ego_fut_masks',         # 自车未来轨迹有效性
            'gt_ego_fut_cmd',           # 驾驶指令 GT
            'fut_boxes'                 # 未来帧的 GT 框(运动预测评测用)
        ],
        meta_keys=['token', 'timestamp']  # 评测需要 token 来对齐结果
    ),
]

input_modality = dict(      # 输入模态声明
    use_lidar=False,        # 不使用激光雷达作为模型输入
    use_camera=True,        # 使用相机(相机-only 方案)
    use_radar=False,        # 不使用毫米波雷达
    use_map=False,          # 不使用先验地图(地图是预测目标，不是输入)
    use_external=False,     # 不使用外部传感器
)

data_basic_config = dict(   # 数据集公共配置，train/val/test 共用
    type=dataset_type,      # 数据集类
    data_root=data_root,    # 数据根目录
    classes=class_names,    # 检测类别
    map_classes=map_class_names,  # 地图类别
    modality=input_modality,      # 输入模态
    version="v1.0-trainval",      # nuScenes 版本；实际会被 infos pkl 里的 metadata.version 覆盖
)
eval_config = dict(         # 评测用的数据集配置(用于生成 GT)
    **data_basic_config,    # 继承上面的公共配置
    ann_file=anno_root + 'nuscenes_infos_val.pkl',  # 验证集 infos
    pipeline=eval_pipeline, # 评测流水线
    test_mode=True,         # 测试模式(关闭随机增强)
)
data_aug_conf = {           # 图像数据增强配置
    "resize_lim": (0.40, 0.47),      # 随机缩放的倍率范围(相对原图 1600x900)
    "final_dim": input_shape[::-1],  # 最终裁剪尺寸 (高, 宽) = (256, 704)
    "bot_pct_lim": (0.0, 0.0),       # 底部裁剪比例范围(0 表示不裁)
    "rot_lim": (-5.4, 5.4),          # 2D 图像旋转角度范围(度)
    "H": 900,                        # 原始图像高度
    "W": 1600,                       # 原始图像宽度
    "rand_flip": True,               # 是否随机水平翻转
    "rot3d_range": [0, 0],           # 3D 框额外旋转范围(0 表示不额外旋转)
}

data = dict(
    samples_per_gpu=batch_size,   # 每张卡每步的样本数 = 8
    workers_per_gpu=batch_size,   # 每张卡的取数进程数 = 8
    train=dict(
        **data_basic_config,      # 继承公共配置
        ann_file=anno_root + "nuscenes_infos_train.pkl",  # 训练集 infos
        pipeline=train_pipeline,  # 训练流水线(带增强)
        test_mode=False,          # 非测试模式
        data_aug_conf=data_aug_conf,  # 增强配置
        with_seq_flag=True,       # 开启序列分组标记(时序训练必需)
        sequences_split_num=2,    # 每个连续序列再切成 2 段(增加可打乱的组数)
        keep_consistent_seq_aug=True,  # 同一序列内共用同一套增强参数(保证时序一致)
    ),
    val=dict(
        **data_basic_config,      # 继承公共配置
        ann_file=anno_root + "nuscenes_infos_val.pkl",  # 验证集 infos
        pipeline=test_pipeline,   # 测试流水线
        data_aug_conf=data_aug_conf,  # 增强配置(test_mode 下只做固定的缩放裁剪)
        test_mode=True,           # 测试模式
        eval_config=eval_config,  # 评测配置
    ),
    test=dict(
        **data_basic_config,      # 继承公共配置
        ann_file=anno_root + "nuscenes_infos_val.pkl",  # 测试也用验证集(nuScenes 无公开 test 标注)
        pipeline=test_pipeline,   # 测试流水线
        data_aug_conf=data_aug_conf,  # 增强配置
        test_mode=True,           # 测试模式
        eval_config=eval_config,  # 评测配置
    ),
)

# ================== training ========================
optimizer = dict(                    # 优化器
    type="AdamW",                    # AdamW(带权重衰减解耦的 Adam)
    lr=4e-4,                         # 基础学习率
    weight_decay=0.001,              # 权重衰减系数
    paramwise_cfg=dict(              # 分组学习率设置
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),  # 骨干网络学习率乘 0.5(实际 2e-4)，避免破坏预训练特征
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))  # 梯度裁剪：L2 范数上限 25
lr_config = dict(                    # 学习率调度
    policy="CosineAnnealing",        # 余弦退火(迭代级)
    warmup="linear",                 # 前 N 次迭代线性预热
    warmup_iters=500,                # 预热迭代数
    warmup_ratio=1.0 / 3,            # 预热起始学习率 = 基础 lr * 1/3
    min_lr_ratio=1e-3,               # 最终学习率下限 = 基础 lr * 1e-3
)
runner = dict(                       # 训练器
    type="IterBasedRunner",          # 按迭代数计(而不是按 epoch)
    max_iters=num_iters_per_epoch * num_epochs,  # 总迭代数 = 439 * 100 = 43900
)

# ================== eval ========================
eval_mode = dict(                    # 评测哪些任务、用哪些阈值
    with_det=True,                   # 评测 3D 检测(NDS / mAP)
    with_tracking=True,              # 评测多目标跟踪(AMOTA 等；需要结果里带 instance_ids)
    with_map=True,                   # 评测在线建图(mAP)
    with_motion=False,               # 不评测运动预测(stage1 未训练)
    with_planning=False,             # 不评测规划(stage1 未训练)
    tracking_threshold=0.2,          # 跟踪评测时的置信度阈值
    motion_threshhold=0.2,           # 运动预测评测时的置信度阈值(用到时才生效)
)
evaluation = dict(                   # 训练过程中的评测钩子
    interval=num_iters_per_epoch*checkpoint_epoch_interval,  # 每 439*20 次迭代评测一次
    eval_mode=eval_mode,             # 把上面的评测模式传给 dataset.evaluate
)