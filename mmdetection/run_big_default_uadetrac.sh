export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
GPU_NUM=8

CONFIG="./configs/faster_rcnn/faster_rcnn_wide_conformer_small_patch16_fpn_1x_uadetrac.py"
WORK_DIR='./work_dir/faster_rcnn_wide_conformer_small_patch16_lr_1e_4_fpn_1x_uadetrac_1344_800'

# Train
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50040 --use_env ./tools/train.py ${CONFIG} --work-dir ${WORK_DIR} --gpus ${GPU_NUM}  --launcher pytorch --cfg-options model.pretrained='./pretrain_models/WideConformer_small_patch16.pth' model.backbone.patch_size=16

# Test on multiple cards
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50040 --use_env ./tools/test.py ${CONFIG} ${WORK_DIR}/latest.pth --launcher pytorch  --eval bbox

# Test on single card
#./tools/test.py ${CONFIG} ${WORK_DIR}/latest.pth --eval bbox