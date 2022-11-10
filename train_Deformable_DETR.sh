set -x

export CUDA_VISIBLE_DEVICES=0,1
#export TORCH_DISTRIBUTED_DETAIL=DEBUG
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_file cityperson --batch_size 1 \
--enc_layers 6 --dec_layers 6 --dim_feedforward 1024 --num_feature_levels 4 --model_name deformable_detr --output_dir ../../data/single_obj_car_correct_8842 --resume '../Deformable-DETR/checkpoint/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth' --with_box_refine --eval
# python main.py --dataset_file cityperson --cityperson_path ../../../dataset/cityscapes/ --batch_size 1 \
# --enc_layers 3 --dec_layers 3 --output_dir ./output 
# origin checkpoint ../Deformable-DETR/exps/r50_deformable_detr_plus_iterative_bbox_refinement_torch19_cityperson/checkpoint.pth
# ../Deformable-DETR/checkpoint/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
