set -x

export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 --use_env generate_visual_result.py --dataset_file cityperson --batch_size 1 \
--enc_layers 6 --dec_layers 6 --dim_feedforward 1024 --num_feature_levels 4 --model_name deformable_detr --output_dir ../../data/single_obj_car_correct_8842_inference --resume '../../data/single_obj_car_correct_8842_generate_atten/checkpoint.pth' --with_box_refine --eval --visual_generate 2

