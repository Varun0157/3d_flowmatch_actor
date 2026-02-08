exp=peract2
tasks=(
  bimanual_push_box
)

# Testing arguments
checkpoint=train_logs/Peract2/denoise3d-Peract2_3dfront_3dwrist-C120-B16-lr1e-4-constant-H3-rectified_flow-pushbox/best.pth
checkpoint_alias=denoise3d-Peract2_3dfront_3dwrist-C120-B16-lr1e-4-constant-H3-rectified_flow-pushbox

max_tries=2
max_steps=25
headless=false
collision_checking=false
seed=0

# Trajectory saving (comment out to disable)
trajectory_save_dir=eval_logs/$exp/$checkpoint_alias/seed$seed/trajectories

# Dataset arguments
data_dir=bimanual_push_box/peract2_test/
dataset=Peract2_3dfront_3dwrist
image_size=256,256

# Model arguments
model_type=denoise3d
bimanual=true
prediction_len=1

action_space=joint  # 'eef' or 'joint'

backbone=clip
fps_subsampling_factor=4

embedding_dim=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

num_ckpts=${#tasks[@]}
for ((i = 0; i < $num_ckpts; i++)); do
  python online_evaluation_rlbench/evaluate_policy.py \
    --checkpoint $checkpoint \
    --task ${tasks[$i]} \
    --max_tries $max_tries \
    --max_steps $max_steps \
    --headless $headless \
    --collision_checking $collision_checking \
    --seed $seed \
    --data_dir $data_dir \
    --dataset $dataset \
    --image_size $image_size \
    --output_file eval_logs/$exp/$checkpoint_alias/seed$seed/${tasks[$i]}/eval.json \
    --model_type $model_type \
    --bimanual $bimanual \
    --prediction_len $prediction_len \
    --backbone $backbone \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $embedding_dim \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --action_space $action_space \
    ${trajectory_save_dir:+--trajectory_save_dir $trajectory_save_dir}
done

python online_evaluation_rlbench/collect_results.py \
  --folder eval_logs/$exp/$checkpoint_alias/seed$seed/
