# #!/bin/bash
# # Used for training TSA-AP model
# set -e

# ### Anomaly Types: platform, trend, extremum, mean, amplitude, frequency, shift
# ### Only platform and mean support histogram distribution now
# anomaly_type_1="platform"
# anomaly_type_2="mean"
# anomaly_num="1"
# gpu=0
# time_id=`date +"%Y%m%d_%H%M%S"`
# clean_df_path="data/processed/cmu_motion.pkl"

# train_data_name_1="mot_${anomaly_type_1}_${anomaly_num}_scaled"
# python src/anom_generator/anomalies.py \
#     --data_path $clean_df_path \
#     --anom_type $anomaly_type_1 \
#     --truncated_length 512 \
#     --aug_num 1 \
#     --name $train_data_name_1 \
#     --seed 0

# train_data_name_2="mot_${anomaly_type_2}_${anomaly_num}_scaled"
# python src/anom_generator/anomalies.py \
#     --data_path $clean_df_path \
#     --anom_type $anomaly_type_2 \
#     --truncated_length 512 \
#     --aug_num 1 \
#     --name $train_data_name_2 \
#     --seed 1

# ### Merge data of two anomalies
# anomaly_type="mixed"
# train_data_name="mot_mix_${anomaly_num}_hist"
# train_ratio=0.8
# val_ratio=0.1
# anomaly_ratio=0.1
# python src/utils/merge_data.py \
#     --data_path_1 $train_data_name_1 \
#     --data_path_2 $train_data_name_2 \
#     --store_path $train_data_name \
#     --train_ratio $train_ratio \
#     --val_ratio $val_ratio \
#     --anomaly_ratio $anomaly_ratio

# ### Run algorithm
# base_name="a=${anomaly_num}_${anomaly_type}"
# config_path="configs/default.yml"
# monitor="val_loss"
# results_path="results/$base_name"
# CUDA_VISIBLE_DEVICES=$gpu python src/main.py \
#     --time_id "$time_id" \
#     --data_path "data/${train_data_name}" \
#     --ckpt_name "$base_name" \
#     --ckpt_monitor "$monitor" \
#     --config_path "$config_path" \
#     # --wandb

#!/bin/bash
# Used for training TSA-AP model with a single fixed platform anomaly
set -e

### Anomaly Type: platform
anomaly_type_1="platform"
anomaly_num="1"
gpu=0
time_id=`date +"%Y%m%d_%H%M%S"`
clean_df_path="data/processed/cmu_motion.pkl"

# Define fixed parameters for platform anomaly
best_point="{
    'ratio_anomaly': 0.1,  # Total ratio of anomalies (e.g., 10% of the data)
    'fixed_level': 0.5,    # Fixed level for the platform anomaly
    'fixed_length': 0.3,   # Fixed length for the platform anomaly
    'fixed_start': 0.2     # Fixed start position for the platform anomaly
}"

train_data_name_1="mot_${anomaly_type_1}_${anomaly_num}_scaled"
python src/anom_generator/anomalies.py \
    --data_path $clean_df_path \
    --anom_type $anomaly_type_1 \
    --truncated_length 512 \
    --aug_num 1 \
    --name $train_data_name_1 \
    --seed 0 \
    --fixed_parameters "$best_point"

### Merge data of the single anomaly
anomaly_type="single_fixed"
train_data_name="mot_single_${anomaly_num}_hist"
train_ratio=0.8
val_ratio=0.1
anomaly_ratio=0.1
python src/utils/merge_data.py \
    --data_path_1 $train_data_name_1 \
    --store_path $train_data_name \
    --train_ratio $train_ratio \
    --val_ratio $val_ratio \
    --anomaly_ratio $anomaly_ratio

### Run algorithm
base_name="a=${anomaly_num}_${anomaly_type}"
config_path="configs/default.yml"
monitor="val_loss"
results_path="results/$base_name"
CUDA_VISIBLE_DEVICES=$gpu python src/main.py \
    --time_id "$time_id" \
    --data_path "data/${train_data_name}" \
    --ckpt_name "$base_name" \
    --ckpt_monitor "$monitor" \
    --config_path "$config_path"
