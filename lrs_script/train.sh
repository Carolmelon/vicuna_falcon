# 和launch.json里面的vicuna train一致
# 24:16
CUDA_VISIBLE_DEVICES="5,6" ~/miniconda3/envs/vicuna/bin/torchrun \
    --nproc_per_node=2 \
    --master_port=20001 \
    "fastchat/train/train.py" \
    --model_name_or_path "lmsys/vicuna-7b-v1.3" \
    --data_path "data/dummy_conversation.json" \
    --bf16 "True" \
    --output_dir "output_vicuna_slow" \
    --num_train_epochs "3" \
    --per_device_train_batch_size "2" \
    --per_device_eval_batch_size "2" \
    --gradient_accumulation_steps "16" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "1200" \
    --save_total_limit "10" \
    --learning_rate "2e-5" \
    --weight_decay "0." \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --logging_steps "1" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
    --tf32 "True" \
    --model_max_length "2048" \
    --gradient_checkpointing "True" \
    --lazy_preprocess "True"

# 加速版
# 14:33
CUDA_VISIBLE_DEVICES="5,6" /ssd1/lirongsheng01/miniconda3/envs/vicuna/bin/torchrun \
    --nproc_per_node=2 \
    --master_port=20001 \
    "fastchat/train/train_mem.py" \
    --model_name_or_path "lmsys/vicuna-7b-v1.3" \
    --data_path "data/dummy_conversation.json" \
    --bf16 "True" \
    --output_dir "output_vicuna" \
    --num_train_epochs "3" \
    --per_device_train_batch_size "2" \
    --per_device_eval_batch_size "2" \
    --gradient_accumulation_steps "16" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "1200" \
    --save_total_limit "10" \
    --learning_rate "2e-5" \
    --weight_decay "0." \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --logging_steps "1" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
    --tf32 "True" \
    --model_max_length "2048" \
    --gradient_checkpointing "True" \
    --lazy_preprocess "True"

# 换底层模型为falcon，tiiuae/falcon-7b
CUDA_VISIBLE_DEVICES="1,5" /ssd1/lirongsheng01/miniconda3/envs/vicuna/bin/torchrun \
    --nproc_per_node=2 \
    --master_port=20002 \
    "fastchat/train/train_falcon.py" \
    --model_name_or_path "tiiuae/falcon-7b" \
    --data_path "data/sharegpt.json" \
    --bf16 "True" \
    --output_dir "output_falcon" \
    --num_train_epochs "3" \
    --per_device_train_batch_size "2" \
    --per_device_eval_batch_size "2" \
    --gradient_accumulation_steps "16" \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps "1200" \
    --save_total_limit "10" \
    --learning_rate "2e-5" \
    --weight_decay "0." \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --logging_steps "1" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "DecoderLayer" \
    --tf32 "True" \
    --model_max_length "2048" \
    --gradient_checkpointing "True" \
    --lazy_preprocess "True"

# 换底层模型为falcon，tiiuae/falcon-7b，小数据集lima
CUDA_VISIBLE_DEVICES="1,4,5,6" /ssd1/lirongsheng01/miniconda3/envs/vicuna/bin/torchrun \
    --nproc_per_node=4 \
    --master_port=20002 \
    "fastchat/train/train_falcon.py" \
    --model_name_or_path "tiiuae/falcon-7b" \
    --data_path "data/lima.json" \
    --bf16 "True" \
    --output_dir "output_falcon_lima_10_epochs_2" \
    --num_train_epochs "10" \
    --per_device_train_batch_size "2" \
    --per_device_eval_batch_size "2" \
    --gradient_accumulation_steps "16" \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps "1200" \
    --save_total_limit "10" \
    --learning_rate "2e-5" \
    --weight_decay "0." \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --logging_steps "1" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "DecoderLayer" \
    --tf32 "True" \
    --model_max_length "2048" \
    --gradient_checkpointing "True" \
    --lazy_preprocess "True"
