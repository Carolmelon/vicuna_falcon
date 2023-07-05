# 启动控制器
python3 -m fastchat.serve.controller

# 启动模型
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path ./output_falcon_lima_10_epochs_2 --port 20006 --worker http://localhost:20006

# 启动模型
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path ./output_falcon_sex 