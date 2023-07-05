python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8483

# 对话模型
curl http://jp02-gpu-a100s06.jp02.baidu.com:8483/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "output_falcon_lima_10_epochs_2",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'

# 色情对话模型
curl http://jp02-gpu-a100s06.jp02.baidu.com:8483/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "output_falcon_sex",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'

# 查询模型
curl http://jp02-gpu-a100s06.jp02.baidu.com:8483/v1/models

# 补全模型
curl http://jp02-gpu-a100s06.jp02.baidu.com:8483/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "output_falcon_lima_10_epochs_2",
    "prompt": "Once upon a time",
    "max_tokens": 41,
    "temperature": 0.5
  }'

