CUDA_VISIBLE_DEVICES=1 python ${workspaceFolder}/fastchat/serve/cli.py --model-path "lmsys/vicuna-7b-v1.3"

CUDA_VISIBLE_DEVICES=5 python fastchat/serve/cli.py --model-path "output_vicuna_2023_7_4"

CUDA_VISIBLE_DEVICES=5 python fastchat/serve/cli.py --model-path "lmsys/vicuna-7b-v1.3"

CUDA_VISIBLE_DEVICES=5 python fastchat/serve/cli.py --model-path "./output_falcon_lima_10_epochs_2"