import torch
import argparse

def combine_steers(model_name):
    # 加载两个checkpoint
    detox_ckpt = torch.load(f'logs/detoxification-{model_name}/checkpoint.pt', weights_only=False)
    sent_ckpt = torch.load(f'logs/sentiment-{model_name}/checkpoint.pt', weights_only=False)

    # 获取两个控制器的projector矩阵
    detox_proj1 = detox_ckpt['projector1']
    detox_proj2 = detox_ckpt['projector2']
    sent_proj1 = sent_ckpt['projector1']
    sent_proj2 = sent_ckpt['projector2']

    # 拼接矩阵
    combined_proj1 = torch.cat([detox_proj1, sent_proj1], dim=0)
    combined_proj2 = torch.cat([detox_proj2, sent_proj2], dim=0)

    # 保存组合后的checkpoint
    combined_ckpt = {
        'projector1': combined_proj1,
        'projector2': combined_proj2,
        'config': {
            'num_steers': 4,  # 2个控制器，每个控制器2个维度
            'rank': detox_ckpt['config']['rank'],
            'adaptor_class': 'multiply'
        }
    }

    # 保存组合后的模型
    torch.save(combined_ckpt, f'logs/combined-{model_name}/checkpoint.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='模型名称，如gpt2-large')
    args = parser.parse_args()
    
    # 创建保存目录
    import os
    os.makedirs(f'logs/combined-{args.model}', exist_ok=True)
    
    combine_steers(args.model)