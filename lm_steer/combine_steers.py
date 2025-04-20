import torch
import argparse

def combine_steers(model_name):
    # 加载两个checkpoint
    detox_ckpt = torch.load(f'logs/detoxification-{model_name}/checkpoint.pt', weights_only=False)
    sent_ckpt = torch.load(f'logs/sentiment-{model_name}/checkpoint.pt', weights_only=False)

    # 打印调试信息
    print("Detox checkpoint content:", detox_ckpt)
    print("Sentiment checkpoint content:", sent_ckpt)

    # 获取两个控制器的projector矩阵
    if isinstance(detox_ckpt, list):
        detox_proj1 = detox_ckpt[0]
        detox_proj2 = detox_ckpt[1]
    elif isinstance(detox_ckpt, dict):
        detox_proj1 = detox_ckpt['projector1']
        detox_proj2 = detox_ckpt['projector2']
    else:
        raise ValueError(f"Unexpected detox checkpoint type: {type(detox_ckpt)}")

    if isinstance(sent_ckpt, list):
        sent_proj1 = sent_ckpt[0]
        sent_proj2 = sent_ckpt[1]
    elif isinstance(sent_ckpt, dict):
        sent_proj1 = sent_ckpt['projector1']
        sent_proj2 = sent_ckpt['projector2']
    else:
        raise ValueError(f"Unexpected sentiment checkpoint type: {type(sent_ckpt)}")

    # 打印调试信息
    print("Detox projector shapes:", detox_proj1.shape if detox_proj1 is not None else None, 
          detox_proj2.shape if detox_proj2 is not None else None)
    print("Sentiment projector shapes:", sent_proj1.shape if sent_proj1 is not None else None, 
          sent_proj2.shape if sent_proj2 is not None else None)

    # 拼接矩阵
    combined_proj1 = torch.cat([detox_proj1, sent_proj1], dim=0)
    combined_proj2 = torch.cat([detox_proj2, sent_proj2], dim=0)

    # 保存组合后的checkpoint
    combined_ckpt = {
        'projector1': combined_proj1,
        'projector2': combined_proj2,
        'config': {
            'num_steers': 4,  # 2个控制器，每个控制器2个维度
            'rank': detox_ckpt.get('config', {}).get('rank', 1000),  # 添加默认值
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