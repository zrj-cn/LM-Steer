# 这个代码可以根据pipeline的输入输出文件，生成文件储存目录，并随机抽取一定数量的prompt，生成新的prompt文件。
import json
import random
import argparse
import os

def sample_prompts(input_file, output_file, num_samples, seed=None):
    # 如果提供了种子，设置随机种子
    if seed is not None:
        random.seed(seed)
    
    # 检查并创建输出目录
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录：{output_dir}")
    
    # 读取所有提示
    with open(input_file, 'r', encoding='utf-8') as f:
        prompts = [line for line in f]
    
    # 随机抽样
    if num_samples > len(prompts):
        print(f"警告：要求的样本数 {num_samples} 大于总数据量 {len(prompts)}，将返回所有数据")
        sampled_prompts = prompts
    else:
        sampled_prompts = random.sample(prompts, num_samples)
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(sampled_prompts):
            if prompt.strip():  # 只写入非空行
                if i < len(sampled_prompts) - 1:  # 如果不是最后一行
                    f.write(prompt)
                else:  # 如果是最后一行，确保不会多出空行
                    f.write(prompt.rstrip('\n'))
    
    print(f"已成功从 {input_file} 随机抽取 {len(sampled_prompts)} 个提示")
    print(f"结果已保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='随机抽取提示数据')
    parser.add_argument('--num_samples', type=int, required=True, help='要抽取的样本数量')
    parser.add_argument('--input_file', type=str, default='./data/prompts/nontoxic_prompts-10k.jsonl', help='输入文件名')
    parser.add_argument('--output_file', type=str, help='输出文件名')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    args = parser.parse_args()
    
    sample_prompts(args.input_file, args.output_file, args.num_samples, args.seed)