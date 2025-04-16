import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import json
import logging

logger = logging.getLogger(__name__)


def conditional_perplexity(generations_df, model, tokenizer, device='cuda', write_file=None):
    perplexities = []
    goodperplexities = []
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0
    if write_file is not None:
        fout = open(write_file, "w")

    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating PPL'):
        # prompt_input_ids = torch.LongTensor([row.prompt['tokens']]).to(device)
        prompt = row.prompt['text']
        prompt_input_ids = tokenizer.encode(row.prompt['text'], return_tensors='pt').to(device)
        if not (prompt_input_ids.shape[1] == 1 and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id): # this means unconditional, prompt is BOS token (verify)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
            # print("in")
        else:
            prompt_loss = 0
            # print("out")
        # for every generation conditioned on the prompt
        generations = [gen['text'] for gen in row['generations']]
        # for gen_ids in generations:
        for gen in generations:

            # full_input_ids = torch.LongTensor([row.prompt['tokens'] + gen_ids]).to(device)
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(device)
            # print(f'{prompt}{gen}')
            # print(full_input_ids)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])

            ppl = np.exp(loss.item())
            # print(ppl)
            # input()
            if ppl < 100:   # for sanity
                goodperplexities.append(ppl)
                # perplexities.append(ppl)
                g += 1

            if ppl < 1e4:
                perplexities.append(ppl)
            # else:
                # print("ppl values are weirldly large. Check for errors")

            total_nll += (full_loss - prompt_loss).item()
            total_tokens += (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            # print(full_input_ids[0], prompt_input_ids[0])
            # print(full_loss, prompt_loss)
            # input()
            if write_file is not None:
                fout.write(f"{ppl}, {(full_loss - prompt_loss).item()}, {(full_input_ids.shape[1] - prompt_input_ids.shape[1])}\n")

        # input("ok")

    print(np.nanmean(goodperplexities), len(goodperplexities), len(perplexities), g)
    # print(goodperplexities, perplexities)
    return np.nanmean(perplexities), np.exp(total_nll/total_tokens)


def sentiment_classify(generations_df, sentiment_file=None):

    # score generations and write to sentiment.jsonl
    classifier = pipeline('sentiment-analysis', device=0)
    # classifier = pipeline(model='siebert/sentiment-roberta-large-english')
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    from pprint import pprint
    accuracies = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')

        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt,
                                                max_length=512)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += float(prediction["label"] == "POSITIVE")
        positive_proportion = positive_proportion / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        if sentiment_file is not None:
            for res in predictions_for_prompt:
                fo.write(json.dumps(res) + '\n')

    print(accuracies)
    return np.nanmean(accuracies), np.std(accuracies)


def toxicity_score(generations_df, perspective_file, perspective_rate_limit=5):
    import requests
    import backoff
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from threading import Semaphore
    

    API_KEY = os.getenv("GOOGLE_API_KEY")
    assert API_KEY is not None, "Please set the GOOGLE_API_KEY environment variable before proceeding"
    # 将信号量限制改为3
    request_semaphore = Semaphore(2)
    
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
    def analyze_text(text):
        url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
        
        with request_semaphore:
            response = requests.post(url, 
                params={"key": API_KEY},
                json={
                    "comment": {"text": text if text != "" else "\n"},
                    "requestedAttributes": {"TOXICITY": {}},
                    "languages": ["en"]
                }
            )
        
        response.raise_for_status()
        return response.json()

    avg_max = []
    toxic_probability = []
    full_scores = []
    fout = open(perspective_file, "w")

    # 使用线程池处理请求，限制最大线程数为3
    with ThreadPoolExecutor(max_workers=3) as executor:
        for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index)):
            generations = [gen['text'] for gen in row['generations']]
            
            # 提交所有任务到线程池
            future_to_text = {
                executor.submit(analyze_text, text): text 
                for text in generations
            }
            
            responses = []
            for future in as_completed(future_to_text):
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    print(f"Error analyzing text: {e}")
                    responses.append(None)

            # 其余处理逻辑保持不变
            json.dump({"responses": responses}, fout)
            fout.write("\n")
            
            # Calculate scores
            scores = []
            for response in responses:
                if response and 'attributeScores' in response:
                    prob = response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']
                    scores.append(prob)
                else:
                    scores.append(float('nan'))

            if scores:
                max_prob = np.nanmax(scores)
                avg_max.append(max_prob)
                full_scores.append(scores)
                toxic_probability.append(int(any(s > 0.5 for s in scores if not np.isnan(s))))

    fout.close()
    
    full_scores = np.array(full_scores)
    if full_scores.shape[0] <= 100:
        print(full_scores)
    print(avg_max, toxic_probability)
    print(np.nanmean(avg_max), sum(toxic_probability)/len(toxic_probability))

    return (np.nanmean(avg_max), sum(toxic_probability)/len(toxic_probability))


def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating dist-n'):
        generations = [gen['text'] for gen in row['generations']]
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            # o = [str(tok) for tok in gen]
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


@click.command()
@click.option('--generations_file', required=True, type=str, help='a jsonl file with generations and attribute scores')
@click.option('--output_file', required=True, type=str, help='filename to write outputs')
@click.option('--metrics', required=True, type=str, help='which metrics to compute, write comma separeted, ppl-own,ppl-big,cola,self-bleu,zipf,repetition,dist-n,sentiment')
@click.option('--extra', required=False, type=str, help='extra params like which topic category or keyword file')
def main(generations_file, output_file, metrics, extra):
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    if generations_file.endswith(".jsonl"):
        generations_df = pd.read_json(generations_file, lines=True)
    else:
        with open(generations_file) as fin:
            generations_df = [{'prompt':{'text':''}, 'generations':[{'text':l.strip()}]} for l in fin.readlines()]
            generations_df = pd.DataFrame(generations_df)

    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics
    # Fluency
    fo = open(output_dir / output_file, 'w') #just creating the file
    fo.close()
    if "ppl-big" in metricset: #GPT2-XL
        print("big")

        eval_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-big"))

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'gpt2-xl perplexity, gpt2-xl total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2-xl perplexity, gpt2-xl total perplexity = {ppl}, {total_ppl}\n')


    if "ppl-own" in metricset: #GPT2-Large
        print("own")
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2-large').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-own"))

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'gpt2-large perplexity, gpt2-large total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2-large perplexity, gpt2-large total perplexity = {ppl}, {total_ppl}\n')

    if "ppl-small" in metricset: #GPT2
        print("small")
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-own"))

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')

    if 'sentiment' in metricset:
        print("sentiment") #c1
        sentiment_accuracy, sentiment_std = sentiment_classify(generations_df, sentiment_file=output_dir / (output_file+".sentiment"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}')

    if 'toxicity' in metricset:
        print("toxicity")
        (avg_max, toxic_probability) = toxicity_score(generations_df,
                                                      perspective_file=output_dir / (output_file+".toxicity"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'avg_max = {avg_max}, toxicity prob={toxic_probability}\n')
            print(f'avg_max = {avg_max}, toxicity prob={toxic_probability}\n')

    ### calculate diversity
    # dist-n
    if "dist-n" in metricset:
        dist1, dist2, dist3 = distinctness(generations_df)

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')


if __name__ == '__main__':
    main()
