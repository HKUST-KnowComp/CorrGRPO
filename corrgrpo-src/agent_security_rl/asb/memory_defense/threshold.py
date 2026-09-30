import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

import re
import os
import json

def process_log_clean(log_file):
    memorys = []
    poison_labels = []
    with open(log_file, 'r', encoding='utf-8') as file:
        log_data = file.read()
    # 1. Split blocks delimited by "**********************************"
    blocks = log_data.split('**********************************')
    for i, block in enumerate(blocks, 1):
        if i == 0 or i%2==1:
            continue

        pattern = re.compile(
            r'''The workflow generated for the problem is.*?(\[\{.*?\}\])''',
            re.DOTALL
        )

        # Find all matches
        matches = pattern.findall(block)
        # print(f"Block {i} - Number of matches: {len(matches)}")
        # Print the extracted JSON containing the workflow
        generated_plan = matches[0]
        print(generated_plan)
        memorys.append(generated_plan)
        poison_labels.append(False)

    return memorys

def parse_plans(plans_json_strs):
    """
    Convert workflow JSON strings into plain text.

    Args:
    - plans_json_strs: List of workflow JSON strings.

    Returns:
    - plans_texts: List of parsed workflow texts.
    """
    plans_texts = []
    for json_str in plans_json_strs:
        # Replace single quotes with double quotes for JSON parsing
        json_str_fixed = json_str.replace("'", '"')
        try:
            plan_list = json.loads(json_str_fixed)
            # Extract each step's 'message' field and concatenate the text
            plan_text = ' '.join([step['message'] for step in plan_list])
            plans_texts.append(plan_text)
        except json.JSONDecodeError as e:
            print(f'Error decoding JSON: {e}')
            continue

    print(f'Total plans converted to text: {len(plans_texts)}')
    return plans_texts

class PerplexityFilter:
    """
    Compute text perplexity using a Hugging Face model.

    Args:
    - model: Loaded language model.
    - tokenizer: Tokenizer associated with the model.
    - threshold: Perplexity threshold.
    - window_size: Sliding-window size; defaults to 'all'.

    Methods:
    - detect(sequence): Check whether the text passes the perplexity filter.
    - get_log_prob(sequence): Compute the negative log-likelihood (NLL) of the text.
    - filter(sequences): Compute NLL for each text and determine whether it passes the filter.
    - filter_window(sequences): Filter text using a sliding window.
    """
    def __init__(self, model, tokenizer, threshold, window_size='all'):
        self.tokenizer = tokenizer
        self.model = model.cuda()  # Ensure that the model runs on the GPU
        self.threshold = threshold
        self.window_threshold = threshold
        self.window_size = window_size
        self.cn_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def detect(self, sequence):
        if self.window_size == 'all':
            return (not self.filter([sequence])[-1][0])
        elif self.window_size <= 0 or type(self.window_size) != int:
            raise ValueError(f"ERROR: window_size={self.window_size}. window_size must be a positive integer.")
        return (not self.filter_window([sequence])[-1][0])

    def get_log_prob(self, sequence):
        input_ids = self.tokenizer.encode(sequence, return_tensors='pt').cuda()
        with torch.no_grad():
            logits = self.model(input_ids, labels=input_ids, use_cache=False).logits
        logits = logits[:, :-1, :].contiguous()
        input_ids = input_ids[:, 1:].contiguous()
        log_probs = self.cn_loss(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        return log_probs

    def filter(self, sequences):
        filtered_log_ppl = []
        passed_filter = []
        for sequence in sequences:
            log_probs = self.get_log_prob(sequence)
            NLL_by_token = log_probs
            if NLL_by_token.mean() <= self.threshold:
                passed_filter.append(True)
                filtered_log_ppl.append(NLL_by_token.mean().item())
            else:
                passed_filter.append(False)
                filtered_log_ppl.append(NLL_by_token.mean().item())
        # print(filtered_log_ppl, passed_filter)
        return filtered_log_ppl, passed_filter

    def filter_window(self, sequences, reverse=False):
        filtered_log_ppl_by_window = []
        passed_filter_by_window = []
        passed = []
        for sequence in sequences:
            sequence_window_scores = []
            passed_window_filter = []
            log_probs = self.get_log_prob(sequence)
            NLL_by_token = log_probs
            for i in np.arange(0, len(NLL_by_token), self.window_size):
                if not reverse:
                    window = NLL_by_token[i:i+self.window_size]
                else:
                    if i == 0:
                        window = NLL_by_token[-self.window_size:]
                    elif -(-i-self.window_size) > len(NLL_by_token) and i != 0:
                        window = NLL_by_token[:-i]
                    else:
                        window = NLL_by_token[-i-self.window_size:-i]
                if window.mean() <= self.window_threshold:
                    passed_window_filter.append(True)
                    sequence_window_scores.append(window.mean().item())
                else:
                    passed_window_filter.append(False)
                    sequence_window_scores.append(window.mean().item())
            if all(passed_window_filter):
                passed.append(True)
            else:
                passed.append(False)
            passed_filter_by_window.append(passed_window_filter)
            filtered_log_ppl_by_window.append(sequence_window_scores)
        return filtered_log_ppl_by_window, passed_filter_by_window, passed

def calculate_perplexities(plans_texts, model, tokenizer, perplexity_threshold, window_size):
    """
    Compute per-text perplexities and their mean for a list of texts.

    Args:
    - plans_texts: List of texts.
    - model: Loaded language model.
    - tokenizer: Tokenizer associated with the model.
    - perplexity_threshold: Perplexity threshold.
    - window_size: Sliding-window size.

    Returns:
    - perplexities: List of per-text perplexities.
    - average_perplexity: Mean perplexity across all texts.
    """
    # Create a PerplexityFilter instance
    filter = PerplexityFilter(model=model, tokenizer=tokenizer, threshold=perplexity_threshold, window_size=window_size)
    perplexities = []

    for plan_text in plans_texts:
        # Compute perplexity
        log_ppl, passed_filter_list = filter.filter([plan_text])
        nll = log_ppl[0]
        # Convert NLL to perplexity
        perplexity = torch.exp(torch.tensor(nll)).item()
        perplexities.append(perplexity)
        # print(f"Perplexity for plan: {perplexity}")

    # Compute mean perplexity
    average_perplexity = sum(perplexities) / len(perplexities) if perplexities else 0.0
    print(f'Average Perplexity over plans: {average_perplexity}')
    return perplexities, average_perplexity

def main():
    # Set the log file path
    log_base = 'logs/clean/gpt-4o-2024-08-06/no_memory/combined_attack-non-aggressive_.log'
    # Extract workflow JSON strings
    plans_json_strs = process_log_clean(log_base)
    # Parse workflows into text
    # plans_texts = parse_plans(plans_json_strs)
    # Load the model and tokenizer
    model_name = 'facebook/opt-2.7b'
    # model_name = 'EleutherAI/gpt-neo-1.3B'

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)

    # Set the perplexity filter threshold and window size
    perplexity_threshold = 2.0  # Adjust for the task as needed
    window_size = 10  # Optional sliding-window size; use 'all' for the full sequence

    # Compute perplexities
    perplexities, average_perplexity = calculate_perplexities(plans_json_strs, model, tokenizer, perplexity_threshold, window_size)
    # Sort the perplexity scores
    ppl_values_sorted = sorted(perplexities)
    print(f"Perplexity values: {ppl_values_sorted}")
    # Compute the 99th percentile
    threshold_ppl = np.percentile(ppl_values_sorted, 99)

    print(f"Perplexity threshold for a 1% FPR: {threshold_ppl}")
    # print(f"Average Perplexity: {average_perplexity}")
if __name__ == "__main__":
    main()
