# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import re
from collections import Counter


LENGTH_REWARD_BUDGET = 100


def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order.

    Reference: Liu S Y, Dong X, Lu X, et al. "Gdpo: Group reward-decoupled normalization policy
    optimization for multi-reward rl optimization."
    arXiv preprint arXiv:2601.05242, 2026.
    """
    if list1 == list2:
        return 1.0

    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)  # Frequency count for list1
    count2 = Counter(list2)  # Frequency count for list2

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0


# custoimzed reward functions: format
def customize_format_reward_func(
    completions, answer, step, max_possible_reward, min_possible_reward, do_print, **kwargs
):
    rewards = []
    responses = [completion[0]["content"] for completion in completions]

    if do_print:
        print("\n======= Answer ======= ")
        print(answer[0])
        print("\n======= Responses ======= ")
        for idx, response in enumerate(responses):
            print(f"*** Response {idx + 1}***\n{response}")

    for response, ans in zip(responses, answer, strict=False):
        reward = min_possible_reward
        if "<response>" in ans and "<tool_call>" not in ans:
            pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
            if (
                re.search(pattern, response, re.DOTALL)
                and response.count("<response>") == 1
                and response.count("</response>") == 1
            ):
                reward = max_possible_reward
        elif "<response>" not in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$"
            if (
                re.search(pattern, response, re.DOTALL)
                and response.count("<tool_call>") == 1
                and response.count("</tool_call>") == 1
            ):
                reward = max_possible_reward
        elif "<response>" in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
            if (
                re.search(pattern, response, re.DOTALL)
                and response.count("<tool_call>") == 1
                and response.count("</tool_call>") == 1
                and response.count("<response>") == 1
                and response.count("</response>") == 1
            ):
                reward = max_possible_reward
        else:
            pattern = r"^<think>.*?</think>$"
            if re.search(pattern, response, re.DOTALL):
                reward = max_possible_reward

        rewards.append(reward)

    if do_print:
        print("\n======= Reward for <format> =======")
        print("Reward function for <format> is called ...")
        print(rewards)

    return rewards


def compute_length_reward(response_length, budget=LENGTH_REWARD_BUDGET):
    """Return 1 when the response token length is within budget, else 0."""
    if budget < 0:
        raise ValueError(f"Length reward budget must be non-negative, got {budget}.")
    return float(int(response_length) <= budget)


def _scale_tool_call_score(score, local_max_possible, max_possible_reward, min_possible_reward):
    """Scale one tool-call score to the same range as the total reward."""
    if local_max_possible == 0:
        # There is nothing to predict for this component, so it is vacuously correct.
        return max_possible_reward
    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward, do_print):
    """Score a tool call and return its total and component rewards.

    The total reward preserves the original scoring calculation. Its raw score
    is decomposed into function-name matching, parameter-name matching, and
    parameter-value matching. Each component is independently mapped to the
    same ``[min_possible_reward, max_possible_reward]`` range as the total.
    """
    if gt_tools == pd_tools:
        if do_print:
            print("Max possible score:", "Exact Match!")
            print("Score:", max_possible_reward)
        return {
            "total_reward": max_possible_reward,
            "function_name_reward": max_possible_reward,
            "parameter_reward": max_possible_reward,
            "values_reward": max_possible_reward,
        }

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    function_name_score = match_score(list(gt_names), list(pd_names))
    parameter_score = 0.0
    values_score = 0.0
    score = function_name_score

    local_max_possible = 1.0
    parameter_max_possible = 0.0
    values_max_possible = 0.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]

        local_max_possible += 1.0 + len(gt_params)
        parameter_max_possible += 1.0
        values_max_possible += len(gt_params)

        best_match = None
        best_match_score = 0.0
        best_match_index = -1
        best_parameter_score = 0.0
        best_values_score = 0.0

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue

            pd_params = pd_tool["parameters"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))

            # Calculate correctness score for parameter values
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            total_score = param_score + correctness_score

            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i
                best_parameter_score = param_score
                best_values_score = correctness_score

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score
            parameter_score += best_parameter_score
            values_score += best_values_score

    if do_print:
        print()
        print("Max possible score:", local_max_possible)
        print("Score:", score)

    function_name_reward = _scale_tool_call_score(
        function_name_score,
        1.0,
        max_possible_reward=0.5,
        min_possible_reward=-0.5,
    )
    parameter_reward = _scale_tool_call_score(
        parameter_score,
        parameter_max_possible,
        max_possible_reward=1,
        min_possible_reward=-1,
    )
    values_reward = _scale_tool_call_score(
        values_score,
        values_max_possible,
        max_possible_reward=1.5,
        min_possible_reward=-1.5,
    )

    total_reward = function_name_reward + parameter_reward + values_reward
    old_total_reward = (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward
    return {
        "total_reward": total_reward,
        "old_total_reward": old_total_reward,
        "function_name_reward": function_name_reward,
        "parameter_reward": parameter_reward,
        "values_reward": values_reward
    }


# custoimzed reward functions: tool call correctness
def customize_correctness_reward_tool(
    completions, answer, step, max_possible_reward, min_possible_reward, do_print, **kwargs
):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response, ans in zip(responses, answer, strict=False):
        reward = {
            "total_reward": 0.0,
            "function_name_reward": 0.0,
            "parameter_reward": 0.0,
            "values_reward": 0.0,
            "old_total_reward": 0.0,
        }

        if "<tool_call>" not in ans:
            # if "<tool_call>" not in response and "</tool_call>" not in response:
            #     reward = max_possible_reward
            # else:
            #     reward = min_possible_reward
            rewards.append(reward)
            continue

        gt_tool_call = ans.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        gt_tools = gt_tool_call.split("\n")
        gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

        try:
            # Change here as a constrint in training: if the format is not correct,
            # directly give the lowest possible score
            assert "<tool_call>" in response
            assert "</tool_call>" in response
            pd_tools = response.split("<tool_call>")[1].split("</tool_call>")[0].strip().split("\n")
            pd_tools = [json.loads(tool) for tool in pd_tools]
            reward = compute_tool_call_reward(
                gt_tools, pd_tools, max_possible_reward, min_possible_reward, do_print
            )  # top reward is 2
        except Exception:
            reward = {
                "total_reward": min_possible_reward,
                "function_name_reward": min_possible_reward,
                "parameter_reward": min_possible_reward,
                "values_reward": min_possible_reward,
                "old_total_reward": min_possible_reward
            }

        rewards.append(reward)

    if do_print:
        print("\n======= Reward for <tool call> =======")
        print("Reward function for <tool call> correctness is called ...")
        print(rewards)
    return rewards


def compute_score(data_source, solution_str, ground_truth, extra_info, step=0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning."
    Proceedings of the 62nd Annual Meeting of the Association for
    Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    exp_name = extra_info.get("experiment_name", "")
    if "llama" in exp_name:
        predict_str = (
            solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
        )
    elif "qwen" in exp_name:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
        # raise NotImplementedError(f"Unknown model name: {exp_name}")

    tool_max_possible = 3.0
    tool_min_possible = -3.0

    format_max_possible = 1.0
    format_min_possible = 0.0

    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]

    do_print = random.randint(1, 64) == 1

    fomrat_score = customize_format_reward_func(
        completions, answer, step, format_max_possible, format_min_possible, do_print
    )[0]
    correctness_rewards = customize_correctness_reward_tool(
        completions, answer, step, tool_max_possible, tool_min_possible, do_print
    )[0]
    correctness_score = correctness_rewards["total_reward"]
    response_length = extra_info.get("response_length")
    if response_length is None:
        # Standalone callers do not have token ids. The GDPO reward manager
        # supplies the exact token length; whitespace counting is only a fallback.
        response_length = len(predict_str.split())
    length_score = compute_length_reward(response_length)

    # score = fomrat_score + correctness_score + length_score
    score = fomrat_score + correctness_score

    result = {
        "score": score,
        "format_reward": fomrat_score,
        "accuracy_reward": correctness_score,
        # "length_reward": length_score,
        "function_name_reward": correctness_rewards["function_name_reward"],
        "parameter_reward": correctness_rewards["parameter_reward"],
        "values_reward": correctness_rewards["values_reward"],
    }

    return result
