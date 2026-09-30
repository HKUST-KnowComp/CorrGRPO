import os
import json
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
from src.utils import get_tool_dict, get_simulated_attacker_tool_response, get_score
from src.params import parse_arguments
from src.prompts.agent_prompts import PROMPT_DICT
from src.output_parsing import evaluate_output_prompted
from src.models import MODELS


def predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=False):
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return output

def main(params):
    print(params)
    model_class = MODELS[params['model_type']](params)
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    output_dir = os.environ.get("INJECAGENT_RESULT_DIR") or f"./results/prompted_{params['model_type']}_{params['model_name']}_{params['prompt_type']}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = './data'
    
    tool_dict = get_tool_dict()
    attacker_response_lock = threading.Lock()
    
    output_files = {
        attack: os.path.join(
            output_dir,
            f"test_cases_{attack}_{params['setting']}.json",
        )
        for attack in ['dh', 'ds']
    }
    for attack in params.get('attacks', ['dh', 'ds']):
        file_name = f"test_cases_{attack}_{params['setting']}.json"
        test_case_file = os.path.join(data_dir, file_name)
        output_file = output_files[attack]
        
        with open(test_case_file, 'r') as f:
            data = json.load(f)
        if params.get('max_cases', 0) > 0:
            data = data[:params['max_cases']]

        existing_data_step1 = {}
        existing_data_step2 = {}
        if params['use_cache']:
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    for line in f.readlines():
                        try:
                            tmp = json.loads(line)
                            if type(tmp['output']) == str and len(tmp['output']):
                                existing_data_step1[(tmp['User Instruction'], tmp['Tool Response'])] = tmp['output']
                            if type(tmp['output Step 2']) == str and len(tmp['output Step 2']):
                                existing_data_step2[(tmp['User Instruction'], tmp['Tool Response'], tmp['output'])] = tmp['output Step 2']
                        except:
                            continue
                print(f"Got {len(existing_data_step1)} existing outputs for the first step and {len(existing_data_step2)} existing outputs for the second step")
                        
        def evaluate_item(item):
            try:
                cache_key_step1 = (item['User Instruction'], item['Tool Response'])
                if params['use_cache'] and cache_key_step1 in existing_data_step1:
                    output = existing_data_step1[cache_key_step1]
                else:
                    output = predict_one_case(
                        system_prompt, user_prompt, params, item,
                        model_class, tool_dict,
                    )
                item['output'] = output

                parsed_output = evaluate_output_prompted(
                    item['output'], item['Attacker Tools'][0],
                    item['Attacker Tools'], item['User Tool'],
                    item["Attacker Instruction"],
                )
                item['eval'] = parsed_output['eval']
                if item['eval'] == 'invalid':
                    item['invalid_reason'] = parsed_output['invalid_reason']

                if not params['only_first_step'] and item['eval'] == 'succ' and attack == 'ds':
                    item["Attacker Tool Parameters"] = parsed_output["Attacker Tool Parameters"]
                    item["Attacker Tool Thought"] = parsed_output["Attacker Tool Thought"]
                    # This helper updates a shared JSON cache, so keep that small
                    # critical section serialized while model calls stay concurrent.
                    with attacker_response_lock:
                        item["Attacker Tool Response"] = get_simulated_attacker_tool_response(
                            item['Attacker Tools'][0],
                            item["Attacker Tool Parameters"],
                            tool_dict,
                        )

                    cache_key_step2 = (
                        item['User Instruction'], item['Tool Response'], item['output'],
                    )
                    if params['use_cache'] and cache_key_step2 in existing_data_step2:
                        output = existing_data_step2[cache_key_step2]
                    else:
                        output = predict_one_case(
                            system_prompt, user_prompt, params, item,
                            model_class, tool_dict, step_2=True,
                        )

                    item['output Step 2'] = output
                    parsed_output = evaluate_output_prompted(
                        output, item['Attacker Tools'][1],
                        item['Attacker Tools'], item['User Tool'],
                        item["Attacker Instruction"],
                    )
                    item['eval Step 2'] = parsed_output['eval']
                    if item['eval Step 2'] == 'invalid':
                        item['invalid_reason  Step 2'] = parsed_output['invalid_reason']
                return item
            except Exception as e:
                print(f"An error occurred: {e} in {output_file}")
                return None

        if not params['only_get_score']:
            num_workers = max(1, params.get('num_workers', 1))
            print(f"Using {num_workers} concurrent model request worker(s)")
            with open(output_file, 'w') as f:
                if num_workers == 1:
                    results = map(evaluate_item, data)
                    for item in tqdm(results, total=len(data)):
                        if item is not None:
                            f.write(json.dumps(item)+'\n')
                else:
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        results = executor.map(evaluate_item, data)
                        for item in tqdm(results, total=len(data)):
                            if item is not None:
                                f.write(json.dumps(item)+'\n')
    
    if all(os.path.exists(path) for path in output_files.values()):
        scores = get_score(output_files)
        print(json.dumps(scores, indent=True))
    else:
        print("Subset inference completed; full score requires both dh and ds result files.")

if __name__ == "__main__":
    params = parse_arguments(agent_type="prompted")

    main(params)
