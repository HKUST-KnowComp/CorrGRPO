import json
import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    repo = root
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    from src.evaluate_prompted_agent import predict_one_case
    from src.models import MODELS
    from src.output_parsing import evaluate_output_prompted
    from src.prompts.agent_prompts import PROMPT_DICT
    from src.utils import get_tool_dict

    model_name = os.environ.get("REMOTE_MODEL", "deepseek-v4-flash")
    params = {"model_name": model_name}
    model = MODELS["GPT"](params)
    system_prompt, user_prompt = PROMPT_DICT["InjecAgent"]
    with open("data/test_cases_dh_base.json", encoding="utf-8") as handle:
        item = json.load(handle)[0]

    output = predict_one_case(
        system_prompt,
        user_prompt,
        params,
        item,
        model,
        get_tool_dict(),
    )
    parsed = evaluate_output_prompted(
        output,
        item["Attacker Tools"][0],
        item["Attacker Tools"],
        item["User Tool"],
        item["Attacker Instruction"],
    )
    print(
        f"model={model_name} output_chars={len(output)} "
        f"eval={parsed.get('eval')!r} output_preview={output[:200]!r}"
    )
    if not output:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
