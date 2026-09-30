import os
import sys

from openai import OpenAI


def main() -> None:
    model = os.environ.get("REMOTE_MODEL", "deepseek-v4-flash")
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("OPENAI_API_KEY and OPENAI_BASE_URL must be set")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=2,
        timeout=60,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0,
        max_tokens=128,
    )
    choice = response.choices[0]
    message = choice.message
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None) or ""
    text = content or reasoning
    print(
        f"model={model} finish_reason={choice.finish_reason!r} "
        f"content={content.strip()!r} reasoning_content={reasoning.strip()[:200]!r}"
    )
    if "OK" not in text.upper():
        sys.exit(1)


if __name__ == "__main__":
    main()
