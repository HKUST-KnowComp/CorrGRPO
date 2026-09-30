# AgentDojo

From corrgrpo-src, run `bash agent_security_rl/run.sh` to train, convert weights, and evaluate.

Set the model, training options, and BENCHMARK at the top of that script.
See the [run instructions](../../README.md).

The matched gpu2 environment also needs `deepdiff` and `tenacity` for AgentDojo.
Cohere and Google SDKs are optional and loaded only when those providers are selected;
local training and local-model evaluation do not require them.
