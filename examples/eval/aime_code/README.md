# AIME Code

AIME benchmark evaluation using `CodeSandboxEnv` (AWS Bedrock AgentCore Code Interpreter).

## Files

- `code_sandbox_env.py` - Environment hook using `CodeSandboxEnv` with Python execution

## Usage

```bash
strands-env eval aime-2024 \
    --env examples/eval/aime_code/code_sandbox_env.py \
    --base-url http://localhost:30000
```

Requires AWS credentials for Bedrock AgentCore. See `strands-env eval --help` for all CLI options.
