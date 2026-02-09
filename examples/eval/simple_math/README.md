# Simple Math

Simple math benchmark using `CalculatorEnv` with a basic calculator tool.

## Files

- `calculator_env.py` - Environment hook using `CalculatorEnv`
- `simple_math_evaluator.py` - Custom evaluator hook with example problems

## Usage

With custom Simple Maths evaluator:
```bash
strands-env eval \
    --evaluator examples/eval/simple_math/simple_math_evaluator.py \
    --env examples/eval/simple_math/calculator_env.py \
    --base-url http://localhost:30000
```

See `strands-env eval --help` for all CLI options.
