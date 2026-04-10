#!/usr/bin/env python3
"""Wrapper around evaluator.py for research-everything convention."""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from evaluator import evaluate

def sanity_check():
    """Run evaluator on initial_program.py (trivial solution) to verify harness works."""
    initial = os.path.join(os.path.dirname(__file__), "initial_program.py")
    result = evaluate(initial)
    if "error" in result and result.get("combined_score", 0) == 0:
        print(f"Sanity check FAILED: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2))
    print("Sanity check PASSED (harness runs, trivial solution scored)")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", type=str, help="Path to solution .py file")
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args()

    if args.sanity_check:
        sanity_check()
    elif args.evaluate:
        result = evaluate(args.evaluate)
        print(json.dumps(result, indent=2))
        if "error" in result and result.get("combined_score", 0) == 0:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
