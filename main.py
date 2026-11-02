"""Interactive CLI:  python main.py -c config.yaml"""
import argparse
import json
import logging

from agent.config import build_agent, load_config


def confirm(name, args):
    print(f"\n[confirm] {name}({json.dumps(args, ensure_ascii=False)})")
    return input("Run this API call? [y/N] ").strip().lower() in ("y", "yes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-q", "--query", help="Ask one question and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    agent = build_agent(load_config(a.config), confirm=confirm)
    if a.query:
        print(agent.run(a.query))
        return
    print("Agent ready. Type 'exit' to quit, 'reset' to clear history.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit"):
            break
        if q == "reset":
            agent.reset()
            continue
        if q:
            print(agent.run(q))


if __name__ == "__main__":
    main()
