import sys

_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"

_TAG = f"{_CYAN}[MULTIGPU]{_RESET}"


def info(msg: str):
    print(f"{_TAG} {msg}", flush=True)

def ok(msg: str):
    print(f"{_TAG} {_GREEN}{msg}{_RESET}", flush=True)

def warn(msg: str):
    print(f"{_TAG} {_YELLOW}{msg}{_RESET}", flush=True)

def err(msg: str):
    print(f"{_TAG} {_RED}{msg}{_RESET}", file=sys.stderr, flush=True)
