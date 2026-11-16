"""Run-time package checks for v12: installed versions must equal the pins in requirements.txt (main venv) or
requirements-v12.txt (torch venv), and Python must be 3.13. Standard library only."""
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {'main': 'requirements.txt', 'torch': 'requirements-v12.txt'}


def pins(kind):
    out = {}
    for line in (ROOT / FILES[kind]).read_text().splitlines():
        line = line.split('#')[0].strip()
        if '==' in line:
            name, ver = line.split('==')
            out[name.strip()] = ver.strip()
    return out


def installed(name):
    """Installed version without a local build tag: the Linux CPU wheel reports torch as '2.15.0+cpu'."""
    try:
        return version(name).split('+')[0]
    except PackageNotFoundError:
        return None


def check(kind):
    bad = {n: (v, installed(n)) for n, v in pins(kind).items() if installed(n) != v}
    if platform.python_version_tuple()[:2] != ('3', '13'):
        bad['python'] = ('3.13', platform.python_version())
    if bad:
        raise SystemExit(f'{FILES[kind]}: version mismatch (pinned, installed): {bad}')
    return {n: installed(n) for n in pins(kind)} | {'python': platform.python_version()}
