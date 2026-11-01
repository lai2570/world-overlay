"""自動部署：GitHub 的 main 有新 commit（隊友的 PR 合併了）就拉下來，不用手動 git pull。

用法（取代直接跑 server.py，後面的參數照樣交給 server.py）：
  .venv/Scripts/python autodeploy.py --https

每 30 秒 git fetch 一次：
- 只改 static/：拉下來馬上生效（server.py 每次都從硬碟讀檔），後端不用重開
- 改到 .py：等目前沒有分析在跑（問 /api/busy）才重開 server.py，中間約 5 秒連不上
- 改到 requirements.txt：先 pip install 再重開
這台電腦有沒提交的修改、或有還沒推上去的 commit 時不拉，只印一次警告。
server.py 當掉也會自動重開。Cloudflare 通道不受影響（它連的是 localhost:8000）。
"""
import json
import os
import secrets
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BRANCH = "main"
EVERY = 30  # 秒


def log(msg: str):
    print(f"[autodeploy {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def start(args: list[str], token: str) -> subprocess.Popen:
    env = dict(os.environ, DEPLOY_TOKEN=token, PYTHONUNBUFFERED="1")
    return subprocess.Popen([sys.executable, "server.py", *args], cwd=ROOT, env=env)


def busy(port: int, https: bool, token: str) -> bool:
    url = f"{'https' if https else 'http'}://127.0.0.1:{port}/api/busy"
    req = urllib.request.Request(url, headers={"X-Deploy-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=5, context=ssl._create_unverified_context()) as r:
            return json.load(r)["running"] > 0
    except (OSError, ValueError, KeyError):
        return False  # 連不上就當沒在忙（可能本來就掛了，重開剛好）


def update() -> list[str] | None:
    """拉 origin/main。回傳改到的檔案；沒有新東西或拉不下來回傳 None。"""
    if git("fetch", "-q", "origin", BRANCH).returncode:
        return None  # 斷網，下次再試
    old, new = git("rev-parse", "HEAD").stdout.strip(), git("rev-parse", f"origin/{BRANCH}").stdout.strip()
    if old == new:
        return None
    if git("merge-base", "--is-ancestor", "HEAD", f"origin/{BRANCH}").returncode:
        warn(f"這台電腦有還沒推上去的 commit，先 git push 或 git pull，{new[:7]} 之後才會自動更新")
        return None
    r = git("merge", "--ff-only", "-q", f"origin/{BRANCH}")
    if r.returncode:
        warn(f"拉不下來（這台電腦有沒提交的修改？）：{r.stderr.strip()}")
        return None
    files = git("diff", "--name-only", old, new).stdout.split()
    log(f"更新到 {new[:7]}「{git('log', '-1', '--format=%s', new).stdout.strip()}」，{len(files)} 個檔案")
    return files


warned = set()


def warn(msg: str):
    if msg not in warned:
        warned.add(msg)
        log(msg)


def main():
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1]) if "--port" in args else 8000
    https = "--https" in args
    token = secrets.token_hex(16)
    proc = start(args, token)
    log(f"每 {EVERY} 秒檢查 GitHub 的 {BRANCH}")
    restart = False
    try:
        while True:
            time.sleep(EVERY)
            if proc.poll() is not None:
                log(f"server.py 停了（代碼 {proc.returncode}），重開")
                proc = start(args, token)
            files = update()
            if files and "requirements.txt" in files:
                log("安裝新的套件")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
            if files and any(f.endswith(".py") or f == "requirements.txt" for f in files):
                restart = True
            if restart:
                if busy(port, https, token):
                    warn("後端有改，等目前的分析跑完再重開")
                    continue
                log("重開 server.py")
                proc.terminate()
                proc.wait()
                proc = start(args, token)
                restart = False
                warned.clear()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
