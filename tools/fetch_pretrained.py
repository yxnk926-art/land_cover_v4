# -*- coding: utf-8 -*-
"""Download registered ImageNet checkpoints and verify their pinned SHA256.

示例（训练服务器需经 7897 代理访问外网）：
    HTTPS_PROXY=http://127.0.0.1:7897 python tools/fetch_pretrained.py --all
    python tools/fetch_pretrained.py --names resnet50.tv_in1k --dir /data/yfk/pretrained

已存在且哈希一致的文件直接跳过；下载先写入临时文件，校验通过后才改名，
哈希不符时删除临时文件并以非零状态退出。代理取自标准 HTTPS_PROXY 环境变量。
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.pretrained import REGISTRY, default_pretrained_dir  # noqa: E402


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(spec, directory):
    target = directory / spec.filename
    if target.is_file() and sha256_of(target) == spec.sha256:
        print(f"[skip] {spec.name}: {target} 已存在且哈希一致")
        return True
    temp = target.with_name(target.name + ".part")
    print(f"[get ] {spec.name} <- {spec.url}")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "land-cover-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(temp, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    actual = sha256_of(temp)
    if actual != spec.sha256:
        temp.unlink()
        print(f"[FAIL] {spec.name}: SHA256 {actual[:12]} != 期望 {spec.sha256[:12]}")
        return False
    os.replace(temp, target)
    print(f"[ ok ] {spec.name}: {target}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--names", nargs="+", choices=sorted(REGISTRY))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--dir", type=Path, default=None,
                        help="目标目录（默认 $LAND_COVER_PRETRAINED_DIR 或仓库同级 pretrained/）")
    args = parser.parse_args()
    directory = (args.dir or default_pretrained_dir()).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(REGISTRY) if args.all else args.names
    ok = all([fetch(REGISTRY[name], directory) for name in names])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
