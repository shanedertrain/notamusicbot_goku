"""Migrate legacy ``input/users.json`` (dict- or list-shaped) to ``config/users.yaml``.

The source file is never modified; a timestamped backup of the previous
destination is written next to it before overwriting.

Usage:
    python -m notamusicbot.tools.migrate_users [--source input/users.json] \
        [--dest config/users.yaml]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from notamusicbot.domain.users import migrate_users_file

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m notamusicbot.tools.migrate_users",
        description="Migrate legacy users.json to config/users.yaml.",
    )
    parser.add_argument("--source", default="input/users.json")
    parser.add_argument("--dest", default="config/users.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source, dest = Path(args.source), Path(args.dest)
    if not source.is_file():
        logger.error("Source file not found: %s", source)
        return 2
    if dest.is_file():
        backup = dest.with_name(f"{dest.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}.yaml")
        shutil.copyfile(dest, backup)
        logger.info("Backed up existing destination to %s", backup)
    users = migrate_users_file(source, dest)
    logger.info("Migrated %d users to %s", len(users), dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
