"""Compact console formatting shared by training and search scripts."""

from datetime import datetime


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def panel(title: str) -> None:
    print(f"\n+-- {title}")


def line(text: str) -> None:
    print(f"|  {text}")


def section(title: str) -> None:
    print(f"+-- {title}")


def close(text: str) -> None:
    print(f"+-- {text}")
