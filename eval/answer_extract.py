#!/usr/bin/env python3
"""Answer extraction for GSM8K-style scoring (fixed scorer).

Precedence: last **bolded** number wins; else last number in text.
Normalization: strip $ and commas; trailing '.' and '.0' variants.

Root-cause note: the original in-run scorer used re.search (FIRST number),
which failed correct markdown answers that carry intermediate numbers
(e.g. "16 eggs ... 3 + 4 = **7** ... = **$18**" with reference 18).
"""
from __future__ import annotations

import re

_BOLD_NUM = re.compile(r"\*\*\\?\$?(-?\d[\d,]*\.?\d*)\\?\$?\*\*")
_ANY_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").strip().rstrip(".")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or None


def final_answer(text: str | None) -> str | None:
    if not text:
        return None
    bolds = _BOLD_NUM.findall(text)
    if bolds:
        return norm_num(bolds[-1])
    nums = _ANY_NUM.findall(text)
    return norm_num(nums[-1]) if nums else None


def matches(text: str | None, ref: str) -> bool:
    got = final_answer(text)
    want = norm_num(ref)
    return got is not None and want is not None and got == want
