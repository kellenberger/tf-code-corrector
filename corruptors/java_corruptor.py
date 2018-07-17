"""Corrupts java source code"""

import random

BRACKETS = ['(', ')', '[', ']', '{', '}']

def corrupt(s):
    s = s.strip()
    if random.random() > 0.9 and len(s) > 1:
        s = _remove_bracket(s)
    if random.random() > 0.9 and len(s) > 1:
        s = _remove_semicolon(s)
    if random.random() > 0.9 and len(s) > 1:
        s = _add_typo(s)
    return s

def _remove_bracket(s):
    bracket_indices = [i for i, c in enumerate(s) if c in BRACKETS]
    if bracket_indices:
        drop_index = random.choice(bracket_indices)
        s = s[:drop_index] + s[drop_index+1:]
    return s

def _remove_semicolon(s):
    semicolon_indices = [i for i, c in enumerate(s) if c == ';']
    if semicolon_indices:
        drop_index = random.choice(semicolon_indices)
        s = s[:drop_index] + s[drop_index+1:]
    return s

def _add_typo(s):
    change_char = random.randint(0, len(s)-2)
    s = s[:change_char] + s[change_char+1] + s[change_char] + s[change_char+2:]
    return s
