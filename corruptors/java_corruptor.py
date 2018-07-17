"""Corrupts java source code"""

import random
import string
import javalang
import re

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

def _misspell_variable(s):
    tree = javalang.parse.parse(s)
    variables = []
    for _, node in tree.filter(javalang.tree.VariableDeclarator):
        variables.append(node)
        
    var_name = random.choice(variables).name
    occurances = []
    for occurance in re.finditer(r'\b(' + var_name + r')\b', s):
        occurances.append(occurance)

    if len(occurances) <= 1:
        return

    chosen_occurance = random.choice(occurances[1:])
    r = random.random()
    if r >= 0.66 and len(var_name) > 1:
        drop_char = random.randint(0, len(var_name) - 1)
        s = s[:chosen_occurance.start()] + var_name[:drop_char] + var_name[drop_char+1:] + s[chosen_occurance.end():]
    elif r >= 0.33:
        add_char = random.choice(string.lowercase)
        add_loc = random.randint(0, len(var_name) - 1)
        s = s[:chosen_occurance.start()] + var_name[:add_loc] + add_char + var_name[add_loc:] + s[chosen_occurance.end():]
    else:
        change_char = random.randint(0, len(var_name) - 2)
        s = s[:chosen_occurance.start()] + var_name[:change_char] + var_name[change_char+1] + var_name[change_char] + var_name[change_char+2:] + s[chosen_occurance.end():]

    return s
