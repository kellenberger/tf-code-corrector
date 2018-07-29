"""Corrupts java source code"""

import random
import string
import javalang
import re

BRACKETS = ['(', ')', '[', ']', '{', '}']

def corrupt(s):
    if not random.random() > 0.75:
        return s

    # switch_statement and misspell_variable are not always possible. That's why they get a higher probability
    if random.random() > 0.5:
        r = _switch_statement_lines(s)
        if not s == r:
            return r

    choice = random.random()
    if choice > 0.65:
        r = _misspell_variable(s)
        if not s == r:
            return r
    if choice > 0.5:
        r = _change_method_return(s)
        if not s == r:
            return r
    if choice > 0.3:
        return _remove_bracket(s)
    else:
        return _remove_semicolon(s)


def corrupt_simple(s):
    if random.random() > 0.5:
        return _remove_bracket(s)
    else:
        return _remove_semicolon(s)


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

def _misspell_variable(s):
    try:
        tree = javalang.parse.parse(s)
    except:
        return s
    variables = []
    for _, node in tree.filter(javalang.tree.VariableDeclarator):
        variables.append(node)

    for _, node in tree.filter(javalang.tree.FormalParameter):
        variables.append(node)

    if len(variables) == 0:
        return s

    random.shuffle(variables)
    for variable in variables:
        var_name = variable.name
        occurances = []
        for occurance in re.finditer(r'\b(' + var_name + r')\b', s):
            occurances.append(occurance)

        if len(occurances) <= 1:
            continue

        chosen_occurance = random.choice(occurances[1:])
        r = random.random()
        try:
            if r >= 0.66 and len(var_name) > 1:
                drop_char = random.randint(0, len(var_name) - 1)
                s = s[:chosen_occurance.start()] + var_name[:drop_char] + var_name[drop_char+1:] + s[chosen_occurance.end():]
            elif r >= 0.33 and len(var_name) > 1:
                change_char = random.randint(0, len(var_name) - 2)
                s = s[:chosen_occurance.start()] + var_name[:change_char] + var_name[change_char+1] + var_name[change_char] + var_name[change_char+2:] + s[chosen_occurance.end():]
            else:
                add_char = random.choice(string.lowercase)
                add_loc = random.randint(0, len(var_name) - 1)
                s = s[:chosen_occurance.start()] + var_name[:add_loc] + add_char + var_name[add_loc:] + s[chosen_occurance.end():]
            break
        except:
            continue

    return s

def _switch_statement_lines(s):
    try:
        tree = javalang.parse.parse(s)
    except:
        return s
    statements = []
    for _, node in tree.filter(javalang.tree.LocalVariableDeclaration):
        start = node.position[1]
        end = s[start:].find(';') + start + 1
        statements.append({'class': type(node), 'start': start, 'end': end})

    for _, node in tree.filter(javalang.tree.StatementExpression):
        node = node.children[1]
        if not isinstance(node, javalang.tree.Assignment) and not isinstance(node, javalang.tree.MethodInvocation):
            continue
        if hasattr(node, 'position') and node.position:
            start = node.position[1]
        elif hasattr(node.children[0], 'position') and node.children[0].position:
            start = node.children[0].position[1]
        else:
            continue
        end = s[start:].find(';') + start + 1
        statements.append({'class': type(node), 'start': start, 'end': end})

    if len(statements) <= 1:
        return s

    statements = sorted(statements, key=lambda x: x['start'])
    possible_statements = []
    previous_statement = statements[0]
    for statement in statements[1:]:
        if previous_statement['end'] + 1 == statement['start'] and previous_statement['class'] != statement['class']:
            possible_statements.append((previous_statement, statement))
        previous_statement = statement

    if len(possible_statements) == 0:
        return s

    first, second = random.choice(possible_statements)
    return s[:first['start']] + s[second['start']:second['end']] + s[first['start']:first['end']] + s[second['end']:]

def _change_method_return(s):
    try:
        tree = javalang.parse.parse(s)
    except:
        return s

    methods = []
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        methods.append(node)

    if len(methods) == 0:
        return s

    random.shuffle(methods)
    for method in methods:
        if method.return_type:
            method_declaration = method.return_type.name + " " + method.name
            new_declaration = "void " + method.name
        else:
            method_declaration = "void " + method.name
            new_declaration = "int " + method.name

        if s.find(method_declaration) != -1:
            s = s.replace(method_declaration, new_declaration, 1)
            break

    return s
