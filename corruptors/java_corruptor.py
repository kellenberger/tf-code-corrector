"""Corrupts java source code"""

import random
import string
import javalang
import re

BRACKETS = ['(', ')', '[', ']', '{', '}']
EOL_ID = 4
CORRUPT_ALL = False
CLASS_START = "public class A {\n"
CLASS_END = "\n}"

def corrupt(s):
    if not CORRUPT_ALL and not random.random() > 0.75:
        return s

    s = _prepare(s)
    choice = random.random()

    if choice > 0.75:
        s = _misspell_variable(s)
    elif choice > 0.5:
        s = _change_method_return(s)
    elif choice > 0.25:
        s = _remove_bracket(s)
    else:
        s = _remove_semicolon(s)

    return _unprepare(s)

    # # switch_statement and misspell_variable are not always possible. That's why they get a higher probability
    # if random.random() > 0.5:
    #     r = _switch_statement_lines(s)
    #     if not s == r:
    #         return r
    #
    # choice = random.random()
    # if choice > 0.65:
    #     r = _misspell_variable(s)
    #     if not s == r:
    #         return r
    # if choice > 0.5:
    #     r = _change_method_return(s)
    #     if not s == r:
    #         return r
    # if choice > 0.3:
    #     return _remove_bracket(s)
    # else:
    #     return _remove_semicolon(s)


def _prepare(s):
    s = re.sub(chr(EOL_ID), "\n", s)
    return CLASS_START + s + CLASS_END

def _unprepare(s):
    s = s[len(CLASS_START):len(s) - len(CLASS_END)]
    s = s.strip()
    return re.sub("\n+", chr(EOL_ID), s)

def _remove_bracket(s):
    bracket_indices = [i for i, c in enumerate(s) if c in BRACKETS]
    bracket_indices = bracket_indices[1:-1]
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

        changed = False
        for chosen_occurance in occurances[1:]:
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
                changed = True
                break
            except:
                continue

        if changed:
            break

    return s

def _switch_statement_lines(s):
    try:
        tree = javalang.parse.parse(s)
    except:
        return s
    statements = []
    for _, node in tree.filter(javalang.tree.LocalVariableDeclaration):
        line = node.position[0] - 1
        statements.append({'class': type(node), 'line': line})

    for _, node in tree.filter(javalang.tree.StatementExpression):
        node = node.children[1]
        if not isinstance(node, javalang.tree.Assignment) and not isinstance(node, javalang.tree.MethodInvocation):
            continue
        if hasattr(node, 'position') and node.position:
            line = node.position[0] - 1
        elif hasattr(node.children[0], 'position') and node.children[0].position:
            line = node.children[0].position[0] - 1
        else:
            continue
        statements.append({'class': type(node), 'line': line})

    if len(statements) <= 1:
        return s

    statements = sorted(statements, key=lambda x: x['line'])
    possible_statements = []
    previous_statement = statements[0]
    for statement in statements[1:]:
        if previous_statement['line'] + 1 == statement['line'] and previous_statement['class'] != statement['class']:
            possible_statements.append((previous_statement, statement))
        previous_statement = statement

    if len(possible_statements) == 0:
        return s

    first, second = random.choice(possible_statements)
    splits = s.split('\n')
    return "\n".join(splits[:first['line']] + [splits[second['line']]] + [splits[first['line']]] + splits[second['line']+1:])

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

def corruptable(s):
    s = _prepare(s)
    try:
        if s == _switch_statement_lines(s):
            return False
        if s == _misspell_variable(s):
            return False
        if s == _change_method_return(s):
            return False
        if s == _remove_bracket(s):
            return False
        if s == _remove_semicolon(s):
            return False
    except RuntimeError:
        return False
    return True
