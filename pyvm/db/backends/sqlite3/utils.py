"""
Sqlite utilities
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

def py2str(values):
    """
    Convert python variables to strings for use in SQLite statements.

    :param values: List of values to convert.
    :returns: List of strings.
    """
    if type(values) in [str, unicode]:
        values = [values]
    strings = []
    for v in values:
        if type(v) in [str, unicode]:
            strings.append("'" + str(v) + "'")
        else:
            strings.append(str(v))
    return strings

def format_search(match_dict, list_op='OR', key_op='AND', op='=',
        valid_fields=None):
    """
    Format a dictionary of search terms into a SQLite search string.

    Parameters
    ----------
    match_dict : dict
        Keywords and values to match.
    list_op : str, optional
        SQLite operator to use when combining multiple values for a single key.
    key_op : str, optional
        SQLite operator to use when combining multiple values.
    valid_fields: list
        List of possible fields to include from `match_dict`. Useful when
        building searches for different tables using the same values.

    Returns
    -------
    sql : str
        SQLite query
    """
    if not valid_fields:
        valid_fields = [k for k in match_dict]

    keys = [k for k in match_dict if k in valid_fields]

    fields = []
    _list_op = ' ' + list_op + ' '

    for k in keys:
        try:
            values = ['{:}{:}{:}'.format(k, op, v)\
                      for v in py2str(match_dict[k])]
        except TypeError:
            values = ['{:}{:}{:}'.format(k, op, py2str([match_dict[k]])[0])]
        fields.append('(' + _list_op.join(values) + ')')
    _key_op = ' ' + key_op + ' '
    sql = _key_op.join(fields)

    return sql
