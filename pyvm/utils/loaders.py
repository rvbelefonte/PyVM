"""
Functions for loading example data
"""
import os


def get_test_data_dirs():
    """
    Finds the absolute paths to data directories.

    The PyVM modules are installed to a custom installation directory.
    That is the path cannot be predicted. This functions searches through
    all PyVM modules for "tests/data" subdirectories.

    :returns: `list` of "tests/data" directories
    """
    from pyvm import __path__ as ROOT
    matches = []
    for root, dirnames, filenames in os.walk(ROOT[0]):
        for _dirname in dirnames:
            if _dirname == 'data' and os.path.basename(root) == 'tests':
                matches.append(os.path.join(root, _dirname))
    return matches


def get_example_file(filename):
    """
    Function to find the absolute path of a test data file.

    The PyVM modules are installed to a custom installation directory.
    That is the path cannot be predicted. This functions searches for all
    PyVM modules and checks weather the file is in any of
    the "tests/data" subdirectories.

    :param filename: A test file name to which the path should be returned.
    :returns: Full path to file.

    >>> get_example_file('example_file.dat')  # doctest: +SKIP
    /path/to/module/tests/data/example_file.dat
    """
    test_data_dirs = get_test_data_dirs()
    for dirname in test_data_dirs:
        filepath = os.path.join(dirname, filename)
        if os.path.isfile(filepath):
            return filepath
    msg = "Could not find file {:} in {:}".format(filename,
                                                  get_test_data_dirs())
    raise IOError(msg)


def get_resources_dirs():
    """
    Finds the absolute paths to resources directories.

    The PyVM modules are installed to a custom installation directory.
    That is the path cannot be predicted. This functions searches through
    all PyVM modules for "tests/data" subdirectories.

    :returns: `list` of "resources" directories
    """
    from pyvm import __path__ as ROOT
    matches = []
    for root, dirnames, filenames in os.walk(ROOT[0]):
        for _dirname in dirnames:
            if _dirname == 'resources':
                matches.append(os.path.join(root, _dirname))
    return matches

def get_resource_file(filename):
    """
    Function to find the absolute path of a resource file.

    The PyVM modules are installed to a custom installation directory.
    That is the path cannot be predicted. This functions searches for all
    PyVM modules and checks weather the file is in any of
    the "resources" subdirectories.

    :param filename: A test file name to which the path should be returned.
    :returns: Full path to file.

    >>> get_resource_file('example_file.dat')  # doctest: +SKIP
    /path/to/module/tests/data/example_file.dat
    """
    dirs = get_resources_dirs()
    for dirname in dirs:
        filepath = os.path.join(dirname, filename)
        if os.path.isfile(filepath):
            return filepath
    msg = "Could not find file {:} in {:}".format(filename,
                                                  dirs())
    raise IOError(msg)
