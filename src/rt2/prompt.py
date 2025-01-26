from __future__ import annotations
"""
MCRT2 Python utilities prompt
"""

__author__ = "Chang-Min Lee"
__copyright__ = "Copyright 2022, Seoul National University"
__credits__ = ["Chang-Min Lee"]
__license__ = None
__maintainer__ = "Chang-Min Lee"
__email__ = "dlc2048@postech.ac.kr"
__status__ = "Production"

from collections.abc import Iterable
import sys
import six


class Prompt:
    def __init__(self, dim: int = 1, argv: None | Iterable = None):
        self._dim  = dim
        self._argv = argv

    def __getitem__(self, args: Iterable | str):
        argv = self._argv if self._argv is not None else sys.argv
        n = len(argv)
        if isinstance(args, six.string_types):
            item_list = [args]
        elif isinstance(args, Iterable):
            item_list = args
        else:
            item_list = [args]
        for i in range(1, n):
            arg = argv[i]
            for item in item_list:
                if arg == item:
                    return argv[i + 1: i + 1 + self._dim] if i + self._dim < n else ""
        return None


class Interactive:
    __argv = []

    @staticmethod
    def wait() -> str:
        line = input()
        if line.isspace() or not line:
            return ''
        Interactive.__argv = line.split()
        return Interactive.function()

    @staticmethod
    def prompt(dim: int = 1) -> Prompt:
        return Prompt(dim, Interactive.__argv)

    @staticmethod
    def function() -> str:
        return Interactive.__argv[0]
