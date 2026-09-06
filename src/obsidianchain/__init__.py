"""obsidianchain — offline Bitcoin forensics prototype (NTRO PS 26146).

Design constraint: every runtime code path must work with no network
interface present. Nothing in this package may open a socket.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
