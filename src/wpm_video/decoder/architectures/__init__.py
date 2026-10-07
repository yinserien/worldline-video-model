"""Built-in decoder architectures.

Importing a module here registers its component in the process-local registry
(:mod:`wpm_video.decoder.registry`). ``conv`` is imported by the package itself, so
the built-in architecture is always available; a custom architecture lives outside
the package and is registered by the program that uses it -- there is no plugin
discovery and no import driven by checkpoint contents.
"""

__all__ = ["conv"]
