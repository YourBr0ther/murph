# client/src/murph_client/__main__.py
import os
import sys

# Suppress ALSA errors on Linux (must happen before pyaudio import)
if sys.platform == "linux":
    import ctypes
    from ctypes import c_char_p, c_int

    def _null_error_handler(filename, line, function, err, fmt):
        pass

    _ERROR_HANDLER = ctypes.CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    _c_error_handler = _ERROR_HANDLER(_null_error_handler)

    try:
        _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
        _asound.snd_lib_error_set_handler(_c_error_handler)
    except OSError:
        pass

    # Redirect stderr during imports to suppress ONNX/JACK C++ warnings
    _stderr_fd = sys.stderr.fileno()
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _saved_stderr_fd = os.dup(_stderr_fd)
    os.dup2(_devnull_fd, _stderr_fd)

import asyncio
from murph_client.main import main

# Restore stderr after imports
if sys.platform == "linux":
    os.dup2(_saved_stderr_fd, _stderr_fd)
    os.close(_saved_stderr_fd)
    os.close(_devnull_fd)

if __name__ == "__main__":
    asyncio.run(main())
