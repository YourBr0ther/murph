# client/src/murph_client/__main__.py
import os
import sys

# Suppress ONNX Runtime warnings (must be set before ANY imports)
os.environ["ORT_LOG_LEVEL"] = "3"
os.environ["ORT_DISABLE_ALL"] = "1"
# Tell ONNX to skip GPU/CUDA provider entirely
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_EXECUTION_PROVIDER"] = "CPUExecutionProvider"

# Suppress ALSA errors on Linux
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

import asyncio
from murph_client.main import main

if __name__ == "__main__":
    asyncio.run(main())
