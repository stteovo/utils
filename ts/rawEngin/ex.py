from base import *

try:
    import Tetrachromacy as tetra

    print("Init Raw Engine (PyRawEngine Version = %s) ... " % (tetra.__version__))
except BaseException as be:
    msg = "开始前请先阅读 Readme.md，并确认正确 install 了 Tetra \n"
    print(msg, be)
raw_engine_obj = None

if not raw_engine_obj:
    raw_engine_obj = tetra.RawEngine()
    raw_engine_obj.InitEnv(type=tetra.EngineType.kSIMD, support_raw=False)






