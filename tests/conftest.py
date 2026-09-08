import sys
from pathlib import Path

# 저장소 루트를 import 경로에 추가한다.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
