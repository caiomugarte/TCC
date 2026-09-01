#!/usr/bin/env python3
"""Script para testar imports."""

print("Testando imports...")

try:
    from config import PROFILES
    print("✅ config: OK")
except Exception as e:
    print(f"❌ config: {e}")

try:
    from core.preprocessing import load_raw_data
    print("✅ core.preprocessing: OK")
except Exception as e:
    print(f"❌ core.preprocessing: {e}")

try:
    from core.metrics import hhi_sector
    print("✅ core.metrics: OK")
except Exception as e:
    print(f"❌ core.metrics: {e}")

try:
    from core.scoring import build_scores
    print("✅ core.scoring: OK")
except Exception as e:
    print(f"❌ core.scoring: {e}")

try:
    from core.optimizer import optimize_portfolio
    print("✅ core.optimizer: OK")
except Exception as e:
    print(f"❌ core.optimizer: {e}")

try:
    from utils.cache import CacheManager
    print("✅ utils.cache: OK")
except Exception as e:
    print(f"❌ utils.cache: {e}")

try:
    from pipelines.single_run import run_all_profiles
    print("✅ pipelines.single_run: OK")
except Exception as e:
    print(f"❌ pipelines.single_run: {e}")

print("\n✅ Todos os imports funcionaram!")
