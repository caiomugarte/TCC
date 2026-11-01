# ibov-ifix-allocation-series proofs

**HEAD**: 065bdda
**Tree**: clean
**Result**: 28 pass, 0 fail; mutations 0 killed, 0 not

| Check | Proof | Ran | Passed | Result |
| --- | --- | --- | --- | --- |
| C1 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_allocation` | 1 | - | pass |
| C1 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_selector_r` | 1 | - | pass |
| C2 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_series_ide` | 1 | - | pass |
| C3 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_series_ide` | 1 | - | pass |
| C4 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_derived_in` | 1 | - | pass |
| C5 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_validate_r` | 1 | - | pass |
| C6 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_validate_m` | 1 | - | pass |
| C6 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation_data -k test_missing_prem` | 1 | - | pass |
| C7 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_validate_d` | 1 | - | pass |
| C8 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_manifest -k test_selection_` | 1 | - | pass |
| C9 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_build_miss` | 1 | - | pass |
| C10 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_build_with` | 1 | - | pass |
| C11 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_build_comm` | 1 | - | pass |
| C12 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_build_mult` | 1 | - | pass |
| C13 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_snapshot_registry -k test_class_seri` | 1 | - | pass |
| C14 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation_data -k test_premium_retu` | 1 | - | pass |
| C15 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation_data -k test_premium_load` | 1 | - | pass |
| C16 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation_data -k test_premium_load` | 1 | - | pass |
| C17 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_basic_snapshot_lo` | 1 | - | pass |
| C17 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_loader_a` | 1 | - | pass |
| C17 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_loader_u` | 1 | - | pass |
| C18 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_fetch_al` | 1 | - | pass |
| C19 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_cli_reje` | 1 | - | pass |
| C20 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_cli_ifix` | 1 | - | pass |
| C21 | `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest tests.test_allocation -k test_snapshot_fetch_me` | 1 | - | pass |
| C22 | `grep -q -e 'data/snapshot-2020/ibovespa.csv' docs/snapshot_registry.md && grep -q -e 'data/snapshot-2020/ifix.` | count unknown | - | pass |
| C23 | `! grep -n -e 'caio_fiis.csv. is required' -e 'are never mapped' -e 'resolves to .caio_fiis.csv.' -e 'no IFIX s` | count unknown | - | pass |
| C24 | `! grep -n -e 'never reads IFIX' py/allocation_data.py && ! grep -n -e 'multi-ticker sleeves rebuilt' py/snapsh` | count unknown | - | pass |
