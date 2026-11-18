import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { STALE_SNAPSHOT_MAX_DAYS, staleSnapshotLabel } from "../lib/snapshot-age.ts";

const viewSource = await readFile(new URL("../components/recommendation/recommendation-view.tsx", import.meta.url), "utf8");

test("snapshot com até 35 dias não mostra aviso", () => {
  assert.equal(STALE_SNAPSHOT_MAX_DAYS, 35);
  assert.equal(staleSnapshotLabel("2026-10-13", new Date(2026, 10, 17)), null);
  assert.equal(staleSnapshotLabel("2026-11-17", new Date(2026, 10, 17)), null);
});

test("snapshot com mais de 35 dias mostra 'dados de DD/MM'", () => {
  assert.equal(staleSnapshotLabel("2026-10-12", new Date(2026, 10, 17)), "dados de 12/10");
  assert.equal(staleSnapshotLabel("2026-07-21", new Date(2026, 10, 17, 23, 59)), "dados de 21/07");
});

test("o limite conta dias de calendário, mesmo virando o ano", () => {
  assert.equal(staleSnapshotLabel("2025-12-31", new Date(2026, 1, 4)), null);
  assert.equal(staleSnapshotLabel("2025-12-30", new Date(2026, 1, 4)), "dados de 30/12");
});

test("data de corte que não é ISO não mostra aviso", () => {
  for (const cutoff of ["unknown", "", "2026-13-01", "2026-02-30", "21/07/2026"]) {
    assert.equal(staleSnapshotLabel(cutoff, new Date(2026, 10, 17)), null, cutoff);
  }
});

test("a carteira Premium concluída mostra o aviso calculado com a data de hoje", () => {
  assert.match(viewSource, /import \{ staleSnapshotLabel \} from "@\/lib\/snapshot-age"/);
  assert.match(
    viewSource,
    /completed\?\.plan === "premium"[\s\S]{0,40}staleSnapshotLabel\(completed\.snapshotCutoff, new Date\(\)\)/,
  );
  assert.match(viewSource, /\{staleSnapshot && <p className="[^"]*" role="status">\{staleSnapshot\}<\/p>\}/);
});
