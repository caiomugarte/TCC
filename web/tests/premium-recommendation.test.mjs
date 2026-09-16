import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const clientSource = await readFile(new URL("../lib/api-client.ts", import.meta.url), "utf8");
const viewSource = await readFile(new URL("../components/recommendation/recommendation-view.tsx", import.meta.url), "utf8");

test("Premium client exposes start and account-scoped polling endpoints", () => {
  assert.match(clientSource, /startPremiumRecommendation/);
  assert.match(clientSource, /\/v1\/premium\/recommendations/);
  assert.match(clientSource, /getRecommendation/);
  assert.match(clientSource, /\/v1\/recommendations\/\$\{recommendationId\}/);
});

test("recommendation view keeps Premium polling separate from Basic creation", () => {
  assert.match(viewSource, /account\.plan !== "premium"/);
  assert.match(viewSource, /startPremiumRecommendation/);
  assert.match(viewSource, /getRecommendation/);
  assert.match(viewSource, /status === "completed" \|\| next\.status === "failed"/);
  assert.match(viewSource, /requestRef\.current \+= 1/);

  const premiumBranch = viewSource.split('if (account.plan !== "premium")')[1];
  assert.ok(premiumBranch);
  assert.ok(!premiumBranch.split("const initial")[1].split("if (!active())")[0].includes("createRecommendation"));
});
