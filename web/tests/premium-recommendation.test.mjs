import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const clientSource = await readFile(new URL("../lib/api-client.ts", import.meta.url), "utf8");
const typesSource = await readFile(new URL("../lib/api-types.ts", import.meta.url), "utf8");
const viewSource = await readFile(new URL("../components/recommendation/recommendation-view.tsx", import.meta.url), "utf8");

test("Premium retains its start endpoint and shared status polling", () => {
  assert.match(clientSource, /startPremiumRecommendation/);
  assert.match(clientSource, /\/v1\/premium\/recommendations/);
  assert.match(clientSource, /getRecommendation/);
  assert.match(clientSource, /\/v1\/recommendations\/\$\{recommendationId\}/);
});

test("Basic and Premium share the recommendation result contract", () => {
  assert.match(clientSource, /getLatestCompletedRecommendation/);
  assert.match(clientSource, /\/v1\/recommendations\/latest-completed/);
  assert.match(clientSource, /startPremiumRecommendation[\s\S]*Promise<Recommendation>/);
  assert.match(clientSource, /getRecommendation[\s\S]*Promise<Recommendation>/);
  assert.match(typesSource, /status: RecommendationStatus/);
  assert.match(typesSource, /stocks: RecommendationConstituent\[\]/);
  assert.match(typesSource, /fiis: RecommendationConstituent\[\]/);
});

test("page load reads stored runs without starting a refresh", () => {
  const loadBlock = viewSource.split("async function loadRecommendation")[1]?.split("async function startNewRun")[0];
  assert.ok(loadBlock);
  assert.match(loadBlock, /getLatestRecommendation/);
  assert.match(loadBlock, /getLatestCompletedRecommendation/);
  assert.doesNotMatch(loadBlock, /createRecommendation|startPremiumRecommendation/);
});

test("explicit action starts either plan and keeps Premium forced retry", () => {
  const startBlock = viewSource.split("async function startNewRun")[1]?.split("useEffect")[0];
  assert.ok(startBlock);
  assert.match(startBlock, /account\.plan === "premium"/);
  assert.match(startBlock, /startPremiumRecommendation\(retryPremium \? \{ force: true \} : \{\}, token\)/);
  assert.match(startBlock, /createRecommendation\(\{\}, token\)/);
  assert.match(viewSource, /onClick=\{\(\) => void startNewRun/);
});

test("page reload resumes polling for queued and running runs", () => {
  const loadBlock = viewSource.split("async function loadRecommendation")[1]?.split("async function startNewRun")[0];
  assert.ok(loadBlock);
  assert.match(loadBlock, /current\?\.status === "queued" \|\| current\?\.status === "running"/);
  assert.match(loadBlock, /pollRecommendation\(current\.id, token, active\)/);
});

test("polling backs off to a cap until the server reports a terminal state", () => {
  const pollBlock = viewSource.split("async function pollRecommendation")[1]?.split("async function loadRecommendation")[0];
  assert.ok(pollBlock);
  assert.match(pollBlock, /while \(active\(\)\)/);
  assert.match(pollBlock, /let delayMs = 500/);
  assert.match(pollBlock, /Math\.min\(delayMs \* 2, 5_000\)/);
  assert.match(pollBlock, /if \(next\.status === "completed" \|\| next\.status === "failed"\) return;/);
  assert.doesNotMatch(pollBlock, /attempt < 60|30_000|30000/);
});

test("current pending or failed status stays separate from latest completed result", () => {
  assert.match(viewSource, /const \[currentRun, setCurrentRun\]/);
  assert.match(viewSource, /const \[latestCompleted, setLatestCompleted\]/);
  assert.match(viewSource, /inProgress && currentRun/);
  assert.match(viewSource, /failed && currentRun/);
  assert.match(viewSource, /completed && \(/);
});

test("both plans render classes, stock and FII sleeves, and Status Invest retrieval time", () => {
  assert.match(viewSource, /completed\.classes\.map/);
  assert.match(viewSource, /completed\.stocks\.map/);
  assert.match(viewSource, /completed\.fiis\.map/);
  assert.match(viewSource, /selector_sources/);
  assert.match(viewSource, /retrieved_at/);
  assert.match(viewSource, /statusInvestRetrievedAt\(completed\)/);
  assert.doesNotMatch(viewSource, /premium && \(stocks\.length/);
});

test("in-progress copy stays truthful and polling stops only on terminal status or unmount", () => {
  assert.match(viewSource, /A execução continua no servidor/);
  assert.match(viewSource, /if \(next\.status === "completed" \|\| next\.status === "failed"\) return;/);
  assert.match(viewSource, /requestRef\.current \+= 1/);
});

const apiSchemaSource = await readFile(new URL("../../api/app/schemas/recommendation.py", import.meta.url), "utf8");

function typeBody(name) {
  const match = typesSource.match(new RegExp(`export type ${name} = \\{([\\s\\S]*?)\\n\\};`));
  assert.ok(match, `${name} is declared in api-types.ts`);
  return match[1];
}

function apiClassBody(name) {
  const header = `\nclass ${name}(`;
  const start = apiSchemaSource.indexOf(header);
  assert.ok(start >= 0, `class ${name} is declared in the API schema`);
  const rest = apiSchemaSource.slice(start + header.length);
  const end = rest.search(/\n(?=\S)/);
  return end < 0 ? rest : rest.slice(0, end);
}

// Same rule as to_camel in api/app/schemas/profile.py (the API alias generator).
function toCamel(snake) {
  const [head, ...tail] = snake.split("_");
  return head + tail.map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join("");
}

test("whole-unit fields are typed for both shapes", () => {
  const constituent = typeBody("RecommendationConstituent");
  assert.match(constituent, /priceBrl: number \| null;/);
  assert.match(constituent, /quantity: number \| null;/);
  assert.match(constituent, /amountBrl: number \| null;/);
  assert.match(constituent, /effectivePortfolioWeight: number \| null;/);
  const allocation = typeBody("AllocationClass");
  assert.match(allocation, /cashResidualBrl\?: number \| null;/);
  assert.match(allocation, /overTargetBrl\?: number \| null;/);
  const dropped = typeBody("DroppedConstituent");
  assert.match(dropped, /priceBrl: number;/);
  assert.match(dropped, /reasons: string\[\];/);
  const recommendation = typeBody("Recommendation");
  assert.match(recommendation, /stocksDropped: DroppedConstituent\[\];/);
  assert.match(recommendation, /fiisDropped: DroppedConstituent\[\];/);
});

test("whole-unit field names match the API schema aliases (REVIEW.md copied-name rule)", () => {
  // Each pair: the web type, the API class, and the snake_case fields the API declares.
  const pairs = [
    ["RecommendationConstituent", "RecommendationConstituent", ["price_brl", "quantity", "amount_brl", "effective_portfolio_weight"]],
    ["AllocationClass", "AllocationClass", ["cash_residual_brl", "over_target_brl"]],
    ["DroppedConstituent", "DroppedConstituent", ["ticker", "sleeve_weight", "portfolio_weight", "target_amount_brl", "price_brl", "reasons"]],
    ["Recommendation", "RecommendationResponse", ["stocks_dropped", "fiis_dropped"]],
  ];
  for (const [webType, apiClass, fields] of pairs) {
    const web = typeBody(webType);
    const api = apiClassBody(apiClass);
    for (const field of fields) {
      assert.match(api, new RegExp(`^    ${field}:`, "m"), `${apiClass}.${field} is declared by the API`);
      assert.match(web, new RegExp(`^  ${toCamel(field)}\\??:`, "m"), `${webType}.${toCamel(field)} is typed in api-types.ts`);
    }
  }
});
