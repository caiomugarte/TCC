import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const manifestPath = new URL("../.next/server/app-paths-manifest.json", import.meta.url);

test("built app contains the complete route shell", async () => {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const routes = Object.keys(manifest);

  for (const route of [
    "/(public)/page",
    "/(auth)/login/[[...login]]/page",
    "/(auth)/signup/[[...signup]]/page",
    "/(protected)/app/onboarding/page",
    "/(protected)/app/recommendation/page",
    "/(protected)/app/portfolio/page",
    "/(protected)/app/review/page",
    "/(protected)/account/page",
  ]) {
    assert.ok(routes.includes(route), `missing built route: ${route}`);
  }
});

test("protected route boundary remains declared in middleware", async () => {
  const middleware = await readFile(new URL("../middleware.ts", import.meta.url), "utf8");

  assert.match(middleware, /createRouteMatcher\(\["\/app\(\.\*\)", "\/account\(\.\*\)"\]\)/);
  assert.match(middleware, /redirectToSignIn\(\{ returnBackUrl: request\.url \}\)/);
});
