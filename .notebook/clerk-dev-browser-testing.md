# Clerk local testing gotcha

## Finding

With a Clerk development publishable key, headless requests without a Clerk
dev-browser token can enter the development handshake and return a `500` or
hang while proxying the local origin. This does not reproduce the signed-in
browser session used by the application.

## Project impact

- `web/middleware.ts` remains the protected-route boundary.
- `web/tests/vertical-slice-smoke.test.mjs` checks the generated route shell and
  middleware declaration without bypassing Clerk.
- The authenticated signup-to-review path requires a browser-backed Clerk
  session until a Playwright fixture is added.
