# Frontend shell verification

Build the web app with Clerk configured, then run:

```sh
npm run typecheck
npm run build
node --test tests/vertical-slice-smoke.test.mjs
```

The smoke test checks the generated Next.js route shell and protected
middleware boundary without adding a browser dependency. Playwright is not
installed in this repository, and Clerk development sessions require a
browser-backed dev-browser token, so the authenticated flow remains a manual
check below.

The API fixture test covers the complete persisted flow without manual
database edits:

```sh
PYTHONPATH=api:py api/.venv/bin/python -m unittest discover -s api/tests -q
```

## Manual T17 checklist

- [ ] Create a fresh Clerk account; signup reaches onboarding without database edits.
- [ ] Complete onboarding, recommendation, portfolio, and review in one session.
- [ ] Reload each completed route; profile, recommendation, portfolio, and review remain available.
- [ ] Sign out; protected routes redirect to login and do not show protected data.
- [ ] Use a second account; it cannot see the first account's profile, recommendation, portfolio, or review.
- [ ] Basic account shows Premium access as locked; changing client state does not unlock server behavior.
- [ ] Keyboard: tab through navigation/forms, radio groups, number inputs, buttons, and error states; visible focus remains present.
- [ ] Mobile: test 320px and 390px widths; no horizontal scrolling, clipped controls, or unreadable text.
- [ ] Unavailable states: missing profile, recommendation, portfolio, and API failure show clear guidance and retry/navigation actions.
