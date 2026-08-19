import { defineConfig } from '@playwright/test'

// PROJECT_PLAN.md §6/§7 Phase 3: browser-measured p95 time-to-first-token, warm.
//
// Does NOT start its own dev server (`webServer` is deliberately unset) -- this is a
// perf harness you run against servers you already have up (`npm run dev` / the API),
// not a CI smoke test, since a real run performs a real PDF ingest and real Gemini
// calls against whatever GEMINI_API_KEY the API process holds.
export default defineConfig({
  testDir: './perf',
  timeout: 120_000,
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: process.env.VITE_APP_URL ?? 'http://localhost:5173',
    trace: 'off',
  },
})
