// Phase 5 / stretch k6 load test (PROJECT_PLAN.md §7 Phase 5 "a load test", §13 stretch
// "k6 load test across concurrent streaming sessions") -- run against the LOCAL
// docker-compose stack (http://localhost:8000), not production, per explicit direction:
// production is currently unhealthy (see METRICS.md Phase 4 section) and this laptop's
// resources are NOT constrained to Render's 0.1 CPU / 512MB, so treat every number this
// produces as a local-machine concurrency/correctness result, not a Render capacity claim.
//
// IMPORTANT -- this is the one Phase 5 deliverable that is NOT free: /query's live path
// (api/routes_query.py) calls Gemini directly and does NOT go through eval/cache.py, so
// every completed VU iteration is a real API call against the SAME key GEMINI_API_KEY the
// local docker-compose api container is configured with. gemini-3.5-flash-lite's free tier
// is RPM 15 / RPD 500 (PROJECT_PLAN.md §5) -- default VUS/DURATION below are kept small on
// purpose to stay well under that, and load_test summary will show `rate_limited` responses
// explicitly if you push past it (that's a legitimate result to record, not a bug).
//
// Usage:
//   k6 run scripts/load_test.js
//   k6 run --env VUS=5 --env DURATION=30s scripts/load_test.js
//
// Requires: local docker-compose stack up (docker compose up -d) with GEMINI_API_KEY set
// for the api service, and network access to it at BASE_URL (default localhost:8000).

import http from "k6/http";
import { check, sleep } from "k6";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const VUS = Number(__ENV.VUS || 3);
const RAMP_DURATION = __ENV.RAMP_DURATION || "10s";
const HOLD_DURATION = __ENV.DURATION || "20s";

// k6's `open()` only works in the init context (module scope), not inside functions --
// smallest PDF in eval/pdfs/, same one used for the earlier live-recovery test, chosen to
// keep setup() (which ingests it fresh) fast.
const PDF_BYTES = open("../eval/pdfs/ULTABEAUTY_2023Q4_EARNINGS.pdf", "b");
const PDF_NAME = "ULTABEAUTY_2023Q4_EARNINGS.pdf";

const SAMPLE_QUESTIONS = [
  "What was total net sales for the period covered by this filing?",
  "What was gross margin for the period covered by this filing?",
  "What were selling, general and administrative expenses for the period?",
];

export const options = {
  scenarios: {
    ramping_load: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: RAMP_DURATION, target: VUS },
        { duration: HOLD_DURATION, target: VUS },
        { duration: "5s", target: 0 },
      ],
    },
  },
  setupTimeout: "120s",
};

function headerCI(res, name) {
  const lower = name.toLowerCase();
  for (const key in res.headers) {
    if (key.toLowerCase() === lower) return res.headers[key];
  }
  return undefined;
}

export function setup() {
  const email = `k6-load-test-${Date.now()}@example.com`;
  const password = "k6-load-test-password-does-not-matter";

  const registerRes = http.post(
    `${BASE_URL}/auth/register`,
    JSON.stringify({ email, password }),
    { headers: { "Content-Type": "application/json" } }
  );
  if (registerRes.status !== 201) {
    throw new Error(`setup: register failed with ${registerRes.status}: ${registerRes.body}`);
  }
  const csrfToken = headerCI(registerRes, "X-Csrf-Token");
  const sessionCookie = registerRes.cookies["chatdoc_session"][0].value;
  const csrfCookie = registerRes.cookies["chatdoc_csrf"][0].value;
  const cookieHeader = `chatdoc_session=${sessionCookie}; chatdoc_csrf=${csrfCookie}`;

  const authHeaders = {
    Cookie: cookieHeader,
    "x-csrf-token": csrfToken,
  };

  const uploadRes = http.post(
    `${BASE_URL}/documents`,
    { file: http.file(PDF_BYTES, PDF_NAME, "application/pdf") },
    { headers: authHeaders }
  );
  if (uploadRes.status !== 202) {
    throw new Error(`setup: upload failed with ${uploadRes.status}: ${uploadRes.body}`);
  }
  const documentId = JSON.parse(uploadRes.body).id;

  // Poll until the freshly-uploaded document is ready -- k6's default setupTimeout (120s
  // above) bounds this; a small ~100KB PDF is typically ready in well under a minute
  // locally (see METRICS.md Phase 5 stage-latency numbers for what to expect).
  let ready = false;
  for (let i = 0; i < 60; i++) {
    const docRes = http.get(`${BASE_URL}/documents/${documentId}`, { headers: authHeaders });
    const doc = JSON.parse(docRes.body);
    if (doc.status === "ready") {
      ready = true;
      break;
    }
    if (doc.status === "failed") {
      throw new Error(`setup: document ingest failed: ${doc.error}`);
    }
    sleep(1);
  }
  if (!ready) {
    throw new Error("setup: document did not become ready within the poll budget");
  }

  return { cookieHeader, csrfToken, documentId };
}

export default function (data) {
  const question = SAMPLE_QUESTIONS[Math.floor(Math.random() * SAMPLE_QUESTIONS.length)];
  const res = http.post(
    `${BASE_URL}/query`,
    JSON.stringify({ document_id: data.documentId, question }),
    {
      headers: {
        "Content-Type": "application/json",
        Cookie: data.cookieHeader,
        "x-csrf-token": data.csrfToken,
      },
      timeout: "60s",
    }
  );

  check(res, {
    "status is 200": (r) => r.status === 200,
    "stream reached terminal done event": (r) => r.body && r.body.includes("event: done"),
    "not rate limited": (r) => !(r.body && r.body.includes("rate limited")),
  });

  // Space out each VU's iterations rather than hammering back-to-back -- keeps total
  // request volume predictable against the RPM budget described in the module docstring.
  sleep(3);
}

export function teardown(data) {
  // Best-effort cleanup of the throwaway document/user this test created; a failure here
  // shouldn't fail the load test itself.
  http.del(`${BASE_URL}/documents/${data.documentId}`, null, {
    headers: { Cookie: data.cookieHeader, "x-csrf-token": data.csrfToken },
  });
}
