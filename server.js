const express = require("express");
const crypto = require("crypto");

const app = express();
app.use(express.json({ limit: "10mb" }));

const PORT = process.env.PORT || 3000;

// In-memory state.
// On Render free instances this resets after restart/redeploy.
const runs = new Map();

function utf8Compare(a, b) {
  return Buffer.compare(Buffer.from(a, "utf8"), Buffer.from(b, "utf8"));
}

function sortUtf8(arr) {
  return [...arr].sort(utf8Compare);
}

function isSafeNonNegativeInt(v) {
  return Number.isSafeInteger(v) && v >= 0;
}

function isFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function validTimestamp(s) {
  if (typeof s !== "string") return false;
  const re =
    /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$/;
  if (!re.test(s)) return false;
  return !Number.isNaN(Date.parse(s));
}

function compactDigest(trainRowIds, evalRowIds, featureNames) {
  const obj = { trainRowIds, evalRowIds, featureNames };
  const json = JSON.stringify(obj);
  return crypto.createHash("sha256").update(json).digest("hex");
}

function invalidSelection(runId) {
  return {
    runId,
    selectedTrialId: null,
    trainRowIds: [],
    evalRowIds: [],
    featureNames: [],
    datasetDigest: null,
    reasonCodes: ["INVALID_INPUT"]
  };
}

function validateSelection(body) {
  if (!body || body.phase !== "select") return false;

  if (
    typeof body.runId !== "string" ||
    body.runId.length === 0 ||
    body.runId.length > 128
  )
    return false;

  if (!Array.isArray(body.forbiddenFeatures)) return false;

  if (
    !Number.isSafeInteger(body.numTrialsLimit) ||
    body.numTrialsLimit <= 0
  )
    return false;

  if (!Array.isArray(body.rows) || body.rows.length === 0) return false;
  if (!Array.isArray(body.trials)) return false;

  const rowIds = new Set();
  for (const r of body.rows) {
    if (!r || typeof r.id !== "string" || typeof r.entity !== "string")
      return false;
    if (rowIds.has(r.id)) return false;
    rowIds.add(r.id);

    if (!validTimestamp(r.eventTime) || !validTimestamp(r.predictionTime))
      return false;

    if (!isSafeNonNegativeInt(r.version)) return false;

    if (r.split !== "TRAIN" && r.split !== "EVAL") return false;

    if (!r.features || typeof r.features !== "object") return false;

    for (const f of Object.values(r.features)) {
      if (!f || typeof f !== "object") return false;
      if (!validTimestamp(f.availableAt)) return false;
    }
  }

  const trialIds = new Set();
  for (const t of body.trials) {
    if (!t || !isSafeNonNegativeInt(t.trialId)) return false;
    if (trialIds.has(t.trialId)) return false;
    trialIds.add(t.trialId);

    if (t.status !== "SUCCEEDED" && t.status !== "FAILED") return false;
  }

  return true;
}

function select(body) {
  const runId = body.runId;

  if (!validateSelection(body)) {
    return invalidSelection(runId);
  }

  if (body.trials.length > body.numTrialsLimit) {
    return {
      runId,
      selectedTrialId: null,
      trainRowIds: [],
      evalRowIds: [],
      featureNames: [],
      datasetDigest: null,
      reasonCodes: ["TRIAL_LIMIT_EXCEEDED"]
    };
  }

  const dedup = new Map();

  for (const r of body.rows) {
    const key = `${r.entity}\u0000${new Date(r.eventTime).toISOString()}`;
    const old = dedup.get(key);

    if (
      !old ||
      r.version > old.version ||
      (r.version === old.version && utf8Compare(r.id, old.id) < 0)
    ) {
      dedup.set(key, r);
    }
  }

  const retained = [...dedup.values()];

  let eligibleFeatures = null;

  for (const row of retained) {
    const names = Object.keys(row.features);

    const good = new Set(
      names.filter(
        (name) =>
          !body.forbiddenFeatures.includes(name) &&
          new Date(row.features[name].availableAt) <=
            new Date(row.predictionTime)
      )
    );

    if (eligibleFeatures === null) {
      eligibleFeatures = good;
    } else {
      eligibleFeatures = new Set(
        [...eligibleFeatures].filter((x) => good.has(x))
      );
    }
  }

  const featureNames = sortUtf8([...eligibleFeatures]);

  const trainRowIds = sortUtf8(
    retained.filter((r) => r.split === "TRAIN").map((r) => r.id)
  );

  const evalRowIds = sortUtf8(
    retained.filter((r) => r.split === "EVAL").map((r) => r.id)
  );

  const succeeded = body.trials.filter(
    (t) => t.status === "SUCCEEDED" && isFiniteNumber(t.evalMetric)
  );

  if (succeeded.length === 0) {
    return {
      runId,
      selectedTrialId: null,
      trainRowIds,
      evalRowIds,
      featureNames,
      datasetDigest: null,
      reasonCodes: ["NO_SUCCESSFUL_TRIAL"]
    };
  }

  succeeded.sort((a, b) => {
    if (a.evalMetric !== b.evalMetric)
      return b.evalMetric - a.evalMetric;
    return a.trialId - b.trialId;
  });

  const selectedTrialId = succeeded[0].trialId;
  const datasetDigest = compactDigest(
    trainRowIds,
    evalRowIds,
    featureNames
  );

  return {
    runId,
    selectedTrialId,
    trainRowIds,
    evalRowIds,
    featureNames,
    datasetDigest,
    reasonCodes: []
  };
}

function selectionFingerprint(body) {
  return crypto
    .createHash("sha256")
    .update(JSON.stringify(body))
    .digest("hex");
}

function evaluate(body) {
  const runId = body?.runId;

  const codes = [];

  const basicValid =
    body &&
    body.phase === "evaluate" &&
    typeof body.runId === "string" &&
    body.runId.length > 0 &&
    body.runId.length <= 128 &&
    isSafeNonNegativeInt(body.selectedTrialId) &&
    typeof body.datasetDigest === "string" &&
    /^[0-9a-f]{64}$/.test(body.datasetDigest) &&
    isFiniteNumber(body.metricFloor) &&
    body.metricFloor >= 0 &&
    body.metricFloor <= 1 &&
    body.requiredSlices &&
    typeof body.requiredSlices === "object" &&
    Array.isArray(body.rows) &&
    isSafeNonNegativeInt(body.bytesProcessed) &&
    isSafeNonNegativeInt(body.maxBytes);

  if (!basicValid) {
    codes.push("INVALID_INPUT");
  }

  if (basicValid) {
    for (const v of Object.values(body.requiredSlices)) {
      if (!isFiniteNumber(v) || v < 0 || v > 1) {
        codes.push("INVALID_INPUT");
        break;
      }
    }
  }

  const stored = runs.get(runId);

  if (
    !stored ||
    stored.response.selectedTrialId === null ||
    stored.response.selectedTrialId !== body?.selectedTrialId ||
    stored.response.datasetDigest !== body?.datasetDigest
  ) {
    codes.push("INVALID_LINEAGE");
  }

  let invalidRow = false;

  if (Array.isArray(body?.rows)) {
    for (const r of body.rows) {
      if (
        !r ||
        !Number.isInteger(r.label) ||
        !Number.isInteger(r.prediction) ||
        (r.label !== 0 && r.label !== 1) ||
        (r.prediction !== 0 && r.prediction !== 1) ||
        typeof r.slice !== "string" ||
        r.slice.length === 0
      ) {
        invalidRow = true;
        break;
      }
    }
  }

  if (invalidRow) codes.push("INVALID_TEST_ROW");

  let testMetric = null;
  let criticalSlicePass = false;

  if (
    basicValid &&
    !invalidRow &&
    body.rows.length > 0
  ) {
    const correct = body.rows.filter(
      (r) => r.label === r.prediction
    ).length;

    testMetric = Number(
      (correct / body.rows.length).toFixed(12)
    );

    const slices = new Map();

    for (const r of body.rows) {
      if (!slices.has(r.slice)) slices.set(r.slice, []);
      slices.get(r.slice).push(r);
    }

    criticalSlicePass = true;

    for (const [name, floor] of Object.entries(
      body.requiredSlices
    )) {
      const rows = slices.get(name);

      if (!rows) {
        codes.push(`MISSING_SLICE:${name}`);
        criticalSlicePass = false;
        continue;
      }

      const acc = Number(
        (
          rows.filter((r) => r.label === r.prediction).length /
          rows.length
        ).toFixed(12)
      );

      if (acc < floor) {
        codes.push(`SLICE_FLOOR:${name}`);
        criticalSlicePass = false;
      }
    }

    if (testMetric < body.metricFloor) {
      codes.push("AGGREGATE_FLOOR");
    }
  }

  if (basicValid && body.bytesProcessed > body.maxBytes) {
    codes.push("BYTE_LIMIT");
  }

  const reasonCodes = sortUtf8([...new Set(codes)]);

  const decision =
    reasonCodes.length === 0 ? "admit" : "reject";

  return {
    runId,
    selectedTrialId: body?.selectedTrialId ?? null,
    datasetDigest: body?.datasetDigest ?? null,
    testMetric,
    criticalSlicePass,
    decision,
    bytesProcessed: body?.bytesProcessed ?? null,
    reasonCodes
  };
}

app.post("/bqml", (req, res) => {
  const body = req.body;

  if (!body || (body.phase !== "select" && body.phase !== "evaluate")) {
    return res.status(400).json({ error: "INVALID_INPUT" });
  }

  if (body.phase === "select") {
    const fp = selectionFingerprint(body);
    const existing = runs.get(body.runId);

    if (existing) {
      if (existing.fingerprint === fp) {
        return res.json(existing.response);
      }
      return res.status(409).json({ error: "RUN_ID_CONFLICT" });
    }

    const response = select(body);

    runs.set(body.runId, {
      fingerprint: fp,
      response
    });

    return res.json(response);
  }

  return res.json(evaluate(body));
});

app.listen(PORT, () => {
  console.log(`BQML gate listening on port ${PORT}`);
});