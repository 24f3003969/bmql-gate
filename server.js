const express = require("express");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(express.json({ limit: "10mb" }));

const PORT = process.env.PORT || 3000;

const DB_FILE = path.join(__dirname, "storage.json");

function loadDb() {
    if (!fs.existsSync(DB_FILE)) {
        return { selections: {} };
    }

    try {
        return JSON.parse(fs.readFileSync(DB_FILE, "utf8"));
    } catch {
        return { selections: {} };
    }
}

function saveDb(db) {
    fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2));
}

const db = loadDb();

function utf8Compare(a, b) {
    return Buffer.compare(Buffer.from(a, "utf8"), Buffer.from(b, "utf8"));
}

function uniqueSorted(arr) {
    return [...new Set(arr)].sort(utf8Compare);
}

function sha256(text) {
    return crypto.createHash("sha256").update(text).digest("hex");
}

function round12(x) {
    return Number(x.toFixed(12));
}

function isSafeNonNegativeInteger(v) {
    return Number.isSafeInteger(v) && v >= 0;
}

function isFiniteNumber(v) {
    return typeof v === "number" && Number.isFinite(v);
}

function validTimestamp(ts) {
    if (typeof ts !== "string") return false;

    const regex =
        /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+\-]\d{2}:\d{2})$/;

    if (!regex.test(ts)) return false;

    const d = new Date(ts);

    return !Number.isNaN(d.getTime());
}

function compactJson(obj) {
    return JSON.stringify(obj);
}

function invalidSelectionResponse(runId) {
    return {
        runId: runId || "",
        selectedTrialId: null,
        trainRowIds: [],
        evalRowIds: [],
        featureNames: [],
        datasetDigest: null,
        reasonCodes: ["INVALID_INPUT"]
    };
}

function sortCodes(codes) {
    return uniqueSorted(codes);
}

function validateSelectionPayload(body) {
    try {
        if (body.phase !== "select") return false;

        if (
            typeof body.runId !== "string" ||
            body.runId.length === 0 ||
            body.runId.length > 128
        ) {
            return false;
        }

        if (!Array.isArray(body.rows) || body.rows.length === 0) {
            return false;
        }

        if (!Array.isArray(body.trials)) {
            return false;
        }

        if (
            !Number.isInteger(body.numTrialsLimit) ||
            body.numTrialsLimit <= 0
        ) {
            return false;
        }

        if (!Array.isArray(body.forbiddenFeatures)) {
            return false;
        }

        return true;
    } catch {
        return false;
    }
}

function canonicalSelectionInput(body) {
    return JSON.stringify(body);
}

function processSelection(body) {
    const result = {
        runId: body.runId,
        selectedTrialId: null,
        trainRowIds: [],
        evalRowIds: [],
        featureNames: [],
        datasetDigest: null,
        reasonCodes: []
    };

    const reasons = [];

    if (!validateSelectionPayload(body)) {
        result.reasonCodes = ["INVALID_INPUT"];
        return result;
    }

    if (body.trials.length > body.numTrialsLimit) {
        reasons.push("TRIAL_LIMIT_EXCEEDED");
    }

    const dedupMap = new Map();

    for (const row of body.rows) {
        try {
            if (
                typeof row.id !== "string" ||
                typeof row.entity !== "string" ||
                !validTimestamp(row.eventTime) ||
                !validTimestamp(row.predictionTime) ||
                !isSafeNonNegativeInteger(row.version) ||
                !["TRAIN", "EVAL"].includes(row.split) ||
                typeof row.features !== "object" ||
                row.features === null
            ) {
                reasons.push("INVALID_INPUT");
                continue;
            }

            const utcEvent = new Date(row.eventTime).toISOString();

            const key = `${row.entity}||${utcEvent}`;

            const existing = dedupMap.get(key);

            if (!existing) {
                dedupMap.set(key, row);
            } else {
                if (row.version > existing.version) {
                    dedupMap.set(key, row);
                } else if (row.version === existing.version) {
                    if (utf8Compare(row.id, existing.id) < 0) {
                        dedupMap.set(key, row);
                    }
                }
            }
        } catch {
            reasons.push("INVALID_INPUT");
        }
    }

    const retainedRows = [...dedupMap.values()];

    if (retainedRows.length === 0) {
        reasons.push("INVALID_INPUT");
    }

    let featureNames = [];

    if (retainedRows.length > 0) {
        let common = null;

        for (const row of retainedRows) {
            const predictionTime = new Date(row.predictionTime).getTime();

            const eligible = new Set();

            for (const [name, feature] of Object.entries(row.features)) {
                if (body.forbiddenFeatures.includes(name)) continue;

                if (
                    !feature ||
                    typeof feature !== "object" ||
                    !validTimestamp(feature.availableAt)
                ) {
                    continue;
                }

                if (
                    new Date(feature.availableAt).getTime() <= predictionTime
                ) {
                    eligible.add(name);
                }
            }

            if (common === null) {
                common = eligible;
            } else {
                common = new Set(
                    [...common].filter(x => eligible.has(x))
                );
            }
        }

        featureNames = [...(common || [])].sort(utf8Compare);
    }

    const trainRowIds = retainedRows
        .filter(r => r.split === "TRAIN")
        .map(r => r.id)
        .sort(utf8Compare);

    const evalRowIds = retainedRows
        .filter(r => r.split === "EVAL")
        .map(r => r.id)
        .sort(utf8Compare);

    result.trainRowIds = trainRowIds;
    result.evalRowIds = evalRowIds;
    result.featureNames = featureNames;

    if (!reasons.includes("INVALID_INPUT")) {
        const digestInput = {
            trainRowIds,
            evalRowIds,
            featureNames
        };

        result.datasetDigest = sha256(
            compactJson(digestInput)
        );
    }

    const eligibleTrials = body.trials.filter(
        t =>
            t.status === "SUCCEEDED" &&
            isFiniteNumber(t.evalMetric)
    );

    if (
        !reasons.includes("TRIAL_LIMIT_EXCEEDED") &&
        !reasons.includes("INVALID_INPUT")
    ) {
        if (eligibleTrials.length === 0) {
            reasons.push("NO_SUCCESSFUL_TRIAL");
        } else {
            eligibleTrials.sort((a, b) => {
                if (b.evalMetric !== a.evalMetric) {
                    return b.evalMetric - a.evalMetric;
                }

                return a.trialId - b.trialId;
            });

            result.selectedTrialId =
                eligibleTrials[0].trialId;
        }
    }

    if (reasons.length > 0) {
        result.selectedTrialId = null;

        if (reasons.includes("INVALID_INPUT")) {
            result.datasetDigest = null;
        }
    }

    result.reasonCodes = sortCodes(reasons);

    return result;
}

function validateDigest(d) {
    return /^[a-f0-9]{64}$/.test(d);
}

function processEvaluation(body) {
    const reasons = [];

    let invalidRowFound = false;

    const output = {
        runId: body.runId,
        selectedTrialId: body.selectedTrialId,
        datasetDigest: body.datasetDigest,
        testMetric: null,
        criticalSlicePass: false,
        decision: "reject",
        bytesProcessed: body.bytesProcessed,
        reasonCodes: []
    };

    let validInput = true;

    if (
        typeof body.runId !== "string" ||
        !isSafeNonNegativeInteger(body.selectedTrialId) ||
        !validateDigest(body.datasetDigest) ||
        !Array.isArray(body.rows)
    ) {
        validInput = false;
    }

    if (
        !isFiniteNumber(body.metricFloor) ||
        body.metricFloor < 0 ||
        body.metricFloor > 1
    ) {
        validInput = false;
    }

    if (
        !isSafeNonNegativeInteger(body.bytesProcessed) ||
        !isSafeNonNegativeInteger(body.maxBytes)
    ) {
        validInput = false;
    }

    if (!validInput) {
        reasons.push("INVALID_INPUT");
    }

    const stored = db.selections[body.runId];

    if (
        !stored ||
        stored.response.selectedTrialId !== body.selectedTrialId ||
        stored.response.datasetDigest !== body.datasetDigest
    ) {
        reasons.push("INVALID_LINEAGE");
    }

    for (const row of body.rows || []) {
        const ok =
            (row.label === 0 || row.label === 1) &&
            (row.prediction === 0 || row.prediction === 1) &&
            typeof row.slice === "string" &&
            row.slice.length > 0;

        if (!ok) {
            invalidRowFound = true;
            break;
        }
    }

    if (invalidRowFound) {
        reasons.push("INVALID_TEST_ROW");
    }

    if (body.bytesProcessed > body.maxBytes) {
        reasons.push("BYTE_LIMIT");
    }

    let criticalSlicePass = true;

    if (
        body.rows.length === 0 ||
        invalidRowFound
    ) {
        criticalSlicePass = false;
    } else {
        const totalCorrect = body.rows.filter(
            r => r.label === r.prediction
        ).length;

        const aggregate =
            totalCorrect / body.rows.length;

        output.testMetric = round12(aggregate);

        if (aggregate < body.metricFloor) {
            reasons.push("AGGREGATE_FLOOR");
        }

        const requiredSlices =
            body.requiredSlices || {};

        for (const [slice, floor] of Object.entries(
            requiredSlices
        )) {
            const rows = body.rows.filter(
                r => r.slice === slice
            );

            if (rows.length === 0) {
                reasons.push(`MISSING_SLICE:${slice}`);
                criticalSlicePass = false;
                continue;
            }

            const correct = rows.filter(
                r => r.label === r.prediction
            ).length;

            const acc = round12(correct / rows.length);

            if (acc < floor) {
                reasons.push(`SLICE_FLOOR:${slice}`);
                criticalSlicePass = false;
            }
        }
    }

    if (
        reasons.includes("INVALID_INPUT") ||
        reasons.includes("INVALID_LINEAGE") ||
        reasons.includes("INVALID_TEST_ROW")
    ) {
        criticalSlicePass = false;
    }

    output.criticalSlicePass = criticalSlicePass;

    output.reasonCodes = sortCodes(reasons);

    if (output.reasonCodes.length === 0) {
        output.decision = "admit";
    }

    return output;
}

app.post("/bqml", (req, res) => {
    const body = req.body;

    if (!body || !body.phase) {
        return res
            .status(400)
            .json({ error: "INVALID_INPUT" });
    }

    if (body.phase === "select") {
        const runId = body.runId;

        const existing = db.selections[runId];

        if (existing) {
            const incoming =
                canonicalSelectionInput(body);

            if (
                incoming === existing.originalInput
            ) {
                return res.json(existing.response);
            }

            return res
                .status(409)
                .json({
                    error: "RUN_ID_CONFLICT"
                });
        }

        const response =
            processSelection(body);

        db.selections[runId] = {
            originalInput:
                canonicalSelectionInput(body),
            response
        };

        saveDb(db);

        return res.json(response);
    }

    if (body.phase === "evaluate") {
        const response =
            processEvaluation(body);

        return res.json(response);
    }

    return res
        .status(400)
        .json({ error: "INVALID_INPUT" });
});

app.listen(PORT, () => {
    console.log(
        `Server listening on ${PORT}`
    );
});