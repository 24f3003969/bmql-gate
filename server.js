const express = require("express");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const app = express();

app.use(express.json({ limit: "10mb" }));

const PORT = process.env.PORT || 3000;
const DB_FILE = path.join(__dirname, "storage.json");

/* =========================================================
   Persistent state
   ========================================================= */

function loadDb() {
    if (!fs.existsSync(DB_FILE)) {
        return { selections: {} };
    }

    try {
        const parsed = JSON.parse(
            fs.readFileSync(DB_FILE, "utf8")
        );

        if (
            !parsed ||
            typeof parsed !== "object" ||
            !parsed.selections ||
            typeof parsed.selections !== "object"
        ) {
            return { selections: {} };
        }

        return parsed;
    } catch {
        return { selections: {} };
    }
}

function saveDb(db) {
    fs.writeFileSync(
        DB_FILE,
        JSON.stringify(db, null, 2),
        "utf8"
    );
}

const db = loadDb();

/* =========================================================
   General helpers
   ========================================================= */

function isPlainObject(value) {
    return (
        value !== null &&
        typeof value === "object" &&
        !Array.isArray(value)
    );
}

function isSafeNonNegativeInteger(value) {
    return (
        typeof value === "number" &&
        Number.isSafeInteger(value) &&
        value >= 0
    );
}

function isFiniteNumber(value) {
    return (
        typeof value === "number" &&
        Number.isFinite(value)
    );
}

function utf8Compare(a, b) {
    return Buffer.compare(
        Buffer.from(a, "utf8"),
        Buffer.from(b, "utf8")
    );
}

function sortUtf8(values) {
    return [...values].sort(utf8Compare);
}

function sortAndDeduplicateCodes(codes) {
    return [...new Set(codes)].sort(utf8Compare);
}

function sha256(value) {
    return crypto
        .createHash("sha256")
        .update(value, "utf8")
        .digest("hex");
}

function round12(value) {
    return Number(value.toFixed(12));
}

/* =========================================================
   Strict timestamp validation
   YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)
   ========================================================= */

function parseTimestamp(value) {
    if (typeof value !== "string") {
        return null;
    }

    const match =
        /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$/
            .exec(value);

    if (!match) {
        return null;
    }

    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    const hour = Number(match[4]);
    const minute = Number(match[5]);
    const second = Number(match[6]);

    const fraction = match[7] || "";
    const milliseconds = Number(
        fraction.padEnd(3, "0") || "0"
    );

    if (month < 1 || month > 12) {
        return null;
    }

    if (hour > 23 || minute > 59 || second > 59) {
        return null;
    }

    if (milliseconds > 999) {
        return null;
    }

    const daysInMonth = new Date(
        Date.UTC(year, month, 0)
    ).getUTCDate();

    if (day < 1 || day > daysInMonth) {
        return null;
    }

    const offset = match[8];

    if (offset !== "Z") {
        const offsetMatch =
            /^([+-])(\d{2}):(\d{2})$/.exec(offset);

        if (!offsetMatch) {
            return null;
        }

        const offsetHours = Number(offsetMatch[2]);
        const offsetMinutes = Number(offsetMatch[3]);

        if (
            offsetHours > 23 ||
            offsetMinutes > 59
        ) {
            return null;
        }
    }

    const timestamp = Date.parse(value);

    if (!Number.isFinite(timestamp)) {
        return null;
    }

    return timestamp;
}

/* =========================================================
   Selection row validation
   ========================================================= */

function validateSelectionRow(row) {
    if (!isPlainObject(row)) {
        return false;
    }

    if (typeof row.id !== "string" || row.id.length === 0) {
        return false;
    }

    if (
        typeof row.entity !== "string" ||
        row.entity.length === 0
    ) {
        return false;
    }

    if (parseTimestamp(row.eventTime) === null) {
        return false;
    }

    if (parseTimestamp(row.predictionTime) === null) {
        return false;
    }

    if (!isSafeNonNegativeInteger(row.version)) {
        return false;
    }

    if (
        row.split !== "TRAIN" &&
        row.split !== "EVAL"
    ) {
        return false;
    }

    if (!isPlainObject(row.features)) {
        return false;
    }

    /*
     * Every feature supplied by the row must have valid
     * feature metadata. We do NOT silently discard malformed
     * feature metadata because that would allow malformed
     * point-in-time data to sneak through.
     */
    for (const [featureName, feature] of Object.entries(
        row.features
    )) {
        if (
            typeof featureName !== "string" ||
            featureName.length === 0
        ) {
            return false;
        }

        if (!isPlainObject(feature)) {
            return false;
        }

        /*
         * "value" is data. We deliberately do not impose
         * restrictions on its contents.
         */
        if (!Object.prototype.hasOwnProperty.call(
            feature,
            "value"
        )) {
            return false;
        }

        if (parseTimestamp(feature.availableAt) === null) {
            return false;
        }
    }

    return true;
}

/* =========================================================
   Trial validation
   ========================================================= */

function validateTrial(trial) {
    if (!isPlainObject(trial)) {
        return false;
    }

    if (!isSafeNonNegativeInteger(trial.trialId)) {
        return false;
    }

    if (
        trial.status !== "SUCCEEDED" &&
        trial.status !== "FAILED"
    ) {
        return false;
    }

    /*
     * evalMetric is relevant for successful trials.
     * Validate it when present so malformed trial objects
     * cannot influence selection.
     */
    if (
        Object.prototype.hasOwnProperty.call(
            trial,
            "evalMetric"
        )
    ) {
        if (!isFiniteNumber(trial.evalMetric)) {
            return false;
        }
    }

    if (
        trial.status === "SUCCEEDED" &&
        !isFiniteNumber(trial.evalMetric)
    ) {
        return false;
    }

    return true;
}

/* =========================================================
   Selection input validation
   ========================================================= */

function validateSelectionInput(body) {
    if (!isPlainObject(body)) {
        return false;
    }

    if (body.phase !== "select") {
        return false;
    }

    if (
        typeof body.runId !== "string" ||
        body.runId.length === 0 ||
        body.runId.length > 128
    ) {
        return false;
    }

    if (
        !Array.isArray(body.forbiddenFeatures)
    ) {
        return false;
    }

    for (const name of body.forbiddenFeatures) {
        if (
            typeof name !== "string" ||
            name.length === 0
        ) {
            return false;
        }
    }

    if (
        !isSafeNonNegativeInteger(
            body.numTrialsLimit
        ) ||
        body.numTrialsLimit <= 0
    ) {
        return false;
    }

    if (
        !Array.isArray(body.rows) ||
        body.rows.length === 0
    ) {
        return false;
    }

    if (!Array.isArray(body.trials)) {
        return false;
    }

    /*
     * Validate ALL rows before deduplication.
     */
    const rowIds = new Set();

    for (const row of body.rows) {
        if (!validateSelectionRow(row)) {
            return false;
        }

        if (rowIds.has(row.id)) {
            return false;
        }

        rowIds.add(row.id);
    }

    /*
     * Validate ALL trial IDs and trial structures.
     */
    const trialIds = new Set();

    for (const trial of body.trials) {
        if (!validateTrial(trial)) {
            return false;
        }

        if (trialIds.has(trial.trialId)) {
            return false;
        }

        trialIds.add(trial.trialId);
    }

    return true;
}

/* =========================================================
   Deduplication
   =========================================================

   Key:
       [entity, UTC(eventTime)]

   Winner:
       highest version
       then UTF-8-byte-smallest ID
   ========================================================= */

function deduplicateRows(rows) {
    const retained = new Map();

    for (const row of rows) {
        const eventTimeUtc =
            new Date(
                parseTimestamp(row.eventTime)
            ).toISOString();

        const key =
            row.entity + "\u0000" + eventTimeUtc;

        const previous = retained.get(key);

        if (!previous) {
            retained.set(key, row);
            continue;
        }

        if (row.version > previous.version) {
            retained.set(key, row);
            continue;
        }

        if (
            row.version === previous.version &&
            utf8Compare(row.id, previous.id) < 0
        ) {
            retained.set(key, row);
        }
    }

    return [...retained.values()];
}

/* =========================================================
   Point-in-time feature selection
   =========================================================

   A feature is eligible iff:

   1. It appears in EVERY retained row.
   2. It is not forbidden.
   3. availableAt <= predictionTime in EVERY retained row.
   ========================================================= */

function computeSharedFeatures(
    retainedRows,
    forbiddenFeatures
) {
    const forbidden = new Set(
        forbiddenFeatures
    );

    let commonFeatures = null;

    for (const row of retainedRows) {
        const predictionTime =
            parseTimestamp(row.predictionTime);

        const eligibleForThisRow = new Set();

        for (const [
            featureName,
            feature
        ] of Object.entries(row.features)) {
            if (forbidden.has(featureName)) {
                continue;
            }

            const availableAt =
                parseTimestamp(
                    feature.availableAt
                );

            if (
                availableAt !== null &&
                availableAt <= predictionTime
            ) {
                eligibleForThisRow.add(
                    featureName
                );
            }
        }

        if (commonFeatures === null) {
            commonFeatures =
                eligibleForThisRow;
        } else {
            commonFeatures = new Set(
                [...commonFeatures].filter(
                    name =>
                        eligibleForThisRow.has(name)
                )
            );
        }
    }

    return sortUtf8([
        ...(commonFeatures || [])
    ]);
}

/* =========================================================
   Dataset digest
   ========================================================= */

function createDatasetDigest(
    trainRowIds,
    evalRowIds,
    featureNames
) {
    /*
     * IMPORTANT:
     * Object insertion order here is intentional and exactly
     * matches the required shape:
     *
     * {trainRowIds,evalRowIds,featureNames}
     */
    const digestObject = {
        trainRowIds,
        evalRowIds,
        featureNames
    };

    const compact =
        JSON.stringify(digestObject);

    return sha256(compact);
}

/* =========================================================
   Selection
   ========================================================= */

function processSelection(body) {
    const output = {
        runId: body.runId,
        selectedTrialId: null,
        trainRowIds: [],
        evalRowIds: [],
        featureNames: [],
        datasetDigest: null,
        reasonCodes: []
    };

    /*
     * First: complete validation.
     *
     * This is intentionally BEFORE deduplication.
     */
    if (!validateSelectionInput(body)) {
        output.reasonCodes = [
            "INVALID_INPUT"
        ];

        return output;
    }

    const reasons = [];

    if (
        body.trials.length >
        body.numTrialsLimit
    ) {
        reasons.push(
            "TRIAL_LIMIT_EXCEEDED"
        );
    }

    /*
     * Second: deduplicate only validated rows.
     */
    const retainedRows =
        deduplicateRows(body.rows);

    /*
     * Third: calculate point-in-time shared features
     * only over retained rows.
     */
    const featureNames =
        computeSharedFeatures(
            retainedRows,
            body.forbiddenFeatures
        );

    /*
     * TRAIN/EVAL IDs are taken ONLY from retained rows.
     */
    const trainRowIds = sortUtf8(
        retainedRows
            .filter(row => row.split === "TRAIN")
            .map(row => row.id)
    );

    const evalRowIds = sortUtf8(
        retainedRows
            .filter(row => row.split === "EVAL")
            .map(row => row.id)
    );

    output.trainRowIds = trainRowIds;
    output.evalRowIds = evalRowIds;
    output.featureNames = featureNames;

    /*
     * Dataset digest is deterministic and only represents
     * the frozen selected dataset/features.
     */
    output.datasetDigest =
        createDatasetDigest(
            trainRowIds,
            evalRowIds,
            featureNames
        );

    /*
     * Only finite SUCCEEDED trials are eligible.
     */
    const successfulTrials =
        body.trials.filter(
            trial =>
                trial.status === "SUCCEEDED" &&
                isFiniteNumber(
                    trial.evalMetric
                )
        );

    if (successfulTrials.length === 0) {
        reasons.push(
            "NO_SUCCESSFUL_TRIAL"
        );
    } else if (
        !reasons.includes(
            "TRIAL_LIMIT_EXCEEDED"
        )
    ) {
        /*
         * Maximize metric.
         *
         * Exact metric ties:
         * smallest integer trialId.
         */
        successfulTrials.sort(
            (a, b) => {
                if (
                    a.evalMetric >
                    b.evalMetric
                ) {
                    return -1;
                }

                if (
                    a.evalMetric <
                    b.evalMetric
                ) {
                    return 1;
                }

                return (
                    a.trialId -
                    b.trialId
                );
            }
        );

        output.selectedTrialId =
            successfulTrials[0].trialId;
    }

    /*
     * Any selection failure => null selected trial.
     */
    if (reasons.length > 0) {
        output.selectedTrialId = null;
    }

    output.reasonCodes =
        sortAndDeduplicateCodes(
            reasons
        );

    return output;
}

/* =========================================================
   Evaluation validation
   ========================================================= */

function validateDigest(value) {
    return (
        typeof value === "string" &&
        /^[a-f0-9]{64}$/.test(value)
    );
}

function validateEvaluationRow(row) {
    if (!isPlainObject(row)) {
        return false;
    }

    if (
        !(
            row.label === 0 ||
            row.label === 1
        )
    ) {
        return false;
    }

    if (
        !(
            row.prediction === 0 ||
            row.prediction === 1
        )
    ) {
        return false;
    }

    if (
        typeof row.slice !== "string" ||
        row.slice.length === 0
    ) {
        return false;
    }

    return true;
}

function validateRequiredSlices(
    requiredSlices
) {
    if (
        !isPlainObject(requiredSlices)
    ) {
        return false;
    }

    for (const [
        slice,
        floor
    ] of Object.entries(
        requiredSlices
    )) {
        if (
            typeof slice !== "string" ||
            slice.length === 0
        ) {
            return false;
        }

        if (
            !isFiniteNumber(floor) ||
            floor < 0 ||
            floor > 1
        ) {
            return false;
        }
    }

    return true;
}

function validateEvaluationInput(body) {
    if (!isPlainObject(body)) {
        return false;
    }

    if (body.phase !== "evaluate") {
        return false;
    }

    if (
        typeof body.runId !== "string" ||
        body.runId.length === 0
    ) {
        return false;
    }

    if (
        !isSafeNonNegativeInteger(
            body.selectedTrialId
        )
    ) {
        return false;
    }

    if (
        !validateDigest(
            body.datasetDigest
        )
    ) {
        return false;
    }

    if (
        !isFiniteNumber(body.metricFloor) ||
        body.metricFloor < 0 ||
        body.metricFloor > 1
    ) {
        return false;
    }

    if (
        !validateRequiredSlices(
            body.requiredSlices
        )
    ) {
        return false;
    }

    if (!Array.isArray(body.rows)) {
        return false;
    }

    if (
        !isSafeNonNegativeInteger(
            body.bytesProcessed
        )
    ) {
        return false;
    }

    if (
        !isSafeNonNegativeInteger(
            body.maxBytes
        )
    ) {
        return false;
    }

    return true;
}

/* =========================================================
   Evaluation
   ========================================================= */

function processEvaluation(body) {
    const output = {
        runId: body.runId,
        selectedTrialId:
            body.selectedTrialId,
        datasetDigest:
            body.datasetDigest,
        testMetric: null,
        criticalSlicePass: false,
        decision: "reject",
        bytesProcessed:
            body.bytesProcessed,
        reasonCodes: []
    };

    const reasons = [];

    /*
     * Input validation.
     */
    if (!validateEvaluationInput(body)) {
        reasons.push("INVALID_INPUT");

        output.reasonCodes =
            sortAndDeduplicateCodes(
                reasons
            );

        return output;
    }

    /*
     * Frozen lineage validation.
     *
     * The selected trial and dataset digest must exactly
     * correspond to a stored successful selection.
     */
    const stored =
        db.selections[body.runId];

    if (
        !stored ||
        !stored.response ||
        stored.response.selectedTrialId === null ||
        stored.response.selectedTrialId !==
            body.selectedTrialId ||
        stored.response.datasetDigest !==
            body.datasetDigest
    ) {
        reasons.push(
            "INVALID_LINEAGE"
        );
    }

    /*
     * Validate every final-test row before any metric
     * computation.
     */
    let invalidTestRow = false;

    for (const row of body.rows) {
        if (!validateEvaluationRow(row)) {
            invalidTestRow = true;
            break;
        }
    }

    if (invalidTestRow) {
        reasons.push(
            "INVALID_TEST_ROW"
        );
    }

    /*
     * Byte gate is independent of aggregate/slice gates.
     */
    if (
        body.bytesProcessed >
        body.maxBytes
    ) {
        reasons.push("BYTE_LIMIT");
    }

    /*
     * Empty rows OR invalid rows:
     *
     * - testMetric = null
     * - skip aggregate checks
     * - skip slice checks
     * - lineage and byte checks still apply
     */
    if (
        body.rows.length === 0 ||
        invalidTestRow
    ) {
        output.testMetric = null;
        output.criticalSlicePass = false;

        output.reasonCodes =
            sortAndDeduplicateCodes(
                reasons
            );

        /*
         * No valid final-test evaluation can be admitted.
         */
        output.decision = "reject";

        return output;
    }

    /*
     * Aggregate accuracy.
     */
    const correct =
        body.rows.reduce(
            (count, row) =>
                count +
                (row.label === row.prediction
                    ? 1
                    : 0),
            0
        );

    const aggregate =
        correct / body.rows.length;

    output.testMetric =
        round12(aggregate);

    if (
        output.testMetric <
        body.metricFloor
    ) {
        reasons.push(
            "AGGREGATE_FLOOR"
        );
    }

    /*
     * Required slices.
     *
     * Every required slice must exist.
     * Every existing required slice must meet
     * its inclusive floor.
     */
    let allRequiredSlicesPass = true;

    for (const [
        sliceName,
        floor
    ] of Object.entries(
        body.requiredSlices
    )) {
        const sliceRows =
            body.rows.filter(
                row =>
                    row.slice ===
                    sliceName
            );

        if (sliceRows.length === 0) {
            reasons.push(
                `MISSING_SLICE:${sliceName}`
            );

            allRequiredSlicesPass = false;

            continue;
        }

        const sliceCorrect =
            sliceRows.reduce(
                (count, row) =>
                    count +
                    (row.label ===
                    row.prediction
                        ? 1
                        : 0),
                0
            );

        const sliceAccuracy =
            round12(
                sliceCorrect /
                    sliceRows.length
            );

        if (
            sliceAccuracy < floor
        ) {
            reasons.push(
                `SLICE_FLOOR:${sliceName}`
            );

            allRequiredSlicesPass =
                false;
        }
    }

    /*
     * criticalSlicePass is about the required-slice gate,
     * not aggregate accuracy or bytes.
     *
     * Therefore aggregate failure and byte failure do NOT
     * by themselves make criticalSlicePass false.
     */
    output.criticalSlicePass =
        allRequiredSlicesPass;

    /*
     * Admission requires ALL gates:
     *
     * - valid input
     * - valid lineage
     * - valid test rows
     * - aggregate floor
     * - every required slice
     * - every required slice floor
     * - byte limit
     */
    if (
        !reasons.includes(
            "INVALID_INPUT"
        ) &&
        !reasons.includes(
            "INVALID_LINEAGE"
        ) &&
        !reasons.includes(
            "INVALID_TEST_ROW"
        ) &&
        !reasons.includes(
            "AGGREGATE_FLOOR"
        ) &&
        !reasons.includes(
            "BYTE_LIMIT"
        ) &&
        allRequiredSlicesPass
    ) {
        output.decision = "admit";
    } else {
        output.decision = "reject";
    }

    output.reasonCodes =
        sortAndDeduplicateCodes(
            reasons
        );

    return output;
}

/* =========================================================
   POST /bqml
   ========================================================= */

app.post("/bqml", (req, res) => {
    const body = req.body;

    /*
     * Unknown/missing phase is specifically required
     * to return HTTP 400 and exactly this JSON.
     */
    if (
        !isPlainObject(body) ||
        typeof body.phase !== "string"
    ) {
        return res
            .status(400)
            .json({
                error: "INVALID_INPUT"
            });
    }

    /* -----------------------------------------------------
       SELECT
       ----------------------------------------------------- */

    if (body.phase === "select") {
        /*
         * runId must be a non-empty string <= 128 characters
         * before looking up persistent state.
         */
        const runId =
            typeof body.runId === "string"
                ? body.runId
                : null;

        if (runId !== null) {
            const existing =
                db.selections[runId];

            if (existing) {
                /*
                 * Identical replay:
                 * return the exact stored response.
                 */
                const incomingInput =
                    JSON.stringify(body);

                if (
                    incomingInput ===
                    existing.originalInput
                ) {
                    return res.json(
                        existing.response
                    );
                }

                /*
                 * Same runId + different selection input.
                 */
                return res
                    .status(409)
                    .json({
                        error:
                            "RUN_ID_CONFLICT"
                    });
            }
        }

        /*
         * Malformed selection still produces the required
         * response shape and is persisted under a valid runId.
         */
        if (
            !validateSelectionInput(body)
        ) {
            const response = {
                runId:
                    typeof body.runId ===
                    "string"
                        ? body.runId
                        : "",
                selectedTrialId: null,
                trainRowIds: [],
                evalRowIds: [],
                featureNames: [],
                datasetDigest: null,
                reasonCodes: [
                    "INVALID_INPUT"
                ]
            };

            /*
             * Persist only when runId itself is valid enough
             * to act as a state key.
             */
            if (
                typeof body.runId ===
                    "string" &&
                body.runId.length > 0 &&
                body.runId.length <= 128
            ) {
                db.selections[
                    body.runId
                ] = {
                    originalInput:
                        JSON.stringify(body),
                    response
                };

                saveDb(db);
            }

            return res.json(response);
        }

        const response =
            processSelection(body);

        /*
         * Complete selection response is frozen.
         */
        db.selections[
            body.runId
        ] = {
            originalInput:
                JSON.stringify(body),
            response
        };

        saveDb(db);

        return res.json(response);
    }

    /* -----------------------------------------------------
       EVALUATE
       ----------------------------------------------------- */

    if (body.phase === "evaluate") {
        const response =
            processEvaluation(body);

        return res.json(response);
    }

    /*
     * Unknown phase.
     */
    return res
        .status(400)
        .json({
            error: "INVALID_INPUT"
        });
});

/* =========================================================
   Health endpoint
   ========================================================= */

app.get("/", (req, res) => {
    res.json({
        status: "ok"
    });
});

/* =========================================================
   Start server
   ========================================================= */

app.listen(PORT, "0.0.0.0", () => {
    console.log(
        `BQML gate listening on port ${PORT}`
    );
});