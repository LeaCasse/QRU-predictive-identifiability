"""Physical IBM QPU test of the preselected V5 collision witness.

Modes: selfcheck, simulate, submit, recover, analyze. Hardware execution
requires an IBM Quantum Open account; only ``submit`` creates a QPU job.
The intervention changes a rotation in the *target circuit* before readout.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qru.circuit import qru_output  # noqa: E402

WITNESS = ROOT / "results" / "predictive" / "latent_twins.json"
SEED = 203
X = (0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi)
ALPHA = (1.0, 2.0)
TRANSVERSE = (1.0, 2.2)
PARALLEL = (1.1, 2.2)
STAGES = ("baseline", "probe", "future")
BONFERRONI_TAIL = .05 / (2 * 2 * 22)  # 22 pairs of A/B Bernoulli rates


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def witness():
    data = WITNESS.read_bytes()
    trials = json.loads(data)["numerical_trials"]
    trial = next(t for t in trials if t["seed"] == SEED)
    return trial, hashlib.sha256(data).hexdigest()


def design(stage, random_seed=203):
    trial, digest = witness()
    rows = []

    def add(label, x, alpha, phase, branch):
        theta = np.array(trial[f"theta_{branch.lower()}"])
        shifted = theta.copy()
        shifted[1, 2] += phase
        ideal_mean = float(qru_output([x], shifted, alpha)[0])
        rows.append({"label": label, "branch": branch, "x": float(x),
                     "alpha": list(alpha), "phase": float(phase),
                     "theta": theta.tolist(), "ideal_mean": ideal_mean})

    if stage == "baseline":
        for x in X:
            for branch in "AB":
                add("passive", x, ALPHA, 0.0, branch)
    elif stage == "probe":
        for phase, label in ((0.0, "anchor"), (np.pi / 2, "informative"),
                             (3 * np.pi / 2, "negative_phase")):
            for branch in "AB":
                add(label, np.pi, ALPHA, phase, branch)
    elif stage == "future":
        for label, alpha in (("transverse", TRANSVERSE), ("parallel", PARALLEL)):
            for x in X:
                for branch in "AB":
                    add(label, x, alpha, 0.0, branch)
        for branch in "AB":
            add("anchor", np.pi, ALPHA, 0.0, branch)
    else:
        raise ValueError(stage)
    rng = np.random.default_rng(random_seed + STAGES.index(stage))
    rows = [rows[i] for i in rng.permutation(len(rows))]
    for i, row in enumerate(rows):
        row["pub_index"] = i
    return {"protocol": "v5_seed203_qpu_v1", "stage": stage,
            "witness_sha256": digest, "rows": rows, "created_utc": utc_now(),
            "note": "Random PUB order is recorded; execution order is not guaranteed."}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def qiskit_template():
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    parameters = ParameterVector("rotation", 6)
    qc = QuantumCircuit(1, 1)
    for layer in range(2):
        qc.ry(parameters[3 * layer], 0)
        qc.rx(parameters[3 * layer + 1], 0)
        qc.rz(parameters[3 * layer + 2], 0)
    qc.measure(0, 0)
    return qc, parameters


def gate_angles(row):
    t = row["theta"]
    return [row["alpha"][0] * row["x"] + t[0][2], t[0][0], t[0][1],
            row["alpha"][1] * row["x"] + t[1][2] + row["phase"],
            t[1][0], t[1][1]]


def selfcheck():
    from qiskit.quantum_info import Statevector
    circuit, params = qiskit_template()
    for stage in STAGES:
        for row in design(stage)["rows"]:
            bound = circuit.remove_final_measurements(inplace=False).assign_parameters(
                dict(zip(params, gate_angles(row))))
            probabilities = Statevector.from_instruction(bound).probabilities()
            expect_z = float(probabilities[0] - probabilities[1])
            if abs(expect_z - row["ideal_mean"]) > 1e-12:
                raise AssertionError((stage, row, expect_z))
    print("PASS: Qiskit circuit and frozen V5 QRU agree for every design row")


def simulate(stage, shots, seed, output):
    payload = design(stage)
    rng = np.random.default_rng(seed + STAGES.index(stage))
    for row in payload["rows"]:
        p = (1 - row["ideal_mean"]) / 2  # probability of bit 1
        bits = rng.binomial(1, p, size=shots).astype(str).tolist()
        row["bitstrings"] = bits
        row["counts"] = {"0": bits.count("0"), "1": bits.count("1")}
    payload.update(source="ideal_binomial_simulator", shots=shots,
                   generated_utc=utc_now())
    write_json(output, payload)
    print(output)


def ibm_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    # Restrict instance discovery to free Open instances; do not pass a CRN,
    # because IBM ignores plan filters when a specific instance is supplied.
    return QiskitRuntimeService(name="qru_open", plans_preference=["open"],
                                region="us-east")


def prepared_hardware(stage, backend, physical_qubit):
    from qiskit.transpiler import generate_preset_pass_manager
    payload = design(stage)
    if not 0 <= physical_qubit < backend.num_qubits:
        raise ValueError("physical qubit is outside this backend")
    circuit, parameters = qiskit_template()
    pass_manager = generate_preset_pass_manager(
        optimization_level=0, backend=backend, initial_layout=[physical_qubit])
    isa = pass_manager.run(circuit)
    if isa.num_parameters != 6:
        raise AssertionError("parameterized circuit was altered by transpilation")
    payload.update(backend=backend.name, physical_qubit=physical_qubit,
                   isa_ops=dict(isa.count_ops()), isa_depth=isa.depth(),
                   qiskit_parameters=[p.name for p in parameters])
    try:
        payload["calibration_last_update"] = str(backend.properties().last_update_date)
    except (AttributeError, TypeError, ValueError):
        payload["calibration_last_update"] = None
    pubs = [(isa, dict(zip(parameters, gate_angles(row)))) for row in payload["rows"]]
    return payload, pubs


def add_result(payload, result):
    if len(result) != len(payload["rows"]):
        raise ValueError("PUB count changed")
    for row, pub in zip(payload["rows"], result):
        bitstrings = pub.data.c.get_bitstrings()
        if any(bit not in ("0", "1") for bit in bitstrings):
            raise ValueError("expected one measured physical qubit")
        row["bitstrings"] = bitstrings
        row["counts"] = {"0": bitstrings.count("0"), "1": bitstrings.count("1")}
    metadata = getattr(result, "metadata", {})
    spans = metadata.get("execution", {}).get("execution_spans") if isinstance(metadata, dict) else None
    if spans is not None:
        try:
            payload["execution_spans"] = [{"start": str(s.start), "stop": str(s.stop),
                                            "size": int(s.size),
                                            "pub_indices": [j for j in range(len(payload["rows"]))
                                                            if np.any(s.mask(j))]} for s in spans]
        except (AttributeError, TypeError):
            payload["execution_spans_note"] = "Consult original job metadata for spans"
    payload["received_utc"] = utc_now()
    payload["status"] = "complete"


def add_usage(payload, job):
    try:
        payload["qpu_seconds"] = float(job.metrics()["usage"]["quantum_seconds"])
    except (KeyError, TypeError, AttributeError, ValueError):
        payload["qpu_seconds_note"] = "See workload usage in IBM Quantum dashboard"


def submit(args):
    from qiskit_ibm_runtime.executor_sampler import Sampler
    service = ibm_service()
    backend = service.backend(args.backend)
    if backend.configuration().simulator:
        raise ValueError("a real physical backend is required")
    payload, pubs = prepared_hardware(args.stage, backend, args.physical_qubit)
    payload.update(source="ibm_quantum_qpu", shots=args.shots,
                   plan_filter="open", status="prepared")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if not args.submit:
        print(json.dumps({"backend": payload["backend"], "stage": args.stage,
                          "physical_qubit": args.physical_qubit,
                          "isa_ops": payload["isa_ops"], "circuits": len(pubs),
                          "shots_each": args.shots,
                          "total_executions": len(pubs) * args.shots,
                          "status": "preview_only; rerun with --submit"}, indent=2))
        return
    write_json(args.output, payload)
    sampler = Sampler(mode=backend)
    # Keep raw counts without randomized circuit transformations.
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.dynamical_decoupling.enable = False
    job = sampler.run(pubs, shots=args.shots)
    payload.update(job_id=job.job_id(), submitted_utc=utc_now(), status="submitted")
    write_json(args.output, payload)  # persist ID before waiting on QPU
    print(f"Submitted {args.stage}: {job.job_id()} -> {args.output}", flush=True)
    try:
        add_result(payload, job.result())
        add_usage(payload, job)
        write_json(args.output, payload)
    except Exception:
        # Retain recoverable job ID even if a network read fails.
        raise
    print(f"Saved {len(payload['rows'])} circuit results -> {args.output}")


def recover(args):
    payload = json.loads(Path(args.file).read_text())
    if payload.get("source") != "ibm_quantum_qpu" or not payload.get("job_id"):
        raise ValueError("file lacks an IBM Quantum job ID")
    if payload.get("status") == "complete":
        raise ValueError("this file is already complete")
    job = ibm_service().job(payload["job_id"])
    add_result(payload, job.result())
    add_usage(payload, job)
    write_json(args.file, payload)
    print(f"Recovered {args.file}")


def mean_interval(row, tail):
    n0, n1 = row["counts"]["0"], row["counts"]["1"]
    n = n0 + n1
    if n <= 0:
        raise ValueError("zero shots")
    lo = 0.0 if n0 == 0 else beta.ppf(tail, n0, n1 + 1)
    hi = 1.0 if n0 == n else beta.ppf(1 - tail, n0 + 1, n1)
    return (n0 - n1) / n, 2 * lo - 1, 2 * hi - 1


def paired(rows, label, x, alpha, phase, tail):
    pairs = [r for r in rows if r["label"] == label and
             abs(r["x"] - x) < 1e-9 and
             np.allclose(r["alpha"], alpha) and abs(r["phase"] - phase) < 1e-9]
    if len(pairs) != 2 or set(r["branch"] for r in pairs) != {"A", "B"}:
        raise ValueError("missing or duplicated paired circuits")
    a, b = (next(r for r in pairs if r["branch"] == branch) for branch in "AB")
    ma, la, ua = mean_interval(a, tail)
    mb, lb, ub = mean_interval(b, tail)
    d = ma - mb
    low, high = la - ub, ua - lb
    return {"A": ma, "B": mb, "difference": d,
            "A_interval": [la, ua], "B_interval": [lb, ub],
            "abs_gap_lower": max(0., low, -high),
            "abs_gap_upper": max(abs(low), abs(high)),
            "n_each": min(sum(a["counts"].values()), sum(b["counts"].values()))}


def ambiguity(p_a, p_b, shots):
    """Expected posterior Bernoulli-branch variance for equal branch priors."""
    ka = binom.pmf(np.arange(shots + 1), shots, p_a)
    kb = binom.pmf(np.arange(shots + 1), shots, p_b)
    total = ka + kb
    return float(np.sum(np.divide(.5 * ka * kb, total,
                                  out=np.zeros_like(total), where=total > 0)))


def risk_bootstrap(baseline, probe, future, repetitions=2000):
    """Parametric 95% interval for the paired four-shot excess-risk gain."""
    rng = np.random.default_rng(58231)
    estimates = []
    for pair in (baseline, probe, future):
        estimates.append([(1 + pair[c]) / 2 for c in "AB"])
    n = min(pair["n_each"] for pair in (baseline, probe, future))
    values = np.empty(repetitions)
    for i in range(repetitions):
        p0, p1, pf = [rng.binomial(n, rates) / n for rates in estimates]
        d = 2 * (pf[0] - pf[1])
        values[i] = d**2 * (ambiguity(*p0, 4) - ambiguity(*p1, 4))
    return [float(t) for t in np.quantile(values, [.025, .975])]


def check_stage(path, baseline_path=None):
    record = json.loads(Path(path).read_text())
    if record.get("status") != "complete" and record.get("source") != "ideal_binomial_simulator":
        raise ValueError("an acquisition is incomplete")
    tail = BONFERRONI_TAIL
    if record["stage"] == "baseline":
        comparisons = [paired(record["rows"], "passive", x, ALPHA, 0., tail) for x in X]
        out = {"baseline_max_gap_upper": max(p["abs_gap_upper"] for p in comparisons),
               "pass": all(p["abs_gap_upper"] < .12 for p in comparisons)}
    elif record["stage"] == "probe":
        positive = paired(record["rows"], "informative", np.pi, ALPHA,
                          np.pi / 2, tail)
        negative = paired(record["rows"], "negative_phase", np.pi, ALPHA,
                          3 * np.pi / 2, tail)
        out = {"phase_gap_lower": positive["abs_gap_lower"],
               "negative_phase_gap_upper": negative["abs_gap_upper"],
               "pass": positive["abs_gap_lower"] > .4 and negative["abs_gap_upper"] < .15}
    else:
        trans = paired(record["rows"], "transverse", np.pi, TRANSVERSE, 0., tail)
        pars = [paired(record["rows"], "parallel", x, PARALLEL, 0., tail)
                for x in X]
        out = {"transverse_gap_lower_at_pi": trans["abs_gap_lower"],
               "parallel_max_gap_upper": max(p["abs_gap_upper"] for p in pars),
               "pass": trans["abs_gap_lower"] > .2 and
                       all(p["abs_gap_upper"] < .15 for p in pars)}
    if baseline_path is not None:
        baseline_record = json.loads(Path(baseline_path).read_text())
        if baseline_record["stage"] != "baseline" or baseline_record["source"] != record["source"]:
            raise ValueError("baseline source or stage mismatch")
        if baseline_record["witness_sha256"] != record["witness_sha256"]:
            raise ValueError("baseline witness mismatch")
        for field in ("backend", "physical_qubit", "plan_filter", "isa_ops"):
            if record["source"] == "ibm_quantum_qpu" and baseline_record[field] != record[field]:
                raise ValueError(f"physical target changed: {field}")
        base = paired(baseline_record["rows"], "passive", np.pi, ALPHA, 0., tail)
        current = paired(record["rows"], "anchor", np.pi, ALPHA, 0., tail)
        stability = []
        for branch in "AB":
            l0, u0 = base[f"{branch}_interval"]
            l1, u1 = current[f"{branch}_interval"]
            stability.append(max(abs(l1 - u0), abs(u1 - l0)))
        out["anchor_stability_upper"] = max(stability)
        out["pass"] = bool(out["pass"] and current["abs_gap_upper"] < .12
                           and max(stability) < .12)
    print(json.dumps({"stage": record["stage"], "source": record["source"],
                      "stage_gate": out}, indent=2))
    return out


def analyze(files, output=None):
    records = [json.loads(Path(f).read_text()) for f in files]
    by_stage = {r["stage"]: r for r in records}
    if set(by_stage) != set(STAGES) or len(records) != 3:
        raise ValueError("provide exactly one complete file per stage")
    for record in records:
        if record.get("status") != "complete" and record.get("source") != "ideal_binomial_simulator":
            raise ValueError("an acquisition is incomplete")
    if len({r["source"] for r in records}) != 1:
        raise ValueError("cannot mix simulated and physical results")
    if len({r["witness_sha256"] for r in records}) != 1:
        raise ValueError("witness changed")
    if records[0]["source"] == "ibm_quantum_qpu":
        for field in ("backend", "physical_qubit", "plan_filter", "isa_ops"):
            if len({json.dumps(r[field], sort_keys=True) for r in records}) != 1:
                raise ValueError(f"hardware identity/compilation changed: {field}")
    # Bonferroni simultaneous coverage across every individual Bernoulli rate
    # used by the gate; 2 tails * 2 branches * 22 sites.
    tail = BONFERRONI_TAIL
    baseline = [paired(by_stage["baseline"]["rows"], "passive", x, ALPHA, 0., tail)
                for x in X]
    probe = paired(by_stage["probe"]["rows"], "informative", np.pi,
                   ALPHA, np.pi / 2, tail)
    negative = paired(by_stage["probe"]["rows"], "negative_phase", np.pi,
                      ALPHA, 3 * np.pi / 2, tail)
    future = [paired(by_stage["future"]["rows"], "transverse", x,
                     TRANSVERSE, 0., tail) for x in X]
    parallel = [paired(by_stage["future"]["rows"], "parallel", x,
                       PARALLEL, 0., tail) for x in X]
    anchor = [paired(by_stage[s]["rows"], "anchor", np.pi, ALPHA, 0., tail)
              for s in ("probe", "future")]
    base_at_pi = baseline[X.index(np.pi)]
    # 95% simultaneous bounds for changes in each A/B anchor mean, calculated
    # by combining the separately covered rate intervals.
    anchor_leak = max(p["abs_gap_upper"] for p in anchor)
    passive_peak = max(p["abs_gap_upper"] for p in baseline)
    # A symmetric hardware offset could leave A/B equal while changing the
    # target response between sessions. Monitor each physical branch too.
    anchor_stability = []
    for other in anchor:
        for branch in "AB":
            l0, u0 = base_at_pi[f"{branch}_interval"]
            l1, u1 = other[f"{branch}_interval"]
            anchor_stability.append(max(abs(l1 - u0), abs(u1 - l0)))
    delta = future[X.index(np.pi)]["A"] - future[X.index(np.pi)]["B"]
    # Equal-budget four-shot comparison; estimates use independent physical
    # response frequencies, not simulated or fitted oracle noise.
    p_pass = [(1 + base_at_pi[c]) / 2 for c in "AB"]
    p_probe = [(1 + probe[c]) / 2 for c in "AB"]
    passive_excess = delta**2 * ambiguity(*p_pass, 4)
    phase_excess = delta**2 * ambiguity(*p_probe, 4)
    risk_interval = risk_bootstrap(base_at_pi, probe, future[X.index(np.pi)])
    gates = {
        "passive_equivalence_mean_gap_upper_below_0p12": passive_peak < .12,
        "phase_identifies_mean_gap_lower_above_0p40": probe["abs_gap_lower"] > .40,
        "negative_phase_mean_gap_upper_below_0p15": negative["abs_gap_upper"] < .15,
        "transverse_mean_gap_lower_at_pi_above_0p20": future[X.index(np.pi)]["abs_gap_lower"] > .20,
        "parallel_mean_gap_upper_all_sites_below_0p15":
            max(p["abs_gap_upper"] for p in parallel) < .15,
        "anchors_mean_gap_upper_below_0p12": anchor_leak < .12,
        "anchor_stability_each_branch_upper_below_0p12": max(anchor_stability) < .12,
        "four_shot_phase_excess_risk_less_than_passive": phase_excess < passive_excess,
        "four_shot_risk_gain_bootstrap_lower_above_zero": risk_interval[0] > 0.,
    }
    gates = {key: bool(value) for key, value in gates.items()}
    out = {"source": records[0]["source"], "simultaneous_confidence": .95,
           "tail_probability_per_binomial_rate": tail,
           "thresholds_are_predeclared_in_script": True,
           "baseline": baseline, "probe": probe, "negative_probe": negative,
           "future_transverse": future, "future_parallel": parallel,
           "anchors": anchor, "anchor_stability_upper": max(anchor_stability),
           "four_shot_excess_risk": {"passive": passive_excess,
                                     "phase": phase_excess,
                                     "difference": passive_excess - phase_excess,
                                     "parametric_bootstrap_95_interval": risk_interval,
                                     "note": "Conditional risk for a programmed two-branch target; bootstrap does not account for nonstationary hardware."},
           "gates": gates, "all_gates_pass": all(gates.values()),
           "interpretation": "Physical mechanism only; the drift was programmed, and a matched path-Fourier model receives identical information."}
    if output:
        write_json(output, out)
    print(json.dumps({"source": out["source"], "gates": gates,
                      "four_shot_excess_risk": out["four_shot_excess_risk"],
                      "saved": output}, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("selfcheck")
    sim = sub.add_parser("simulate")
    sim.add_argument("--stage", choices=STAGES, required=True)
    sim.add_argument("--shots", type=int, default=8192)
    sim.add_argument("--seed", type=int, default=731)
    sim.add_argument("--output", required=True)
    hw = sub.add_parser("submit")
    hw.add_argument("--stage", choices=STAGES, required=True)
    hw.add_argument("--backend", required=True)
    hw.add_argument("--physical-qubit", type=int, required=True)
    hw.add_argument("--shots", type=int, default=8192)
    hw.add_argument("--output", required=True)
    hw.add_argument("--submit", action="store_true", help="actually consume QPU time")
    rec = sub.add_parser("recover")
    rec.add_argument("--file", required=True)
    check = sub.add_parser("check")
    check.add_argument("--file", required=True)
    check.add_argument("--baseline", help="verify same target and stable passive anchor")
    ana = sub.add_parser("analyze")
    ana.add_argument("files", nargs=3, help="baseline probe future JSON")
    ana.add_argument("--output")
    args = p.parse_args()
    if args.mode in ("submit", "simulate") and args.shots <= 0:
        p.error("shots must be positive")
    if args.mode == "selfcheck":
        selfcheck()
    elif args.mode == "simulate":
        simulate(args.stage, args.shots, args.seed, args.output)
    elif args.mode == "submit":
        submit(args)
    elif args.mode == "recover":
        recover(args)
    elif args.mode == "check":
        check_stage(args.file, args.baseline)
    else:
        analyze(args.files, args.output)


if __name__ == "__main__":
    main()
