"""Render the final structured report (Outputs 1/2/3) into readable Markdown."""

from __future__ import annotations

from typing import Any


def _quote_line(q: dict[str, Any]) -> str:
    mark = "✓" if q.get("verified") else "?"
    return f'  - [{mark}] "{q.get("text", "").strip()}"  _( {q.get("locator", "?")} )_'


def render_markdown(report: dict[str, Any]) -> str:
    out: list[str] = []
    out.append(f"# Reproduction Plan: {report.get('paper_title') or 'Untitled'}\n")
    out.append(f"- **Paper:** {report.get('paper_url', '')}")
    out.append(f"- **Code:** {report.get('repo_url', '')}")
    out.append(f"- **Confidence:** {report.get('confidence', 'unknown')}")
    if report.get("confidence_rationale"):
        out.append(f"  - {report['confidence_rationale']}")
    out.append("")

    # Output 1
    sel = report.get("output1_experiment_selection", {}) or {}
    out.append("## Output 1 — Selected Experiment (paper-grounded)\n")
    out.append(f"**Experiment:** {sel.get('selected_experiment', '?')}")
    out.append(f"**Target result:** {sel.get('target_result', '?')}\n")
    out.append(f"**Why this one:** {sel.get('why_this_one', '')}\n")
    if sel.get("justification_quotes"):
        out.append("**Exact paper text / table used to choose it:**")
        for q in sel["justification_quotes"]:
            out.append(_quote_line(q))
        out.append("")
    if sel.get("alternatives_considered"):
        out.append("**Alternatives considered:** " + ", ".join(sel["alternatives_considered"]))
        out.append("")

    # Output 2
    cfg = report.get("output2_configuration", {}) or {}
    out.append("## Output 2 — Configuration (paper + code reconciled)\n")
    for key in ("model", "dataset", "data_prep", "seed", "hardware", "environment"):
        field = cfg.get(key) or {}
        if field.get("value") or field.get("note"):
            line = f"- **{key.replace('_', ' ').title()}:** {field.get('value', '')}"
            if field.get("note"):
                line += f"  _({field['note']})_"
            out.append(line)
    out.append("")
    hps = cfg.get("hyperparameters") or []
    if hps:
        out.append("**Hyperparameters:**\n")
        out.append("| name | paper | code default | chosen | status |")
        out.append("|---|---|---|---|---|")
        for h in hps:
            out.append(
                f"| {h.get('name','')} | {h.get('paper_value','')} | "
                f"{h.get('code_default','')} | {h.get('chosen_value','')} | "
                f"{h.get('status','')} |"
            )
        out.append("")

    # Output 3
    plan = report.get("output3_replication_plan", {}) or {}
    out.append("## Output 3 — Commands to Replicate (code-grounded)\n")
    for c in plan.get("commands", []) or []:
        vmark = "✓ flags verified" if c.get("flags_verified") else "⚠ flags unverified"
        src = (c.get("source") or {}).get("path", "")
        out.append(f"### {c.get('step', '?').title()}  _({vmark})_")
        out.append("```bash")
        out.append(c.get("cmd", ""))
        out.append("```")
        if src:
            out.append(f"_source: {src}_")
        if c.get("expected"):
            out.append(f"_expected: {c['expected']}_")
        if c.get("notes"):
            out.append(f"_note: {c['notes']}_")
        out.append("")
    if plan.get("assumptions"):
        out.append("**Plan assumptions:**")
        for a in plan["assumptions"]:
            out.append(f"- {a}")
        out.append("")

    # Output 4 — AMD environment
    amd = report.get("output4_amd_environment")
    if amd:
        out.append("## Output 4 — AMD/ROCm Environment\n")
        out.append(f"- **Base image:** {amd.get('base_image', '?')}")
        if amd.get("base_image_rationale"):
            out.append(f"  - {amd['base_image_rationale']}")
        out.append(f"- **Build status:** {amd.get('build_status', 'not_attempted')}")
        if amd.get("image_tag"):
            out.append(f"- **Image:** `{amd['image_tag']}`")
        ti = amd.get("torch_after_build") or amd.get("torch_baseline") or {}
        if ti:
            out.append(
                f"- **Torch integrity:** torch={ti.get('torch')} "
                f"vision={ti.get('torchvision')} audio={ti.get('torchaudio')} "
                f"| GPU={ti.get('cuda_available')} | intact={ti.get('intact')}"
            )
        if amd.get("has_custom_kernel"):
            out.append(f"- **Custom kernel:** yes — {amd.get('kernel_migration_note', '')}")
        if amd.get("smoke_test_cmd"):
            out.append(
                f"- **Smoke test:** `{amd['smoke_test_cmd']}` "
                f"(passed={amd.get('smoke_test_passed')})"
            )
        for p in amd.get("patches", []) or []:
            out.append(f"  - patch `{p.get('path')}`: {p.get('reason')}")
        for n in amd.get("amd_notes", []) or []:
            out.append(f"  - note: {n}")
        out.append("")

    # Output 5 — Reproduction result
    rr = report.get("output5_reproduction_result")
    if rr:
        out.append("## Output 5 — Reproduction Result (executed on AMD)\n")
        out.append(f"- **Executed:** {rr.get('executed')}")
        if rr.get("command_run"):
            out.append(f"- **Command:** `{rr['command_run']}`")
        out.append(f"- **Achieved:** {rr.get('achieved_metric', '?')}")
        out.append(f"- **Paper target:** {rr.get('paper_target', '?')}")
        out.append(f"- **Match:** {rr.get('match')}  (delta: {rr.get('delta', '?')})")
        if rr.get("notes"):
            out.append(f"- _{rr['notes']}_")
        out.append("")

        # Exec-Score metrics (AutoReproduce/ReproduceBench-compatible), computed here.
        try:
            from .metrics import metric_from_report

            m = metric_from_report(report, report.get("repo_url", "run"))
            out.append("### Exec-Score metrics (ReproduceBench-compatible)\n")
            out.append(f"- **Exec Rate:** {'1.00 (executed)' if m.executed else '0.00 (not executed)'}")
            gap_pct = m.gap * 100
            out.append(
                f"- **Performance Gap (\u2193):** {gap_pct:.2f}%"
                + (
                    f"  (P_ref={m.ref_value:g}, P_agent={m.agent_value:g})"
                    if m.executed and m.agent_value is not None and m.ref_value is not None
                    else ""
                )
            )
            if m.note:
                out.append(f"  - _{m.note}_")
            out.append("")
        except Exception:
            pass

    # Gaps
    gaps = report.get("gaps_and_assumptions") or []
    out.append("## Gaps & Assumptions\n")
    if gaps:
        for g in gaps:
            out.append(f"- {g}")
    else:
        out.append("- None recorded.")
    out.append("")

    return "\n".join(out)
