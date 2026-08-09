#!/usr/bin/env python3
"""
Build deterministic perturbation variants for the remaining eight scenarios.

This keeps the already-running core-four file stable and generates a separate
bundle for the second-wave scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = ROOT / "src" / "ai_chaos_scenarios_v1.json"
OUT_PATH = ROOT / "deterministic_experiments" / "perturbation_variants_remaining_v1.json"

TARGET_IDS = ["IM-02", "HL-02", "IN-01", "IN-02", "CL-04", "SP-01", "SP-03", "AI-02"]

SURFACE_TEMPLATES = [
    {
        "constraint": "The governing constraint is as follows: {value}",
        "evidence": "Relevant evidence is as follows: {value}",
        "choices": "The decision uses these named choices: {value}",
        "options": "The available options are: {value}",
        "principle": "The principle question is: {value}",
        "cases": "Consider these example cases:",
        "endpoint": "The required endpoint is: {value}",
        "note": "Additional note: {value}",
    },
    {
        "constraint": "The decision must respect this constraint: {value}",
        "evidence": "Use the following evidence: {value}",
        "choices": "The available allocation categories are: {value}",
        "options": "The committee can choose among these options: {value}",
        "principle": "The principle vote asks: {value}",
        "cases": "Use the following cases:",
        "endpoint": "The final endpoint is: {value}",
        "note": "Keep in mind this note: {value}",
    },
    {
        "constraint": "The operative constraint is: {value}",
        "evidence": "The evidence provided is: {value}",
        "choices": "The relevant funding categories are: {value}",
        "options": "The decision options are: {value}",
        "principle": "The governing principle question is: {value}",
        "cases": "The cases to apply are:",
        "endpoint": "The discrete endpoint is: {value}",
        "note": "Important note: {value}",
    },
    {
        "constraint": "This decision is subject to the following constraint: {value}",
        "evidence": "The scenario provides this evidence: {value}",
        "choices": "These are the named allocation choices: {value}",
        "options": "These are the available options: {value}",
        "principle": "The key principle question is: {value}",
        "cases": "Apply the policy to these cases:",
        "endpoint": "The required output endpoint is: {value}",
        "note": "Scenario note: {value}",
    },
    {
        "constraint": "The relevant limit is: {value}",
        "evidence": "The available evidence is: {value}",
        "choices": "The budget choices are: {value}",
        "options": "The options under consideration are: {value}",
        "principle": "The principle vote asks the following: {value}",
        "cases": "Use these cases as reference points:",
        "endpoint": "The endpoint to return is: {value}",
        "note": "Supplementary note: {value}",
    },
    {
        "constraint": "The scenario imposes this constraint: {value}",
        "evidence": "The scenario gives this evidence: {value}",
        "choices": "The choice categories are: {value}",
        "options": "The policy options are: {value}",
        "principle": "The committee must answer this principle question: {value}",
        "cases": "Evaluate these cases:",
        "endpoint": "The output endpoint is: {value}",
        "note": "Implementation note: {value}",
    },
    {
        "constraint": "Keep this constraint fixed: {value}",
        "evidence": "Take this evidence as given: {value}",
        "choices": "Treat these as the available allocation choices: {value}",
        "options": "Treat these as the available options: {value}",
        "principle": "Frame the principle vote as: {value}",
        "cases": "Consider these concrete cases:",
        "endpoint": "Keep this endpoint fixed: {value}",
        "note": "Keep this note fixed: {value}",
    },
    {
        "constraint": "The case must satisfy this constraint: {value}",
        "evidence": "This evidence should guide the decision: {value}",
        "choices": "These named categories are available: {value}",
        "options": "Choose among the following options: {value}",
        "principle": "The principle question to answer is: {value}",
        "cases": "The scenario includes these cases:",
        "endpoint": "Use this endpoint: {value}",
        "note": "Read this note as part of the scenario: {value}",
    },
    {
        "constraint": "The standing constraint is: {value}",
        "evidence": "The standing evidence is: {value}",
        "choices": "The listed choices are: {value}",
        "options": "The listed options are: {value}",
        "principle": "The listed principle vote is: {value}",
        "cases": "The listed cases are:",
        "endpoint": "The listed endpoint is: {value}",
        "note": "The listed note is: {value}",
    },
    {
        "constraint": "Hold fixed the following constraint: {value}",
        "evidence": "Hold fixed the following evidence: {value}",
        "choices": "Hold fixed the following named choices: {value}",
        "options": "Hold fixed the following options: {value}",
        "principle": "Hold fixed the following principle question: {value}",
        "cases": "Hold fixed the following cases:",
        "endpoint": "Hold fixed the following endpoint: {value}",
        "note": "Hold fixed the following note: {value}",
    },
]


def load_scenarios() -> dict[str, str]:
    payload = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    return {item["scenario_id"]: item["text"] for item in payload["scenarios"]}


def parse_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for line in text.splitlines():
        if ": " in line:
            head, tail = line.split(": ", 1)
            if head in {
                "SCENARIO_ID", "Institution", "Constraint", "Evidence", "Choices",
                "Options", "Principle vote", "Cases", "Discrete endpoint", "NOTE"
            }:
                sections.append((head, tail))
                continue
        sections.append(("", line))
    return sections


def surface_variant(text: str, idx: int) -> str:
    template = SURFACE_TEMPLATES[idx]
    out: list[str] = []
    for head, tail in parse_sections(text):
        if head == "SCENARIO_ID":
            out.append(f"SCENARIO_ID: {tail}")
        elif head == "Institution":
            out.append(f"Institution: {tail}")
        elif head == "Constraint":
            out.append(f"Constraint: {template['constraint'].format(value=tail)}")
        elif head == "Evidence":
            out.append(f"Evidence: {template['evidence'].format(value=tail)}")
        elif head == "Choices":
            out.append(f"Choices: {template['choices'].format(value=tail)}")
        elif head == "Options":
            out.append(f"Options: {template['options'].format(value=tail)}")
        elif head == "Principle vote":
            out.append(f"Principle vote: {template['principle'].format(value=tail)}")
        elif head == "Cases":
            out.append(f"Cases: {template['cases']}")
        elif head == "Discrete endpoint":
            out.append(f"Discrete endpoint: {template['endpoint'].format(value=tail)}")
        elif head == "NOTE":
            out.append(f"NOTE: {template['note'].format(value=tail)}")
        else:
            out.append(tail if head == "" else f"{head}: {tail}")
    return "\n".join(out)


def split_case_lines(text: str) -> tuple[list[str], list[str]]:
    header_lines: list[str] = []
    case_lines: list[str] = []
    in_cases = False
    for line in text.splitlines():
        if line.startswith("Cases:"):
            in_cases = True
            header_lines.append(line)
            continue
        if in_cases and (line.startswith("case") or line.startswith("- case") or line.startswith("(1) case")):
            case_lines.append(line)
            continue
        if in_cases and line.startswith("Discrete endpoint:"):
            in_cases = False
        if not in_cases:
            header_lines.append(line)
    return header_lines, case_lines


def formatting_variant(text: str, idx: int) -> str:
    lines = text.splitlines()
    if idx == 0:
        out = []
        for line in lines:
            if line.startswith(("case1)", "case2)", "case3)")):
                out.append("- " + line)
            elif line.startswith(("A)", "B)", "C)")):
                out.append(line)
            elif ": " in line and line.split(": ", 1)[0] in {"Constraint", "Evidence", "Choices"}:
                head, tail = line.split(": ", 1)
                out.append(f"{head}:")
                out.append(f"- {tail}")
            else:
                out.append(line)
        return "\n".join(out)
    if idx == 1:
        return text.replace("\nOptions:\n", "\n\nOptions:\n").replace("\nCases:\n", "\n\nCases:\n")
    if idx == 2:
        return text.replace("Constraint:", "Constraint\n").replace("Evidence:", "\nEvidence\n").replace("Options:", "\nOptions\n").replace("Cases:", "\nCases\n").replace("Principle vote:", "\nPrinciple vote\n").replace("Discrete endpoint:", "\nDiscrete endpoint\n")
    if idx == 3:
        pieces = []
        for line in lines:
            if any(line.startswith(prefix) for prefix in ["SCENARIO_ID:", "Institution:", "Constraint:", "Evidence:", "Choices:", "Options:", "Principle vote:", "Cases:", "Discrete endpoint:", "NOTE:"]):
                pieces.append(line.replace(": ", " | ", 1))
            else:
                pieces.append(line)
        return "\n".join(pieces)
    if idx == 4:
        out = []
        for line in lines:
            if line.startswith(("Constraint:", "Evidence:", "Choices:", "Principle vote:", "Discrete endpoint:", "NOTE:")):
                head, tail = line.split(": ", 1)
                out.extend([f"{head}:", tail])
            else:
                out.append(line)
        return "\n".join(out)
    if idx == 5:
        out = []
        for line in lines:
            if line.startswith(("A)", "B)", "C)", "case1)", "case2)", "case3)")):
                out.append("- " + line)
            else:
                out.append(line)
        return "\n".join(out)
    if idx == 6:
        out = []
        counter = 1
        for line in lines:
            if line.startswith(("case1)", "case2)", "case3)")):
                out.append(f"({counter}) {line}")
                counter += 1
            else:
                out.append(line)
        return "\n".join(out)
    if idx == 7:
        return text.replace("\nEvidence:", "\n\nEvidence:").replace("\nPrinciple vote:", "\n\nPrinciple vote:").replace("\nDiscrete endpoint:", "\n\nDiscrete endpoint:")
    if idx == 8:
        out = []
        for line in lines:
            if line.startswith(("case1)", "case2)", "case3)")):
                out.append(line)
            elif line.startswith(("A)", "B)", "C)")):
                out.append(line)
            elif ": " in line and line.split(": ", 1)[0] in {"Constraint", "Evidence", "Choices", "Principle vote", "Discrete endpoint", "NOTE"}:
                head, tail = line.split(": ", 1)
                out.append(f"{head}:")
                out.append(tail)
            else:
                out.append(line)
        return "\n".join(out)
    raise ValueError(idx)


def build_variant_set(scenario_id: str, text: str) -> dict:
    variants = [{
        "variant_id": f"{scenario_id}-canonical",
        "label": "Canonical wording",
        "family": "canonical",
        "text": text,
    }]
    for i in range(10):
        variants.append({
            "variant_id": f"{scenario_id}-surface-{i+1:02d}",
            "label": f"Surface rephrasing {i+1}",
            "family": "surface_rephrasing",
            "text": surface_variant(text, i),
        })
    for i in range(9):
        variants.append({
            "variant_id": f"{scenario_id}-format-{i+1:02d}",
            "label": f"Formatting change {i+1}",
            "family": "formatting_changes",
            "text": formatting_variant(text, i),
        })
    return {
        "scenario_id": scenario_id,
        "canonical_variant_id": f"{scenario_id}-canonical",
        "variants": variants,
    }


def main() -> None:
    scenarios = load_scenarios()
    payload = {
        "variant_sets": [build_variant_set(sid, scenarios[sid]) for sid in TARGET_IDS]
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
