# Agents

## Purpose

`akreon-control` is a general-purpose control and estimation library.

Owns optimal and robust control, estimation, and future MPC, MHE, and CBF
formulations. See docs/ROADMAP.md for implementation status.
Depends on Akreon::Linalg and Akreon::Convex.

## Principles

- Prefer clarity over cleverness.
- Prefer technical rigor over convenience.
- Preserve stable developer APIs where practical.
- Keep interfaces typed, explicit, and testable.
- Keep numerical and control semantics clear and documented.

## Rules

- Domain logic that improves the developer-facing core is appropriate here.
- Do not move closed runner/build/deployment logic into this repo.
- Prefer explicit models, configs, and result objects over loose dictionaries.
- Validate inputs early and surface actionable errors.
- Do not introduce hidden global state or surprising defaults.
- Prioritize constexpr functionality always for real-time safety code.

## Style guide

Please abide by the checked-in
[clang-format](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
configuration in this repo.

After changing C++ headers or sources, run:
`clang-format -i <touched-files>`

To audit formatting across the tracked core sources, use:
`cmake --build <build-dir> --target clang_format_check`

After code changes, run the relevant build and test flow. For host work, prefer:
`cmake --preset host-default` (supply local dependency overrides from README.md as needed)
`cmake --build --preset host-default -j`
`ctest --test-dir out/build/host-default --output-on-failure`

## When changing code

Before making a change, check:

1. Does this belong in the core?
2. Is this clearer for developers?
3. Will this break existing code or workflows?
4. Are assumptions, units, timing, and semantics explicit?
5. Is the behavior tested?
6. Are there regressions across clang, gcc, or target CPU architectures?

## Testing

Changes should include relevant automated coverage in this repo via `ctest`.

## Report

Include:

- Summary
- Changeset
- Core impact
- Testing
- Gaps
- Security issues and/or leaks, when present
- Ablation study results or regressions, when that analysis was actually run

## Goal

Keep `akreon-control` clear, stable, technically rigorous, and easy for developers to use.
