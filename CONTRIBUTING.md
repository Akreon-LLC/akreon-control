# Contributing

Use the [README](README.md) for setup and package-consumer commands, the
[architecture guide](docs/ARCHITECTURE.md) for ownership boundaries, and the
[support roadmap](docs/ROADMAP.md) for feature status. See
[tests/README.md](tests/README.md) for verification coverage and [AGENTS.md](AGENTS.md)
for agent-specific instructions.

## Documentation ownership

| Source | Owns |
| --- | --- |
| `README.md` | Setup, dependency resolution, install/consumer commands, and documentation entry points |
| `docs/ARCHITECTURE.md` | Component ownership, source map, and data flow |
| `docs/ROADMAP.md` | Compatibility, support status, known gaps, and future work |
| `tests/README.md` | Test inventory and verification scope |
| Header comments | API contracts and algorithm details, rendered by Doxygen |
| `CONTRIBUTING.md` | Change workflow, formatting, and CI administration |

Link to the owning document instead of copying its tables, commands, or details.

## Changes and formatting

Preserve typed, constexpr APIs and numerical semantics. Keep generic numerical
work and domain-specific logic in their owning libraries. Update the relevant
header comments when API contracts or algorithms change; update the roadmap
when support status changes.

Follow `.clang-format` and use clang-format 22 for consistent results:

```sh
cmake --build build --target clang_format_check
cmake --build build --target clang_format_fix
```

These targets are available for a top-level build when clang-format is installed.

## CI configuration

CI checks out dependencies in `.deps`. While repositories are private, set
`AKREON_DEPENDENCY_TOKEN` with read access. Pin reviewed dependency refs through
`AKREON_LINALG_REF`, `AKREON_CONVEX_REF`. These CI refs are separate from the
library's FetchContent release pins described in the README.

Python dependencies and binding tests are optional. Keep notebooks importing the
installed `akreon_native` API. See [bindings/README.md](bindings/README.md) for the contract.
