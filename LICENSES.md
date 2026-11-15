# License Ledger

This document tracks the license of every component that goes into agan. Only Apache-2.0, MIT, BSD-2/3, and similarly permissive licenses are allowed in the core.

## Components

| Component | License | Source | Notes |
|---|---|---|---|
| (add entries as components are adopted) | | | |

## Policy
- **Core code**: Apache-2.0
- **Model weights**: Must be compatible with all inputs; documented in `MODELS.md`
- **Training data**: Only open-license sources; datasheet in `data/datasheet.md`
- **Dependencies**: Pinned versions; license checked before adding

## License Checking
Before adding any dependency, model, or dataset:
1. Verify license is permissive (Apache-2.0, MIT, BSD, CC-BY, CC0, etc.)
2. Record in this ledger
3. Ensure no GPL, AGPL, SSPL, or other viral/restrictive licenses in core
