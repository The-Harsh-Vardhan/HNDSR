---
applyTo: "**/*.{c,cc,cpp,cxx,h,hh,hpp,py,js,jsx,ts,tsx,rs,go,java,kt,kts,cs,swift,m,mm}"
---

Use Gerard J. Holzmann's Power of 10 as the default coding baseline.

- Prefer simple control flow and avoid recursion unless the bound is explicit and justified.
- Put hard limits, budgets, or timeouts on loops, retries, polling, and traversal over external inputs.
- Avoid unbounded steady-state allocation and favor explicit resource budgets after startup.
- Keep functions small enough to review as one unit, using about 60 lines as a warning threshold.
- Use explicit assertions, invariants, preconditions, and postconditions where silent failure would matter.
- Keep variables, mutable state, and visibility at the narrowest practical scope.
- Check fallible return values and validate inputs at module boundaries.
- Keep metaprogramming, feature flags, and build-time branching restrained and readable.
- Limit indirection, dynamic dispatch, and pointer-like mechanisms unless the payoff is explicit.
- Enable strict warnings and static analysis early, and fix warnings instead of normalizing them away.
- If a Power of 10 rule does not map cleanly to the current language or framework, say so explicitly instead of inventing a new "Holzmann rule."
