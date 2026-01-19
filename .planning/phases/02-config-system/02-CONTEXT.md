# Phase 2: Config System - Context

**Gathered:** 2026-01-19
**Status:** Ready for planning

<vision>
## How This Should Work

A single Config dataclass that serves as the control center for the entire pipeline. You create a Config, set the options you need, and everything downstream respects it.

The key architectural insight: **fixed interfaces, flexible implementations**. The Config doesn't just hold values — it specifies *which* algorithms and architectures to use. Classification, clustering, autoencoder architecture, wavelength ranking — these are all pluggable components with standardized inputs/outputs.

Want to add a new classification method? Implement the interface, register it, and it's immediately available in Config. The system is designed for extensibility without changing core code.

```python
# Built-ins via string identifiers
config = Config(
    sample_name="Lichens_2",
    classifier="knn",
    clustering="kmeans",
    autoencoder_architecture="standard"
)

# Custom implementations passed directly
config = Config(
    sample_name="Lichens_2",
    classifier=MyCustomClassifier,
    clustering=my_clustering_function
)
```

</vision>

<essential>
## What Must Be Nailed

All three are must-haves:

- **Clean, typed interface** — IDE autocomplete, clear documentation of what each option does, type hints throughout
- **Drop-in replacement** — Existing code can switch to new Config with minimal changes, backward compatibility where possible
- **Validation with smart defaults** — Sensible defaults for everything, validates that settings make sense together

Plus the architectural foundation:

- **Pluggable component system** — Fixed interfaces for classification, clustering, autoencoder, wavelength ranking
- **Dual registration** — String identifiers for built-ins, direct class/function for custom implementations

</essential>

<boundaries>
## What's Out of Scope

- **File loading (YAML/JSON)** — Config is just the dataclass; loading from files is a separate concern for another plan
- **Runtime reconfiguration** — Config is set once at initialization, not changed mid-analysis
- **GUI/CLI for configuration** — Programmatic only; no interactive config tools
- **Implementing all algorithms** — Define the interfaces; actual implementations come in later phases

</boundaries>

<specifics>
## Specific Ideas

**Pluggable components identified:**

1. **Classification methods** — Currently KNN, but the interface should support SVM, Random Forest, neural classifiers, etc.
2. **Autoencoder architectures** — Configurable layer sizes, activation functions, latent dimensions
3. **Clustering algorithms** — K-means, DBSCAN, hierarchical, spectral clustering
4. **Wavelength ranking methods** — Different attribution/importance scoring approaches

**Interface pattern:**
Each pluggable component has a fixed interface (input types, output types, required methods). Implementations are interchangeable as long as they satisfy the interface.

**Registration pattern:**
- Built-ins registered by name: `"knn"`, `"kmeans"`, `"standard"`
- Custom implementations passed directly as class or callable
- Config validates that the provided implementation satisfies the required interface

</specifics>

<notes>
## Additional Context

This Config system is foundational — it affects how the entire library is structured. The pluggable architecture vision should inform Phase 3 (Data Types), Phase 4 (Analysis Engine), and beyond.

The user emphasized this is about the "whole flow" — Config isn't just settings, it's the specification of *which* pipeline to run.

Full variant coverage: all 5+ existing variant behaviors should be expressible through Config options, ensuring nothing from the research codebase is lost.

</notes>

---

*Phase: 02-config-system*
*Context gathered: 2026-01-19*
