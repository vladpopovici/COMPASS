# HistoCompass: An Open, Extensible, and Verifiable Platform for Annotating 2D Spatial Biology Data

**Status**: proposal draft, for discussion with thesis coordinator
**Relationship to COMPASS**: HistoCompass is a bounded, bachelor-thesis-scoped
artifact built on the architecture and decisions recorded in the COMPASS
project (`decisions.md`, `overview.md`). It reuses COMPASS's core data
model (pyramidal raster protocol, annotation store, layer/group concept)
but is its own scoped deliverable, not a claim to ship COMPASS itself.

---

## 1. Formal category fit — open item, confirm before finalizing

The faculty's "innovative solution" category text states the output should
"mainly" contain instructions for a newly created laboratory task plus a
discussion of its contribution to students and methodological instructions
for teachers, while separately listing "creating a software interface for
processing experimental data" as an example problem type.

**This proposal is platform-first**: the teaching angle is not the primary
deliverable. This may or may not satisfy the category as formally
interpreted — **confirm with the thesis coordinator before finalizing.**
Fallback if needed: a short appendix (a few paragraphs, not a chapter)
sketching one lab-exercise use of the finished platform, without pulling
weight away from the platform framing.

## 2. The problem, stated narrowly

Pathology and spatial-omics annotation workflows routinely combine ad hoc
scripts, notebooks, and standalone external tools (segmentation models,
classifiers, custom pipelines), with no standard contract for how such a
tool should be invoked from a viewer, and typically no durable record of
*which* tool version, with *which* parameters, produced a given
annotation. **Reproducibility of the annotation-generation step itself**
is the specific, narrow, tractable problem this project addresses.

**Explicitly not the claim being made**: this is not a better or faster
general-purpose spatial-omics visualization or interactive-analysis
platform than existing tools in that space (see section 6). The
contribution is architectural/interoperability-focused, not a rendering
or analysis-UX contribution.

## 3. Scope — four components, working end-to-end on real data

1. **Viewer**: Qt/vispy pyramidal pan/zoom, annotation overlay, per-layer
   visibility toggle. (Subset of the full COMPASS viewer design — see
   `decisions.md` section 6-7 for the fuller rationale this is scoped
   down from.)
2. **Xenium connector**: imports a public 10x Genomics Xenium dataset into
   COMPASS's schema — morphology image to pyramidal raster, cell
   boundaries to polygon annotation layer, per-cell transcript counts to
   sparse per-object attributes. Demonstrates real multiplexed data
   flowing through the full pipeline, not a synthetic stand-in.
3. **Docker tool hook**: a minimal, deliberately bespoke contract (not
   aligned to an existing standard such as BioImage.IO for this thesis —
   see section 5 for the trade-off).
   - *Input*: a mounted directory containing the image region as a plain,
     tool-agnostic file (e.g. TIFF tile) plus a small `params.json`
     (region bounds in source-image coordinates, mpp, tool-specific
     parameters).
   - *Output*: a mounted directory where the container writes one
     annotation file in a simple, documented interchange format (e.g. a
     GeoJSON FeatureCollection or flat Parquet of geometry + attributes)
     — deliberately not COMPASS's internal schema, so a tool author never
     needs to know COMPASS internals.
   - COMPASS's importer reads the interchange file, converts it into the
     internal annotation store, and writes a provenance record alongside
     the resulting annotation group.
4. **Provenance record**, attached to every annotation group produced via
   the tool hook: container image digest (not just tag — digest is what
   makes it verifiable), the exact `params.json` used, wall-clock
   runtime, timestamp, and the COMPASS version that ran it.

**Worked demonstration requirement**: at least one real external tool (a
simple nucleus segmentation or basic classifier — doesn't need to be
novel) wrapped in Docker and run through the hook against the imported
Xenium data, end to end. This is what makes the architecture chapter a
demonstrated system rather than a design document.

## 4. Deliverables

- Working software (viewer + connector + tool hook + provenance
  tracking), open source.
- Thesis text: architecture/decision-log chapter (what was built and why,
  in the same rationale-plus-rejected-alternatives style as COMPASS's own
  `decisions.md`), the tool-hook contract as a documented specification,
  a worked-example walkthrough (the end-to-end Xenium-import-then-external-tool
  demonstration), and a limitations/future-work section.
- Future-work section should explicitly name: BioImage.IO alignment,
  provenance-record verification/replay tooling, and the teaching
  application as a brief mention (one paragraph), not a chapter.

## 5. Design decisions already made, with rationale

- **Docker tool-hook contract is bespoke, not BioImage.IO-aligned, for
  this thesis.** Considered and deferred: BioImage.IO's model-packaging
  spec is the closest existing convention for "standardized
  image-in/prediction-out containers" in this space, but adopting it adds
  scope beyond a single bachelor term. State this explicitly in the
  thesis as a considered trade-off (tractability vs. standards alignment),
  with BioImage.IO alignment named as natural future work — a deliberate
  engineering decision, not an oversight.
- **No direct comparison to MilliMap or PRISM in the thesis framing.**
  Both are close, recent, well-funded prior art (MilliMap: bioRxiv,
  May 2026, closed-loop interactive statistical analysis; PRISM: bioRxiv,
  Dec 2024, napari-spatialdata-based multiplexed-tissue analysis). Rather
  than inviting a comparison the thesis can't credibly win at this scope,
  HistoCompass is framed in different, more accurate vocabulary:
  reproducible annotation ingestion and containerized tool
  interoperability, not interactive statistical analysis. This is a
  genuinely different category of contribution (data-engineering /
  interoperability vs. analysis-UX), and keeping the scope narrow (per
  section 3) is what keeps this framing honest rather than evasive.
- **Xenium chosen as the proof-of-concept data source** because its
  native output (morphology image + cell segmentation + per-cell
  transcript counts) already matches COMPASS's core data model
  (raster + map/polygon annotations + sparse per-object attributes)
  without any artificial data preparation — see `decisions.md` section 1
  and 4 for why that data model was chosen in the first place.

## 6. Competitive landscape (for the thesis's related-work section)

| Tool | Relevant strength | Why not directly comparable to HistoCompass's scope |
|---|---|---|
| QuPath | Mature, huge plugin ecosystem, excellent pyramidal image handling | JVM-based; extensibility is Groovy/Java scripts, not a language-agnostic containerized contract |
| napari-spatialdata | Python-native, scverse-ecosystem-integrated | Requires SpatialData/AnnData (deliberately avoided in COMPASS — see `decisions.md` section on connector isolation); plugin model is Python-API-coupled |
| Vitessce | Rich linked-view visualization, widely used in spatial-omics community | Web/browser-based; configuration via code, not a no-code GUI; displays precomputed results, doesn't trigger new analysis |
| TissUUmaps | GPU-accelerated, handles 10^7+ points (exceeds Vitessce in published benchmarks); v4 (2026) moving toward OME-Zarr and community plugin architecture | Browser-based rendering even in its "native desktop" packaging; general-purpose viewer, not built around a provenance-tracked external-tool contract |
| MilliMap (May 2026) | Closed-loop interactive statistical analysis, real-time linked views | Different problem category — interactive analysis/statistics, not reproducible annotation ingestion; Python-API-coupled, not a language-agnostic container contract |
| PRISM (Dec 2024) | Modular multiplexed-tissue analysis on napari-spatialdata | Same SpatialData/napari coupling as above |
| Xenium Explorer (10x Genomics) | Polished, vendor-optimized viewing of Xenium data specifically | Closed-source, single-vendor, no extensibility |

## 7. Milestone shape (single-term bachelor scope)

1. Viewer subset (raster pan/zoom + static annotation overlay, no tool
   hook yet)
2. Xenium connector (import to visible in viewer)
3. Annotation store + layer toggle, wired to real imported data
4. Docker tool-hook contract + runner (mechanism only, tested first with
   a trivial "echo back a fixed annotation" container)
5. Provenance recording
6. One real external tool wrapped and run through the hook, end to end
7. Write-up (architecture chapter, contract specification, worked
   example, limitations/future work)

## 8. Open items

- [ ] Confirm formal category fit with thesis coordinator (section 1).
- [ ] Decide target/example external tool for the worked demonstration
      (a simple, existing nucleus-segmentation model is a reasonable
      default — doesn't need to be novel).
- [ ] Decide interchange annotation format for the tool-hook output
      (GeoJSON vs. flat Parquet — GeoJSON is more universally readable by
      third-party tool authors; Parquet is more efficient at scale but
      less accessible as a contract for simple external tools).
