# Methodology

## Scope

UbiStruct-QA was used by CASP17 group `ubi` to score supplied
models for H2441, H2444, T2461, T2463 and T2464. Quality estimates for all five
targets were submitted to CASP. The three target classes use a
shared parser and output contract but separate score definitions.

Scores are relative estimates. They are conservatively bounded because no
target-native calibration data were available during prediction.

## Immune complexes

H2441 was treated as an antigen/enzyme plus VHH; H2444 as an antigen plus Fab
heavy and light chains. Candidate chains are mapped by canonical identifiers where available,
otherwise by expected length. Interfaces use representative-atom contacts at 10 Å, tight
contacts at 6 Å, completeness, interface-size plausibility and CA-clash
penalties.

The interface core is:

- 48% family-balanced contact-support percentile;
- 18% parsed interface confidence;
- 17% native interface confidence when available;
- 3% actifpTM;
- 8% completeness; and
- 6% interface-size plausibility.

Penalties are then applied for absent, sparse, excessively dense or clashing
interfaces. H2444 interface weights are AB 0.50, AC 0.33 and BC 0.17.

The whole-model score before antibody blending is:

```text
0.72 * weighted_interface
+ 0.10 * model_confidence
+ 0.08 * completeness
+ 0.10 * antigen_reference_available_pair_Calpha_distance_score
```

The final value retains 97% of this score and adds 3% antibody-template
distance agreement. The calculation uses the lDDT thresholds 0.5, 1, 2 and
4 Å on available Cα reference pairs; missing pairs are skipped. This differs
from the original all-atom lDDT definition. Antigen references were PDB 4RL2
and 6SVK. NanoBodyBuilder2 and ABodyBuilder2 from ImmuneBuilder supplied the
VHH and Fv references. These references assess the individual components; they cannot confirm their
relative placement in the complex.

CASP models are grouped by submitting group and MassiveFold models by generator
family. Source contributions are balanced, and duplicate support fingerprints
are collapsed before averaging.

## Peptide assemblies

T2463 was scored as an amyloid assembly. Its physical term is:

- 28% cross-beta geometry;
- 14% repeat consistency;
- 18% interface-edge quality;
- 14% repeated backbone hydrogen bonding;
- 10% contact-graph regularity;
- 6% packing;
- 6% completeness; and
- 4% clash avoidance.

Before final rescaling, its score is 75% physical evidence, 15% informative predictor confidence
and 10% family-balanced, duplicate-collapsed cluster support.

The public PDB entry title for 9SR1, recorded during analysis, motivated
the use of helical geometry for T2464. It was metadata, not released target coordinates. Its physical term
is 32% alpha geometry, 18% repeat consistency, 20% interface-contact quality,
10% contact-graph regularity, 10% packing, 6% completeness and 4% clash
avoidance. Before rescaling, its score is 80% physical evidence, 15% confidence and 5%
cluster support. The exact-sequence cross-beta structure PDB 7QV6 is retained
only as an alternate-polymorph diagnostic and receives zero final-score weight.

## Designed A24 cage

T2461 uses no cross-model consensus. Its physical score contains:

- 21% cage contact-graph closure and shell geometry;
- 20% interface quality and repeatability;
- 13% subunit structural repeatability;
- 11% hard-clash avoidance;
- 10% A24/sequence completeness; and
- 25% label-invariant family geometry from the A24 homolog PDB 6FDB.

Before final rescaling, the score is 84% physical evidence and 16% informative predictor
confidence. The 6FDB term represents family-level cage geometry, not an exact
residue-level template.

## Confidence and repeated models

Confidence handling differs by scorer. The immune parser reads B factors as
confidence without a variance-based placeholder check; missing predictor-supplied
interface confidence can fall back to the same parsed evidence. The peptide
scorer neutralizes its primary parsed term when uninformative, but its secondary
fallback can retain the raw constant value. The cage scorer neutralizes constant
fields. The code name `native` refers to predictor-supplied confidence, never
experimental native-structure accuracy.

Duplicate group-level contact vectors and cluster distributions are collapsed.
Peptide clusters are nevertheless fitted to individual model records first;
coordinate repetition can affect centroids, centrality and within-group counts.
Immune within-group frequencies retain sampling dependence too. Distinct-interface
selection handles some chain-swap equivalence, but the workflow is not a universal
coordinate-deduplication algorithm. Weight sensitivity and leave-family-out
analyses measure ranking dependence, not native accuracy.

## Reproduction boundaries

The immune H2444 heavy-light (BC) consensus term uses absolute support rather
than its cohort percentile. The antigen-heavy and antigen-light terms retain
percentile support. Reproducing the historical rankings requires the same model
pool, identifiers, coordinates, confidence sidecars and embedded reference
constants. Dependency and numerical-library changes may affect peptide clustering.

See [scientific references](references.md) for the original methods and structures.
