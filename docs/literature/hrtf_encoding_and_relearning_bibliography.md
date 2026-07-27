# HRTF Encoding, Spectral-to-Spatial Mapping, and Relearning — Annotated Bibliography

Compiled 2026-07-24. Two threads: (A) how HRTF/spectral cues are neurally encoded and mapped to spatial locations, and (B) how that mapping is relearned when the cues change. Section C covers the computational models and training paradigms that connect the two — most relevant to the VR/pybinsim training arm of this project.

Annotations summarize the core claim and why it matters for the relearning work. Citation details were reconstructed from search; verify volume/page numbers against the DOI before quoting in a manuscript.

---

## A. Neural encoding of spectral cues and their mapping to space

**Middlebrooks, J. C., & Green, D. M. (1991). Sound localization by human listeners. *Annual Review of Psychology*, 42, 135–159.**
The canonical framing of the two-channel picture: azimuth from interaural time/level differences, elevation and front/back from the spectral shape imposed by the pinna. Introduces the idea that the received spectrum is compared against a stored set of directional transfer functions to read out elevation. Still the reference point for the "spectral-to-spatial" mapping stage. https://pubmed.ncbi.nlm.nih.gov/2018391/

**Reiss, L. A. J., & Young, E. D. (2005). Spectral edge sensitivity in neural circuits of the dorsal cochlear nucleus. *Journal of Neuroscience*, 25(14), 3680–3691.**
DCN type IV neurons respond to the *rising spectral edge* of a notch aligned near best frequency rather than to the notch trough itself. Direct physiological support for edge-based rather than trough-based cue extraction — relevant to the group-delay/rising-edge feature detection used in this project. https://www.jneurosci.org/content/25/14/3680

**Davis, K. A., Ramachandran, R., & May, B. J. (2003). Auditory processing of spectral cues for sound localization in the inferior colliculus. *Journal of the Association for Research in Otolaryngology*, 4(2), 148–163.**
Type O units in the central IC — the main target of ascending DCN projections — inherit spectral-edge sensitivity, positioning the IC as the midbrain stage where notch/edge information is integrated with binaural cues. Bridges DCN edge coding and higher spatial representations. https://link.springer.com/article/10.1007/s10162-002-2002-5

**Schnupp, J. W. H., King, A. J., & Carlile, S. (1998). Altered spectral localization cues disrupt the development of the auditory space map in the superior colliculus of the ferret. *Journal of Neurophysiology*, 79(2), 1053–1069.**
Removing the pinna/concha in infancy yields SC units with bilobed, broadly tuned, mis-aimed spatial responses; the space map does not fully compensate from binaural cues alone. Establishes that outer-ear spectral cues are constitutive of the space map, not just a refinement. https://journals.physiology.org/doi/full/10.1152/jn.1998.79.2.1053

**Hyde, P. S., & Knudsen, E. I. (2002). The optic tectum controls visually guided adaptive plasticity in the owl's auditory space map. *Nature*, 415, 73–76.**
In the barn owl, visual (optic tectum) signals instruct plasticity of the auditory space map. The clearest model system for how a cross-modal teaching signal recalibrates an auditory spatial representation — conceptual backbone for visual-feedback training paradigms in humans. https://www.nature.com/articles/415073a

**Trapeau, R., & Schönwiesner, M. (2018). The encoding of sound source elevation in the human auditory cortex. *Journal of Neuroscience*, 38(13), 3252–3265.**
fMRI voxel tuning functions for elevation depend on *listening experience* with spectral cues, not just their physical form, and shift with perceptual adaptation. Direct human-cortex evidence that the spectral-to-spatial map is plastic and experience-dependent. https://www.jneurosci.org/content/38/13/3252

---

## B. Adaptation and relearning of spectral cues

**Hofman, P. M., Van Riswick, J. G. A., & Van Opstal, A. J. (1998). Relearning sound localization with new ears. *Nature Neuroscience*, 1(5), 417–421.**
The foundational mold study: pinna molds abolish elevation performance, which recovers over weeks; crucially, learning the new cues does not overwrite the original map — subjects localize normally the instant molds are removed. Establishes coexistence of two mappings and long-timescale relearning. https://www.nature.com/articles/nn0998_417

**Van Wanrooij, M. M., & Van Opstal, A. J. (2005). Relearning sound localization with a new ear. *Journal of Neuroscience*, 25(22), 5413–5424.**
Extends the mold paradigm with systematic tracking of the recovery time course and the spectral basis of the adjustment. A core methodological reference for how to quantify relearning of elevation cues. https://www.jneurosci.org/content/25/22/5413

**Trapeau, R., Aubrais, V., & Schönwiesner, M. (2016). Fast and persistent adaptation to new spectral cues for sound localization suggests a many-to-one mapping mechanism. *Journal of the Acoustical Society of America*, 140(2), 879–890.**
Adaptation to molds is faster and more persistent than classically assumed, and old + new cues both remain usable — arguing for a many-to-one spectral-to-spatial mapping rather than a single remapped template. Central to how retention and interference are conceptualized in this project. https://pubs.aip.org/asa/jasa/article/140/2/879

**Zonooz, B., & Van Opstal, A. J. (2019). Differential adaptation in azimuth and elevation to acute monaural spatial hearing after training with visual feedback. *eNeuro*, 6(6), ENEURO.0219-19.2019.**
With acute monaural plugging, visual-feedback training rapidly reweights level/spectral/binaural cues to recover *azimuth* — at the cost of *elevation* accuracy. Shows adaptation is cue-reweighting under a shared objective, and that azimuth and elevation trade off. https://www.eneuro.org/content/6/6/ENEURO.0219-19.2019

**Zonooz, B., Arani, E., & Van Opstal, A. J. (2018). Learning to localise weakly-informative sound spectra with and without feedback. *Scientific Reports*, 8, 17933.**
Listeners gradually learn to read elevation from spectrally *degraded* stimuli carrying only weak but consistent elevation cues; feedback accelerates but is not strictly required. Speaks directly to cue salience thresholds — how much spectral structure the mapping needs to latch onto. https://www.nature.com/articles/s41598-018-36422-z

**Carlile, S. (2014). The plastic ear and perceptual relearning in auditory spatial perception. *Frontiers in Neuroscience*, 8, 237.**
Review synthesizing mold/altered-cue studies into a plasticity framework: timescales, retention, the role of active vs. passive exposure, and coexisting maps. Good orienting read for framing the project's relearning questions. https://www.frontiersin.org/articles/10.3389/fnins.2014.00237

**Mendonça, C. (2014). A review on auditory space adaptations to altered head-related cues. *Frontiers in Neuroscience*, 8, 219.**
Complementary review cataloguing adaptation paradigms (molds, non-individual HRTFs, VR) and the factors that govern adaptation rate and completeness. Useful for situating a chosen training protocol against prior designs. https://www.frontiersin.org/articles/10.3389/fnins.2014.00219

**"Happy new ears": Rapid adaptation to novel spectral cues in vertical sound localization. (2024). *iScience*, 27(12), 111308.**
Recent demonstration of rapid (single-session-scale) adaptation to novel spectral cues in the vertical plane, pushing the fast end of the adaptation-rate distribution. Relevant benchmark for how quickly a well-designed training loop can move the map. https://www.cell.com/iscience/fulltext/S2589-0042(24)02533-1

---

## C. Computational models and training/VR paradigms

**Baumgartner, R., Majdak, P., & Laback, B. (2014). Modeling sound-source localization in sagittal planes for human listeners. *Journal of the Acoustical Society of America*, 136(2), 791–802.**
The standard template-comparison model (in AMT as `baumgartner2014`): incoming DTF-filtered spectrum is compared to an internal template set to yield a probabilistic polar-angle response. The template set is explicitly framed as learned via cross-modal feedback — a computational instantiation of the relearning target. Directly usable to predict/score localization under modified HRTFs. https://pmc.ncbi.nlm.nih.gov/articles/PMC4582445/

**Lladó, P., Majdak, P., Barumerli, R., & Baumgartner, R. (2025). Spectral weighting of monaural cues for auditory localization in sagittal planes. *Trends in Hearing*, 29.**
Compares five spectral-weighting schemes within the sagittal-plane model, reparametrized for fair comparison — quantifying which frequency bands the mapping actually weights. Relevant if you want a principled weighting for cue-editing manipulations. https://doi.org/10.1177/23312165251317027

**Iida, K., Itoh, M., Itagaki, A., & Morimoto, M. (2007). Median plane localization using a parametric model of the head-related transfer function based on spectral cues. *Applied Acoustics*, 68(8), 835–850.**
A parametric HRTF built from just the first spectral peak (P1) and first two notches (N1, N2) reproduces measured-HRTF elevation accuracy. The empirical justification for reducing HRTFs to a few notch/peak features — the premise behind the project's notch-editing manipulations. https://www.sciencedirect.com/science/article/abs/pii/S0003682X07000151

**Parseihian, G., & Katz, B. F. G. (2012). Rapid head-related transfer function adaptation using a virtual auditory environment. *Journal of the Acoustical Society of America*, 131(4), 2948–2957.**
Gamified "hot-and-cold" VR training (~12-min sessions) drives fast adaptation to non-individual HRTFs, with retention. The template for accelerated, active, closed-loop training — closely matched to this project's VR training design. https://asa.scitation.org/doi/10.1121/1.3687448

**Steadman, M. A., Kim, C., Lestang, J.-H., Goodman, D. F. M., & Picinali, L. (2019). Short-term effects of sound localization training in virtual reality. *Scientific Reports*, 9, 18284.**
Modern VR training study measuring short-term gains and the contribution of gamification/active listening to non-individual-HRTF adaptation. Practical reference for VR training design, feedback structure, and outcome metrics. https://www.nature.com/articles/s41598-019-54811-w

**Kacelnik, O., Nodal, F. R., Parsons, C. H., & King, A. J. (2006). Training-induced plasticity of auditory localization in adult mammals. *PLoS Biology*, 4(4), e71.**
In adult ferrets with a monaural plug, active, reward-based training with sensory feedback drives recovery of localization far beyond passive exposure. The animal-model case that *training*, not mere exposure, is the active ingredient — motivating closed-loop feedback protocols. https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0040071

---

## Threads worth pulling

Retention without interference — Hofman 1998, Van Wanrooij & Van Opstal 2005, and Trapeau 2016 converge on coexisting old/new mappings ("many-to-one"), which frames whether your cue manipulations create a genuinely new map or reweight the existing one.

Edges vs. troughs — Reiss & Young 2005 and Davis 2003 argue the brainstem encodes rising spectral edges, aligning with the project's group-delay/rising-edge notch detection; Iida 2007 and Lladó 2025 give the perceptual/weighting counterpart.

Training as the active ingredient — Kacelnik 2006, Parseihian & Katz 2012, Steadman 2019, and Zonooz 2019 all point to active closed-loop feedback (ideally cross-modal, cf. Hyde & Knudsen 2002) as what accelerates relearning — the design premise of the VR training arm.

Cue reweighting and trade-offs — Zonooz & Van Opstal 2019 shows azimuth recovery can cost elevation accuracy, a caution for interpreting one-plane training outcomes.
