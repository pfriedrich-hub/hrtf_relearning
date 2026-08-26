# Reading list: source spectrum × interaural spectral difference

Organised by the job each paper does for *this* experiment, not by topic. For
the wider ill-posed-problem literature see the memo in
`source_spectrum_isd_design.md` §1.

**Verification status.** Every citation below was checked to exist, with its
venue, through search. Where I was able to open the source and confirm what it
says, the entry is marked **[read]**. Everything else is from my own knowledge
of these papers and should be checked before it goes into a manuscript — I have
been wrong about details of papers I could not open.

---

## 1. The one that reframes the whole experiment

**Baumgartner, R., Majdak, P., & Laback, B. (2014).** Modeling sound-source
localization in sagittal planes for human listeners. *JASA* 136, 791–802.
[link](https://pubs.aip.org/asa/jasa/article-abstract/136/2/791/842722/Modeling-sound-source-localization-in-sagittal)

**Baumgartner, R., Majdak, P., & Laback, B. (2014).** Acoustic and non-acoustic
factors in modeling listener-specific performance of sagittal-plane sound
localization. *Front. Psychol.* 5:319.
[open access](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2014.00319/full)
**[read]**

The Frontiers paper states the model's treatment of the two ears explicitly:
it applies "a binaural weighting function (Morimoto, 2001; Macpherson and
Sabin, 2007), which reduces the contribution of the contralateral ear with
increasing lateral angle of the target sound", and "the binaural weighting
function is applied to each monaural SI, and the sum of the weighted monaural
SIs yields the binaural SI."

**Why this matters more than anything else on the list:** the standard
computational model of sagittal-plane localization combines the ears by
*weighted summation of monaural spectral indices*. There is **no interaural
difference term in it at all**. So it predicts that removing the interaural
spectral difference has no source-spectrum-specific consequence — which makes
our interaction a genuine test between two model classes rather than a
manipulation check:

| account                                                                 | prediction for the interaction                    |
| ----------------------------------------------------------------------- | ------------------------------------------------- |
| weighted monaural summation (Baumgartner / Morimoto / Macpherson–Sabin) | **I ≈ 0** — no ISD term exists to remove          |
| interaural spectral difference read-out                                 | **I > 0** — the source-cancelling channel is gone |

A null is therefore a positive result for the standard model, not a failed
experiment. Say so in the preregistration.

Note also what the model does at the midline: with equal weights on both ears,
the diotic condition replaces the average of two slightly different monaural
SIs with one ear's SI at full weight. That is not obviously worse — it could be
marginally *better*. Worth simulating with `baumgartner2014` in the AMT on the
two SOFAs before running anyone; it is a free prediction and it costs an
afternoon.

## 2. Where the binaural weighting function comes from

**Morimoto, M. (2001).** The contribution of two ears to the perception of
vertical angle in sagittal planes. *JASA* 109, 1596–1603.
[link](https://pubs.aip.org/asa/jasa/article/109/4/1596/549990/The-contribution-of-two-ears-to-the-perception-of)

**Macpherson, E. A., & Sabin, A. T. (2007).** Binaural weighting of monaural
spectral cues for sound localization. *JASA* 121, 3677–3688.
[link](https://pubs.aip.org/asa/jasa/article-abstract/121/6/3677/537202/Binaural-weighting-of-monaural-spectral-cues-for)

Both ask *how much each ear contributes*, by occluding or by putting different
spectra at the two ears — neither asks whether listeners compute L−R. That gap
is the hole this experiment is aimed at. Read them for the weighting function
and for the methodological precedent of putting mismatched spectra at the two
ears, which is what the diotic condition does in the limit.

## 3. Is there an interaural spectral difference cue at all?

**Searle, C. L., Braida, L. D., Davis, M. F., & Colburn, H. S. (1976).** Model
for auditory localization. *JASA* 60, 1164–1175.
[link](https://pubs.aip.org/asa/jasa/article-abstract/60/5/1164/635990/Model-for-auditory-localization)
The original proposal that *interaural pinna disparity* is one of the cues.
This is the ancestor of the hypothesis being tested.

**Jin, C., Corderoy, A., Carlile, S., & van Schaik, A. (2004).** Contrasting
monaural and interaural spectral cues for human sound localization. *JASA* 115,
3124–3141.
[link](https://pubs.aip.org/asa/jasa/article-abstract/115/6/3124/545926/Contrasting-monaural-and-interaural-spectral-cues)
The closest existing test of the same question. Read it first and read it
properly — if it already answers what we are asking, the experiment needs
re-aiming, and if it does not, its design will tell us why not.

**Hofman, P. M., & Van Opstal, A. J. (2003).** Binaural weighting of pinna cues
in human sound localization. *Exp. Brain Res.* 148, 458–470.
[link](https://link.springer.com/article/10.1007/s00221-002-1320-5)

**Van Wanrooij, M. M., & Van Opstal, A. J. (2007).** Sound localization under
perturbed binaural hearing. *J. Neurophysiol.* 97, 715–726.
[open access](https://journals.physiology.org/doi/full/10.1152/jn.00260.2006)

**Wightman, F. L., & Kistler, D. J. (1997).** Monaural sound localization
revisited. *JASA* 101, 1050–1063. The demonstration that "monaural"
localization collapses once the source spectrum is unknown — the closest
existing result to our in-band condition, and the reason we expect an in-band
cost at all.

## 4. The threat to the experiment

**Hartmann, W. M., & Wittenberg, A. (1996).** On the externalization of sound
images. *JASA* 99, 3678–3688.
[link](https://pubs.aip.org/asa/jasa/article-abstract/99/6/3678/751322/On-the-externalization-of-sound-images)
Manipulations confirmed from the AMT's `data_hartmann1996` **[read]**: one
condition zeroes the ILDs up to harmonic *n′*; the other preserves the
interaural spectral level differences while flattening the right-ear HRTF.
So they separated exactly the two things this experiment separates, and
measured externalization rather than elevation.

**This is the main risk to the diotic condition.** The received wisdom from
this paper is that the *interaural* spectral differences carry externalization
while monaural spectral detail does not — and zeroing the ISD is precisely what
our manipulation does. If externalization collapses in the diotic blocks, an
elevation deficit there is confounded, and the interaction cannot be read as
being about source-spectrum inference. Two mitigations, both already in the
protocol: the externalization rating after every block, and the fact that our
manipulation is confined to the pinna band rather than the low harmonics
Hartmann and Wittenberg zeroed. **Verify the direction of their result from the
paper itself before relying on either of those** — I could not open it.

Follow-ups worth having to hand:
- **Modeling perceived externalization of a static, lateral sound image.**
  *Acta Acustica* (2020).
  [open access](https://acta-acustica.edpsciences.org/articles/aacus/full_html/2020/05/aacus200024/aacus200024.html)
- **The role of spectral detail in the binaural transfer function on perceived
  externalization in a reverberant environment.** *JASA* 139, 2992 (2016).
  [link](https://pubs.aip.org/asa/jasa/article/139/5/2992/838291/The-role-of-spectral-detail-in-the-binaural)

## 5. Is 1.2 dB of ISD even usable?

The premise of the experiment is that AS's midline ISD carries a real cue —
1.21 dB rms in the cue band on the 0.5–16 kHz axis, 5.92 on the lab's 3.5–16 kHz
axis (see [[project-midline-isd-and-az-expansion]]). Whether that is above
threshold is a psychophysics question with its own literature:

**Supin, A. Y., et al. (1999).** Ripple depth and density resolution of rippled
noise. *JASA*. [link](https://pubmed.ncbi.nlm.nih.gov/10573895/)

**Effect of level on spectral-ripple detection threshold for listeners with
normal hearing and hearing loss.**
[open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443170/)

**Spectral aliasing in an acoustic spectral ripple discrimination task.**
[open access](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7112708/) — a
methodological warning about ripple stimuli that is worth reading given what
our own stimulus does.

Caveat when reading these across: ripple *detection* thresholds are measured
monaurally on broadband noise, and the quantity we care about is an *interaural*
spectral difference under a roving source. The numbers do not transfer
directly, but they bracket the plausible range and tell us whether 1.2 dB is
near threshold or comfortably above it. If it is near threshold, the experiment
is underpowered by construction and the manipulation should be run on subjects
with deeper ISDs (FS, at 2.05 dB, rather than AS).

## 6. The method for the exploratory reverse correlation

**Hofman, P. M., & Van Opstal, A. J. (2002).** Bayesian reconstruction of sound
localization cues from responses to random spectra.
[link](https://pubmed.ncbi.nlm.nih.gov/11956811/)

**Reconstructing spectral cues for sound localization from responses to rippled
noise stimuli.** *PLOS ONE* (2017).
[open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC5363849/)

**Zonooz, B., Arani, E., Körding, K. P., & Van Opstal, A. J. (2019).** Spectral
weighting underlies perceived sound elevation. *Sci. Rep.*
[open access](https://www.nature.com/articles/s41598-018-37537-z)

**Zonooz, B., & Van Opstal, A. J. (2019).** Learning to localise
weakly-informative sound spectra with and without feedback. *Sci. Rep.*
[open access](https://www.nature.com/articles/s41598-018-36422-z)

These are the methods papers for recovering a spectral weight function from
responses to random spectra. With `rms_cue > 0` our stimulus finally excites
the cue band, so the same analysis becomes available inside it — see the
exploratory section of the protocol.

## 7. Background that shaped the stimulus, not the design

**Zakarauskas, P., & Cynader, M. S. (1993).** A computational theory of
spectral cue localization. *JASA* 94, 1323–1331.

**Hofman, P. M., & Van Opstal, A. J. (1998).** Spectro-temporal factors in
two-dimensional human sound localization. *JASA* 103, 2634–2648.
[PDF](https://www.mbfys.ru.nl/~johnvo/InfoInternships/StartupPapers/hofman_jasa98.pdf)

**Macpherson, E. A., & Middlebrooks, J. C. (2003).** Vertical-plane sound
localization probed with ripple-spectrum noise. *JASA* 114, 430–445.
[link](https://pubmed.ncbi.nlm.nih.gov/12880054/) — already cited in
`stimulus.py`; the direct ancestor of our ripple conditions.

**Baumgartner, R., Barumerli, R., Brands, B., & Majdak, P. (2026).** Short-Term
Statistical Learning Mitigates the Ill-Posed Problem of Sound Localization.
*Trends in Hearing*.
[open access](https://journals.sagepub.com/doi/full/10.1177/23312165261465030)
The source of our within-block learning prediction.

**Lladó, P., Majdak, P., Barumerli, R., & Baumgartner, R. (2025).** Spectral
Weighting of Monaural Cues for Auditory Localization in Sagittal Planes.
*Trends in Hearing*.
[open access](https://journals.sagepub.com/doi/full/10.1177/23312165251317027)

**Ideal-observer model of human sound localization of sources with unknown
spectrum.** *Sci. Rep.* (2025).
[open access](https://www.nature.com/articles/s41598-025-91001-3)

**Barumerli, R., Majdak, P., Baumgartner, R., Reijniers, J., & Geronazzo, M.
(2023).** A Bayesian model for human directional localization of broadband
static sound sources. *Acta Acustica*.
[open access](https://acta-acustica.edpsciences.org/articles/aacus/full_html/2023/01/aacus210056/aacus210056.html)

---

## Reading order, if time is short

1. Baumgartner Frontiers 2014 §binaural weighting — 20 minutes, and it changes
   how the experiment should be framed.
2. Jin et al. 2004 — the closest prior test; read before finalising the design.
3. Hartmann & Wittenberg 1996 — the threat.
4. Macpherson & Sabin 2007 — the precedent for mismatched ears.
5. Macpherson & Middlebrooks 2003 — what in-band ripple does to responses.
