"""Generate docs/pipeline_schematic.pdf — the HRIR chain, recording to testing.

Regenerate after any change to the chain:

    python hrtf_relearning/docs/make_pipeline_schematic.py

Page 1 is the signal chain and shows where ILD and ITD are set. Page 2 is the
experiment side, the acceptance numbers, and the v1/v2 provenance.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / 'pipeline_schematic.pdf'

INK = '#1b1b1f'
MEASURED = '#2f6f9f'      # came off the rig
MODEL = '#c8791f'         # spherical head model
MODIFY = '#8d4c9e'        # cue manipulation
CHECK = '#3f8f5f'         # QC
RENDER = '#6b6b76'        # playback plumbing
FADE = '#f4f4f6'
NOTE = '#c1443c'


def box(ax, x, y, w, h, text, colour=INK, fill=FADE, fs=8.0, weight='normal',
        lw=1.1, style='round,pad=0.35'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                linewidth=lw, edgecolor=colour, facecolor=fill, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=INK,
            zorder=3, weight=weight, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, colour=INK, lw=1.2, style='-|>'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=11, linewidth=lw, color=colour,
                                 zorder=1, shrinkA=1.5, shrinkB=1.5))


def label(ax, x, y, text, fs=7.2, colour=INK, ha='left', style='italic'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=colour,
            style=style, zorder=3, linespacing=1.45)


def blank(fig):
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    return ax


# ---------------------------------------------------------------- page 1
def page_chain(pdf):
    fig = plt.figure(figsize=(16.5, 10.5))
    ax = blank(fig)

    ax.text(4, 96.5, 'HRIR pipeline — recording to testing', fontsize=15,
            weight='bold', color=INK)
    ax.text(4, 93.4, 'Only the az = 0 arc is measured. Everything else is that arc '
                     'times a spherical-head model — so 19 measurements, not 475.',
            fontsize=9, color='#55555f')

    for x, c, t in [(64, MEASURED, 'measured'), (73.5, MODEL, 'model'),
                    (81, MODIFY, 'modification'), (91, CHECK, 'QC')]:
        ax.add_patch(FancyBboxPatch((x, 95.7), 1.6, 1.1, boxstyle='round,pad=0.15',
                                    linewidth=1.1, edgecolor=c, facecolor='white'))
        ax.text(x + 2.3, 96.3, t, fontsize=7.5, va='center', color='#55555f')

    # --- acquisition -----------------------------------------------------
    box(ax, 20, 87.5, 26, 5.2,
        'sweep recording   ·   dome + in-ear mics   ·   MESM speaker EQ\n'
        'az = 0 only, 19 elevations  −37.5° … +37.5°        → npz',
        MEASURED, '#eaf1f7', 8.2)
    box(ax, 52, 87.5, 20, 5.2,
        'reference recording\nmic at head centre, head absent   → npz',
        MEASURED, '#eaf1f7', 8.2)
    arrow(ax, 20, 84.9, 20, 82.0)
    arrow(ax, 52, 84.9, 52, 83.3, MEASURED); arrow(ax, 52, 83.3, 30.2, 83.3, MEASURED, style='-')
    arrow(ax, 30.2, 83.3, 30.2, 82.0, MEASURED)

    for cx, name, sub in [(13, 'compute\\_ir', 'deconvolve sweeps'),
                          (27.5, 'equalize', 'inverse-filter, window,\nonset → 1 ms, crop'),
                          (43, 'lowfreq\\_extrapolate', 'model anchors\nbelow 800 Hz')]:
        box(ax, cx, 79.4, 12.4, 5.0, f'$\\bf{{{name}}}$\n{sub}', MEASURED, 'white', 7.6)
    arrow(ax, 19.2, 79.4, 21.3, 79.4); arrow(ax, 33.7, 79.4, 36.8, 79.4)
    arrow(ax, 43, 76.9, 43, 73.6)

    # --- the arc ---------------------------------------------------------
    box(ax, 43, 70.4, 34, 5.2, 'MIDLINE ARC   —   19 measured DTFs',
        MEASURED, '#d6e6f2', 10.5, 'bold', lw=1.8)
    label(ax, 61.5, 70.4, 'read back from the SOFA:\nthe expansion never touches\naz = 0 magnitudes',
          7.2, MEASURED)

    arrow(ax, 43, 67.8, 43, 66.0, MEASURED, style='-')
    arrow(ax, 43, 66.0, 19, 66.0, MEASURED, style='-')
    arrow(ax, 43, 66.0, 67, 66.0, MEASURED, style='-')
    arrow(ax, 19, 66.0, 19, 45.0, MEASURED)
    arrow(ax, 67, 66.0, 67, 63.6)
    label(ax, 20.2, 56, 'native\n(unmodified)', 8, MEASURED, style='normal')

    # --- modify ----------------------------------------------------------
    ax.add_patch(FancyBboxPatch((50.5, 47.0), 33, 14.2, boxstyle='round,pad=0.4',
                                linewidth=1.4, edgecolor=MODIFY, facecolor='#f6eef8', zorder=0))
    ax.text(67, 59.8, 'MODIFY   (the 19 measured directions)', fontsize=9,
            weight='bold', ha='center', color=MODIFY)
    box(ax, 67, 56.0, 30, 3.6,
        '$\\bf{donor\\ selection}$   ranked by detail strength · VSI-dis 0.40 ± 0.05 · ridge < 0.5',
        MODIFY, 'white', 7.4)
    box(ax, 67, 51.6, 30, 3.9,
        '$\\bf{donor\\_detail\\_dtf}$   env₄(own) + detail(donor),  BOTH ears\n'
        'own phase · own broadband energy',
        MODIFY, 'white', 7.4)
    box(ax, 67, 47.2, 30, 3.9,
        '$\\bf{envelope\\_dtf}$   n_keep = 4,  UNTRAINED ear only\n'
        'ERB fit 700 Hz–18 kHz,  AVERAGED over elevation',
        MODIFY, 'white', 7.4)
    arrow(ax, 67, 54.0, 67, 53.6, MODIFY); arrow(ax, 67, 49.6, 67, 49.2, MODIFY)
    label(ax, 84, 47.2, 'averaging is what\nremoves the cue —\nidentical shapes cannot\nencode elevation',
          7.0, MODIFY)
    arrow(ax, 67, 45.2, 67, 42.8, MODIFY)

    # --- QC --------------------------------------------------------------
    box(ax, 67, 40.2, 30, 4.4,
        '$\\bf{qc\\_midline}$   modified arc vs native arc\n'
        'ILD per band · ITD from interaural phase · elevation & azimuth SD',
        CHECK, '#eaf4ee', 7.6, lw=1.5)
    label(ax, 83.5, 40.2, 'complete check:\nthe model adds nothing\nat az = 0, so every\ndeviation starts here',
          7.0, CHECK)
    arrow(ax, 67, 38.0, 67, 35.4, CHECK)
    arrow(ax, 19, 45.0, 19, 35.4, MEASURED)

    # --- expansion -------------------------------------------------------
    box(ax, 43, 31.8, 62, 6.2,
        '$\\bf{expand\\_azimuths\\_with\\_binaural\\_cues}$      19 → 475 directions\n'
        'az −50° … +50°, 25 steps of 4.17°     ·     run once per set,\n'
        'identical parameters',
        MODEL, '#fbf1e3', 8.6, lw=1.6)
    box(ax, 89, 31.8, 17, 6.2, 'spherical head model\nDuda & Martens (1998)\nr = 0.0875 m',
        MODEL, 'white', 7.6)
    arrow(ax, 80.4, 31.8, 74.2, 31.8, MODEL)

    label(ax, 12.5, 26.4,
          '$\\bf{ILD}$   magnitude × |H_sph(az,el,f)| / |H_sph(0,el,f)| ,  per ear', 7.8,
          MODEL, style='normal')
    label(ax, 12.5, 24.0,
          '$\\bf{ITD}$   interaural phase   IPD_sph(az,el,f) − IPD_sph(0,el,f)', 7.8,
          MODEL, style='normal')
    label(ax, 55, 25.2, 'both relative to frontal at the same elevation, both zero at az = 0\n'
                        'ITD is geometry alone → a magnitude-only edit cannot move it,\n'
                        'so native and modified sets share it bit-for-bit',
          7.1, MODEL)

    arrow(ax, 19, 28.7, 19, 21.6, MEASURED); arrow(ax, 67, 28.7, 67, 21.6, MODIFY)

    box(ax, 19, 18.8, 24, 4.4, '$\\bf{<ID>.sofa}$\n475 directions', MEASURED, 'white', 8)
    box(ax, 67, 18.8, 32, 4.4,
        '$\\bf{<ID>\\_donor\\_<D>\\_env4\\_<ear>.sofa}$\n'
        '475 directions  ·  GLOBAL_ModificationParams', MODIFY, 'white', 8)
    label(ax, 84.5, 18.8, 'monaural reduction is\nIN the file, not a\nrender-time switch', 7.0, MODIFY)

    arrow(ax, 19, 16.6, 19, 13.8, RENDER); arrow(ax, 67, 16.6, 67, 13.8, RENDER)

    box(ax, 43, 11.2, 62, 4.4,
        '$\\bf{hrtf2binsim}$    mirror (blocks C/D)  ·  late reverb DRR 20 dB  ·  '
        'DT990 headphone inverse  ·  level → REFERENCE_LEVEL    →  MAT database',
        RENDER, '#f0f0f2', 8.2)
    arrow(ax, 43, 9.0, 43, 6.8, RENDER)
    box(ax, 43, 4.6, 46, 3.6, 'pybinsim  +  head tracker      →      participant',
        RENDER, 'white', 8.4, 'bold')

    ax.text(4, 8.6, 'mirror stays at render time — a channel/source swap commutes\n'
                    'with everything above it, so blocks C and D come off one file.',
            fontsize=7.4, color='#55555f', va='top', linespacing=1.5)

    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- page 2
def page_protocol(pdf):
    fig = plt.figure(figsize=(16.5, 10.5))
    ax = blank(fig)
    ax.text(4, 96.5, 'Protocol, verification, and build provenance', fontsize=15,
            weight='bold', color=INK)

    # --- protocol --------------------------------------------------------
    ax.text(4, 91.5, 'PROTOCOL', fontsize=10, weight='bold', color=INK)
    days = [
        ('Day 1', '#eaf1f7', [
            'status',
            'native reference — binaural, native HRTF, full field ±35°',
            'build donor  (+ pre-stage ranks 1–2 for an in-session swap)',
            'baseline A — trained ear, modified, trained hemifield',
            'baseline D — trained ear, MIRRORED modified, mirrored hemifield']),
        ('Adaptation days  (×N)', '#f6eef8', [
            'PRE test',
            'train — AR game, 90 s games of head-tracked target trials',
            'POST test']),
        ('Final day', '#eaf4ee', [
            'counterbalanced 2×2 (Ear × Side), Williams Latin square',
            'A trained/same        B trained/mirrored',
            'C untrained/same    D untrained/mirrored   ← main']),
    ]
    y = 88
    for title, colour, items in days:
        h = 3.0 + 2.6 * len(items)
        ax.add_patch(FancyBboxPatch((4, y - h), 44, h, boxstyle='round,pad=0.4',
                                    linewidth=1.2, edgecolor='#9a9aa4',
                                    facecolor=colour, zorder=0))
        ax.text(6, y - 2.0, title, fontsize=9, weight='bold', color=INK)
        for k, item in enumerate(items):
            ax.text(7.5, y - 4.6 - 2.6 * k, '·  ' + item, fontsize=8, color=INK)
        y -= h + 2.2
    ax.text(4, y - 0.5, 'localization test  —  sector (7°, 14°), el ±35°, 3 targets/sector,\n'
                        'min separation 20°, noise, midline excluded for one-sided blocks',
            fontsize=8, color='#55555f', va='top', linespacing=1.6)

    # --- provenance ------------------------------------------------------
    ax.text(4, 37, 'BUILD PROVENANCE', fontsize=10, weight='bold', color=INK)
    ax.add_patch(FancyBboxPatch((4, 8.0), 44, 26.0, boxstyle='round,pad=0.4',
                                linewidth=1.0, edgecolor='#c0c0c8',
                                facecolor='#fafafb', zorder=0))
    ax.text(6, 32.4,
            '$\\bf{v1}$ — FS, GS, IR, TS, PF.  Donor detail on the finished 475-\n'
            'direction SOFA; monaural reduction applied again at render time by\n'
            'hrtf2binsim, per direction, on an already-expanded set. Measured\n'
            'after the fact:\n'
            '  ·  the untrained ear kept ~half the elevation cue (2.30 dB\n'
            '     against 4.66 dB unprocessed, 5.7–11.3 kHz)\n'
            '  ·  its 0.2–2 kHz ILD sat 9–12 dB from native, further lateral\n'
            '     than any real direction in that subject\'s own HRTF\n\n'
            '$\\bf{v2}$ — from 2026-08.  Both steps on the 19 measured az = 0\n'
            'DTFs, then one expansion. Envelope fitted on an ERB axis and\n'
            'averaged over elevation.\n\n'
            'v1 subjects are NOT rebuilt: swapping builds mid-cohort would put a\n'
            'discontinuity inside their own pre/post comparison.',
            fontsize=7.5, color=INK, va='top', linespacing=1.62)

    # --- verification ----------------------------------------------------
    ax.text(54, 91.5, 'VERIFIED  —  v2, vs each subject’s own native midline arc',
            fontsize=10, weight='bold', color=INK)
    rows = [
        ('subj', 'ILD bb', 'ILD 0.2–2k', 'ILD 2–16k', 'ITD', 'cue leak'),
        ('FS', '0.000', '3.12', '0.54', '0.0000', '0.000'),
        ('GS', '0.000', '1.47', '0.95', '0.0000', '0.000'),
        ('IR', '0.000', '4.03', '1.51', '0.0000', '0.000'),
        ('TS', '0.000', '1.04', '0.97', '0.0000', '0.000'),
        ('PF', '0.000', '1.46', '1.55', '0.0000', '0.000'),
    ]
    xs = [57, 64, 72.5, 81, 88, 96]
    yt = 88
    for r, row in enumerate(rows):
        if r == 0:
            ax.plot([54.5, 97], [yt - 1.4, yt - 1.4], color='#9a9aa4', lw=0.9)
        for x, cell in zip(xs, row):
            ax.text(x, yt, cell, fontsize=8, ha='right', color=INK,
                    weight='bold' if r == 0 else 'normal')
        yt -= 3.1
    ax.text(54.5, yt + 0.6,
            'dB, dB, dB, µs, dB.   ITD tolerance is 1 µs — a magnitude-only\n'
            'modification must not move it at all, and it does not.\n'
            'cue leak = elevation spread of the untrained ear, 5.7–11.3 kHz.',
            fontsize=7.4, color='#55555f', va='top', style='italic', linespacing=1.6)

    ax.text(54, 62, 'WHAT THE ENVELOPE CHOICE COSTS', fontsize=10, weight='bold', color=INK)
    tbl = [('', 'cue leak', 'ILD 0.2–2k'),
           ('unprocessed ear', '4.66', '1.80'),
           ('linear n=4, per direction  (v1)', '2.30', '10.92'),
           ('ERB n=4, per direction', '1.85', '2.10'),
           ('ERB n=4, elevation-averaged  (v2)', '0.00', '2.22')]
    yy = 58.6
    for r, (a, b, c) in enumerate(tbl):
        w = 'bold' if r == 0 else 'normal'
        col = CHECK if 'v2' in a else INK
        ax.text(55, yy, a, fontsize=8, color=col, weight=w)
        ax.text(84, yy, b, fontsize=8, ha='right', color=col, weight=w)
        ax.text(95, yy, c, fontsize=8, ha='right', color=col, weight=w)
        if r == 0:
            ax.plot([54.5, 95], [yy - 1.4, yy - 1.4], color='#9a9aa4', lw=0.9)
        yy -= 3.0
    ax.text(54.5, yy + 0.4,
            'Smoothing in FREQUENCY has to trade cue removal against level\n'
            'accuracy — both come out of the same n_keep coefficients.\n'
            'Averaging over ELEVATION removes the cue outright and leaves\n'
            'the level free. Only possible because the arc is all az = 0, so\n'
            'the head shadow is added afterwards by the model and survives.',
            fontsize=7.5, color='#55555f', va='top', style='italic', linespacing=1.65)

    ax.text(54, 29, 'CONVENTION  (unchanged, documented)', fontsize=10,
            weight='bold', color=INK)
    ax.text(54.5, 26,
            'SOFA azimuth is CCW-positive: +az is the LEFT hemifield.\n\n'
            'TRAINED_HEMI = (−35, 0) for a left-trained subject therefore selects\n'
            'SOFA az 325–360, the RIGHT hemifield — so "trained hemifield" means\n'
            'contralateral to the trained ear. Left as-is for cohort continuity;\n'
            'the 2×2 is invariant under a global sign flip of the azimuth axis.',
            fontsize=8, color=INK, va='top', linespacing=1.7)

    ax.text(4, 5.5, 'hrtf_relearning · regenerate with docs/make_pipeline_schematic.py',
            fontsize=7.5, color='#8a8a94')

    pdf.savefig(fig); plt.close(fig)


def main():
    with PdfPages(OUT) as pdf:
        page_chain(pdf)
        page_protocol(pdf)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
