import numpy
from matplotlib import pyplot as plt
from collections import OrderedDict
import hrtf_relearning as hr

subject_id = 'AvS'

def learning_plot(subject_id, figsize=(17.5, 6.5)):
    """
    Plot single subject stimulus response pattern
    and draw SD RMS and EG indications
    """
    fig_width = figsize[0] / 2.54  # convert to inches
    fig_height = figsize[1] / 2.54
    dpi = 264
    fs = 8  # label fontsize
    markersize = 2
    lw = .7
    params = {'font.family':'Helvetica', 'xtick.labelsize': fs, 'ytick.labelsize': fs, 'axes.labelsize': fs,
              'boxplot.capprops.linewidth': lw, 'lines.linewidth': lw,
              'ytick.direction': 'in', 'xtick.direction': 'in', 'ytick.major.size': 2,
              'xtick.major.size': 2, 'axes.linewidth': lw, 'axes.spines.right': False, 'axes.spines.top': False}
    plt.rcParams.update(params)

    # get data
    localization_dict = hr.Subject(subject_id).localization

    # correct keys
    old_key = "AvS_1_KU100_loc_18.12_15.04"
    new_key = "AvS_KU100_loc_18.12_15.04"
    if old_key in localization_dict:
        print(f"Renaming key {old_key} to {new_key}")
        localization_dict[new_key] = localization_dict.pop(old_key)

    # sort by day
    items = sorted(localization_dict.items(), key=lambda kv: parse_loc_key(kv[0]))
    by_day = OrderedDict()
    for k, seq in items:
        day = k.split("_")[3]  # "12.01"
        if seq.finished:
            by_day.setdefault(day, []).append((k, seq))

    # for now only look at last sequence per day
    # data = []
    # for i, (k, seq) in enumerate(by_day.items()):
    #     # workaround: avoid picking uso test (last test on day 3) - in the future, use seq.stim_type attribute
    #     if i == 2:
    #         data.append(hr.localization_accuracy(seq[-2][1]))
    #     elif i != 2:
    #         data.append(hr.localization_accuracy(seq[-1][1]))
    # data = numpy.array(data)  # days x subjects x metrics

    # pick all sequences
    data = []  # list over days, each element: (n_meas_that_day x n_metrics)
    times_by_day = []  # list over days, each element: list[str] length n_meas_that_day

    for day, loc_tests in by_day.items():
        _data = []
        _times = []

        # ensure within-day order by time (since by_day preserves insertion, but safe)
        loc_tests_sorted = sorted(loc_tests, key=lambda x: parse_loc_key(x[0]))

        for k, seq in loc_tests_sorted:
            _data.append(hr.localization_accuracy(seq))  # -> (n_metrics,)
            _times.append(key_time_str(k))

        data.append(numpy.vstack(_data))  # shape (n_meas, n_metrics)
        times_by_day.append(_times)  # list length n_meas

    # ----- plot ----- #
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True, dpi=dpi) #  gridspec_kw = {'width_ratios': [1, 1]}
    ax0 = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=2, rowspan=2)
    ax1 = plt.subplot2grid(shape=(2, 3), loc=(0, 2), colspan=1)
    ax2 = plt.subplot2grid(shape=(2, 3), loc=(1, 2), colspan=1)
    axes = fig.get_axes()

    labels = ['Elevation Gain', 'RMSE (deg)', 'SD (deg)']
    days = numpy.arange(1, len(data)+1)
    marker_dist = 0.15

    for i, axis in enumerate(axes):
        label = labels[i]

        for day_idx, _data in enumerate(data):
            n = _data.shape[0]

            _x = numpy.full(n, day_idx + 1, dtype=float)
            _x = numpy.concatenate(([_x[0] - marker_dist], _x[1:-1], [_x[-1] + marker_dist]))

            y = _data[:, i]
            axis.plot(_x, y, c="0", label=label, lw=lw)

            # annotate timestamps
            if i == 0:  # only for first plot
                for j in range(n):
                    t = times_by_day[day_idx][j]
                    axis.annotate(
                        t,
                        (_x[j], y[j]),
                        textcoords="offset points",
                        xytext=(3, 3),  # pixels offset
                        ha="left",
                        va="bottom",
                        fontsize=7,
                        color="0.3",
                    )
        for day_idx in range(len(data) - 1):
            # last point of current day
            y0 = data[day_idx][-1, i]
            x0 = (day_idx + 1) + marker_dist

            # first point of next day
            y1 = data[day_idx + 1][0, i]
            x1 = (day_idx + 2) - marker_dist

            axis.plot(
                [x0, x1],
                [y0, y1],
                linestyle=":",
                color="0.5",
                linewidth=1,
                zorder=0,  # behind markers
            )
    # for i, axis in enumerate(axes):
    #     label = labels[i]
    #     # marker_dist = .15
    #     # if i > 0:      # adjust distance between final markers for left and right plots separately
    #     #     marker_dist = .3
    #     for day, _data in enumerate(data):
    #         _y_ticks = [day+1] * _data.shape[0]
    #         _y_ticks = numpy.concatenate(([_y_ticks[0] - marker_dist], _y_ticks[1:-1], [_y_ticks[-1] + marker_dist]))
    #         axis.plot(_y_ticks, _data[:, i], c='0', label=label, lw=lw)  # week 1 learning curve
    #
    #         # small time stamps


        # error bars
        # axis.errorbar([0, 5, 10], ef_mean[:3, i], capsize=3, yerr=localization_dict['Ears Free']['SE'][:3, i],
        #                fmt="o", c=str(w1_color), markersize=markersize, fillstyle='full',
        #                      markerfacecolor='white', markeredgewidth=.5)  # error bar ears free
        # # error bars and markers m1_mean adaptation
        # axis.errorbar([marker_dist, 1, 2, 3, 4, 5-marker_dist], m1_mean[:6, i], capsize=2, yerr=localization_dict
        #                 ['Earmolds Week 1']['SE'][:6, i], fmt="o", c=str(w1_color), markersize=markersize, markeredgewidth=.5)
        # # error bars and markers m2_mean adaptation and persistence
        # axis.errorbar([5+marker_dist,  6,  7,  8, 9, 10-marker_dist, 15], m2_mean[:7, i], capsize=2,
        #                      yerr=localization_dict['Earmolds Week 2']['SE'][:7, i], fmt="s", c=str(w2_color),
        #                      markersize=markersize, markeredgewidth=.5)
        # error bars and markers m1_mean persistence
        # axis.errorbar([10+marker_dist], m1_mean[6, i], capsize=2, yerr=localization_dict
        #                 ['Earmolds Week 1']['SE'][6, i], fmt="o", c=str(w1_color), markersize=markersize, markeredgewidth=.5)  # err m1_mean
        #

        # axes ticks and limits
        axis.set_xticks(days)
        axis.set_xticklabels(days)
        axis.set_ylabel(label)
    axes[1].set_xticklabels([])
    axes[0].set_xlabel('Days')
    axes[2].set_xlabel('Days')

    axes[0].set_ylim(0, 1.02)
    axes[0].set_yticks(numpy.arange(0, 1.2, 0.2))

    # kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False, linewidth=lw)
    # axes[0].plot((1+0.005, 1+0.01), (-.03, +.03), **kwargs)
    # kwargs.update(transform=axes[1].transAxes)  # switch to the right axis
    # axes[1].plot((-0.01*11, -0.005*11), (-.03, +.03), **kwargs)

    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axes[0].set_yticklabels(ticklabels)


    axes[1].set_yticks(numpy.arange(0, 26, 5))
    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[0] = '0'

    axes[2].set_yticks(numpy.arange(0, 10, 2))
    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[0] = '0'

    # axes[1].set_yticks(numpy.linspace(0, 0.0001, 20))
    # axes[2].set_yticks(numpy.linspace(0, 0.0001, 5))

    # annotations
    # axes[0].annotate('insertion', xy=(0, .08), xycoords=axes[0].get_xaxis_transform(), fontsize=fs,
    #             xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    # axes[0].annotate('replacement', xy=(5, .08), xycoords=axes[0].get_xaxis_transform(), fontsize=fs,
    #             xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    # axes[0].annotate('removal', xy=(10, .5), xycoords=axes[0].get_xaxis_transform(), fontsize=fs,
    #             xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    # horizontal lines
    for y in numpy.linspace(.1, 1, 9):
        axes[0].axhline(y=y, xmin=0, xmax=20, color='0.9', linewidth=.5, zorder=-1)
    for y in numpy.arange(5, 22, 5):
        axes[1].axhline(y=y, xmin=0, xmax=20, color='0.9', linewidth=.5, zorder=-1)
    for y in numpy.arange(2, 9, 2):
        axes[2].axhline(y=y, xmin=0, xmax=20, color='0.9', linewidth=.5, zorder=-1)

    plt.tight_layout(pad=1.08, h_pad=.5, w_pad=None, rect=None)

    # subplot labels
    axes[0].annotate('A', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.1, 1.005), fontsize=fs)
    axes[1].annotate('B', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.3, 1.005), fontsize=fs)
    axes[2].annotate('C', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.3, 1.005), fontsize=fs)

    fig.suptitle(f'Subject {subject_id}')
    plt.show()
    return fig, axis

# --- helpers ---
import datetime
import re

def parse_loc_key(key: str) -> datetime.datetime:
    """
    Parse localization dict keys and return a datetime that sorts correctly
    by day AND time.

    Supported anywhere in the key:
      - dd.mm_HH:MM
      - dd.mm_HH.MM
      - ISO (yyyy-mm-ddTHH:MM:SS)

    Unknown formats sort last.
    """
    # 1) ISO (future-proof)
    try:
        return datetime.datetime.fromisoformat(key)
    except Exception:
        pass

    # 2) dd.mm_HH[:.]MM anywhere in the string
    match = re.search(r"(\d{2})\.(\d{2})_(\d{2})[:.](\d{2})", key)
    if match:
        d, m, hh, mm = match.groups()
        now = datetime.datetime.now()

        year = now.year
        month = int(m)

        # Handle year rollover (Dec → Jan)
        if month > now.month + 1:
            year -= 1

        return datetime.datetime(
            year=year,
            month=month,
            day=int(d),
            hour=int(hh),
            minute=int(mm),
        )

    # 3) Fallback → last
    return datetime.datetime.max

def key_time_str(k: str) -> str:
    # expects "..._dd.mm_HH.MM" or "..._dd.mm_HH:MM"
    # returns "HH:MM"
    parts = k.split("_")
    if len(parts) >= 2:
        t = parts[-1]
        # normalize dot to colon
        t = t.replace(".", ":")
        # keep only HH:MM if something extra is present
        m = re.search(r"(\d{2}:\d{2})", t)
        return m.group(1) if m else t
    return ""