import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go


def plot_distance_matrix(matrix, names, width=10, height=10, vmax=None, cmap="viridis"):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.imshow(matrix, vmin=None, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontdict={"fontsize": 15})
    ax.set_yticklabels(names, fontdict={"fontsize": 15})

    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, round(matrix[i, j], 2),
                           ha="center", va="center", color="w", size=15)
    fig.tight_layout()
    plt.show()


def plot_contour(x, y, x_axis_title="", y_axis_title="", names=None, showlabels=True, max_x=1.2, max_y=1.2):
    traces = list()
    if names is None:
        names = []
    traces.append(go.Histogram2dContour(
        x=x,
        y=y,
        colorscale='Greys',
        reversescale=True,
        xaxis='x',
        yaxis='y',
        showscale=False,
        contours={"coloring": "fill", "showlabels": showlabels, "start": -1, "labelfont": {"color": "white"}}
    ))
    traces.append(go.Scatter(
        x=x,
        y=y,
        xaxis='x',
        yaxis='y',
        mode='markers',
        hovertext=names,
        marker=dict(
            color='rgba(0,0,0,0.3)',
            size=5,
            line=dict(width=1,
                      color='Grey')
        )
    ))
    traces.append(go.Histogram(
        y=y,
        xaxis='x2',
        marker=dict(
            color="gray",
            opacity=0.5,
        ),
    ))
    traces.append(go.Histogram(
        x=x,
        yaxis='y2',
        marker=dict(
            color="gray",
            opacity=0.5,
        ),
    ))

    layout = go.Layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showline=True,
            mirror=True,
            range=[0, max_x],
            title={"text": x_axis_title, "font": {"size": 25}},
            tickfont={"size": 20}
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showline=True,
            mirror=True,
            range=[0, max_y],
            title={"text": y_axis_title, "font": {"size": 25}},
            tickfont={"size": 20}
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.85, 1],
            showgrid=False,
            showline=False,
            showspikes=False,
            showticklabels=False
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.85, 1],
            showgrid=False,
            showline=False,
            showspikes=False,
            showticklabels=False
        ),
        height=1000,
        width=1000,
        bargap=0,
        hovermode='closest',
        showlegend=False,
        plot_bgcolor="white"
    )

    return go.Figure(traces, layout)


def plot_difference(differences, names, name_theirs, their_color, name_ours="Caretta", x_axis_title="Alignments",
                    y_axis_title="Difference in core columns"):
    inds = np.argsort(differences)
    names = [names[i] for i in inds]
    zeros = sorted(list(np.where(differences[inds] == 0)[0]) * 2)
    lower = sorted(list(np.where(differences[inds] < 0)[0]) * 2)
    higher = sorted(list(np.where(differences[inds] > 0)[0]) * 2)
    min_d, max_d = min(differences), max(differences)
    # print(len(list(np.where(differences[inds] == 0)[0])) + len(list(np.where(differences[inds] > 0))), len(inds))
    traces = [go.Scatter(name=f"{name_ours} performs equally", mode="lines", y=[min_d, max_d] * (len(zeros) // 2), x=zeros,
                         line={"color": "black", "width": 7}, opacity=0.1),
              go.Scatter(name=f"{name_ours} outperforms", mode="lines", y=[min_d, max_d] * (len(higher) // 2), x=higher,
                         line={"color": "green", "width": 7}, opacity=0.1),
              go.Scatter(name=f"{name_ours} underperforms", mode="lines", y=[min_d, max_d] * (len(lower) // 2), x=lower,
                         line={"color": "red", "width": 7}, opacity=0.1),
              go.Scatter(name=f"{name_ours} - {name_theirs}", mode="markers+lines", y=differences[inds], x=np.arange(len(differences)),
                         line={"color": their_color, "width": 2}, marker={"color": their_color, "size": 5, "opacity": 0.5},
                         opacity=0.75, hoverinfo='text', text=names)]
    layout = go.Layout(dict(
        plot_bgcolor="white",
        yaxis=dict(title=dict(text=y_axis_title, font={"size": 20}),
                   tickfont={"size": 15}, showgrid=False, showline=True, mirror=True, range=[min_d, max_d]),
        xaxis=dict(title=dict(text=x_axis_title, font={"size": 20}),
                   tickfont={"size": 15}, showgrid=False, showline=True, mirror=True),
        height=750,
        width=1000,
        bargap=0,
        hovermode='closest',
        showlegend=True, legend=dict(orientation="h", y=1.1, font=dict(size=15))))
    return go.Figure(traces, layout)
