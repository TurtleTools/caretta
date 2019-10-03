import numpy as np
import plotly.graph_objs as go


def plot_contour(x, y, x_axis_title="", y_axis_title="", showlabels=True):
    traces = list()
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
            title={"text": x_axis_title, "font": {"size": 25}},
            tickfont={"size": 20}
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showline=True,
            mirror=True,
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
    )
    return go.Figure(traces, layout)


def plot_difference(differences, name_theirs, name_ours="Caretta", x_axis_title="Alignments", y_axis_title="Difference in core columns"):
    inds = np.argsort(differences)
    zeros = sorted(list(np.where(differences[inds] == 0)[0]) * 2)
    lower = sorted(list(np.where(differences[inds] < 0)[0]) * 2)
    higher = sorted(list(np.where(differences[inds] > 0)[0]) * 2)
    min_d, max_d = min(differences), max(differences)
    traces = [go.Scatter(name=f"Similar to {name_ours}", mode="lines", y=[min_d, max_d] * (len(zeros) // 2), x=zeros,
                         line={"color": "black", "width": 10}, opacity=0.1),
              go.Scatter(name=f"{name_ours} outperforms", mode="lines", y=[min_d, max_d] * (len(higher) // 2), x=higher,
                         line={"color": "green", "width": 10}, opacity=0.1),
              go.Scatter(name=f"{name_ours} underperforms", mode="lines", y=[min_d, max_d] * (len(lower) // 2), x=lower,
                         line={"color": "red", "width": 10}, opacity=0.1),
              go.Scatter(name=f"{name_ours} - {name_theirs}", mode="lines", y=differences[inds], x=np.arange(len(differences)),
                         line={"color": "purple", "width": 2},
                         opacity=0.75)]
    layout = go.Layout(dict(
        yaxis=dict(title=dict(text=y_axis_title, font={"size": 25}),
                   tickfont={"size": 15}, showgrid=False, showline=True, mirror=True, range=[min_d, max_d]),
        xaxis=dict(title=dict(text=x_axis_title, font={"size": 25}),
                   tickfont={"size": 15}, showgrid=False, showline=True, mirror=True),
        height=700,
        width=1000,
        bargap=0,
        hovermode='closest',
        showlegend=True, legend=dict(orientation="h", y=1.1)))
    return go.Figure(traces, layout)
