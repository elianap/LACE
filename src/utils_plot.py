def barPlotPlotly(x, y, title, reverse=True, xaxis_title="", hovertext=None):
    import plotly.graph_objects as go

    if reverse:
        x.reverse()
        y.reverse()
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            orientation="h",
            marker=dict(color="#bee2e8", line=dict(color="black", width=0.4)),
            hovertext=hovertext,
            hoverlabel=dict(namelength=-1, font_size=10),
        )
    )

    axis = dict(
        showline=False,
        zeroline=False,
        showgrid=False,
    )

    fig.update_layout(
        title={"text": title},
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode="closest",
        plot_bgcolor="rgb(248,248,248)",
        width=500,
        height=350,
        xaxis=axis,
        yaxis=axis,
    )
    fig.update_xaxes(title_text=xaxis_title)
    return fig


def barPlotMPL(
    x, y, title, reverse=True, xaxis_title="", colors="#bee2e8", fontsize=14
):
    import matplotlib.pyplot as plt

    fig, pred_ax = plt.subplots(1, 1)

    pred_ax.set_title(title, fontsize=fontsize)
    pred_ax.set_xlabel(xaxis_title, fontsize=fontsize)

    pred_ax.barh(
        y, width=x, align="center", color=colors, linewidth="1", edgecolor="black"
    )
    pred_ax.tick_params(labelsize=fontsize)
    if reverse:
        pred_ax.invert_yaxis()
    return fig