import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Plotter():
    def __init__(self):
        pass

    def update_history(self, history):
        self.history = history

    def plot_loss(self, t, history=None, return_fig=True, logscale=False) -> go.Figure:
        self.update_history(history) if history is not None else None

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                y = self.history['loss'],
                name = 'Loss'
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                y = self.history['lr'],
                name = 'Learning rate',
                line=dict(color="lightgreen", dash="solid"),
            ),
            secondary_y=True
        )

        fig.update_layout(
            title = f'Loss plot of timestep {t}',
            xaxis_title = 'Epoch',
            yaxis = dict(
                title = 'Loss',
                type = "log" if logscale else "linear",
                tickformat=".0e" if logscale else None,
            ),
            yaxis2 = dict(
                title = 'Learning rate',
                showgrid = False,
                side = 'right',
                type="log",
                tickformat=".0e",
            )
        )

        if return_fig:
            return fig
        else:
            raise NotImplementedError("Plot not implemented")