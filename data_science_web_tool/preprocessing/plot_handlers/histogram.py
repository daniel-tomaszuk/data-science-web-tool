import base64
import io

import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.plot_handlers.base import BasePlotHandler


class HistogramPlotHandler(BasePlotHandler):
    def create_image(self) -> str:
        plot_kwargs = self.get_plot_kwargs()
        if not plot_kwargs:
            return ""

        plt.figure(figsize=self.DEFAULT_FIGSIZE)
        sns.histplot(
            kde=True,
            **plot_kwargs,
        )
        plt.xlabel(self.axis_x_name)
        plt.ylabel("Count")

        title = f"Histogram of {self.axis_x_name}"
        if self.axis_y_name:
            title += f" .vs {self.axis_y_name}"
            plt.ylabel(self.axis_y_name)

        plt.title(title)
        plt.grid(True)

        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{base64_img}"
