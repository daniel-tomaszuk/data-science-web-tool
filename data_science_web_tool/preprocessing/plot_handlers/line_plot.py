import base64
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing.plot_handlers.base import BasePlotHandler


class LinePlotHandler(BasePlotHandler):
    def create_image(self) -> str:
        plt.figure(figsize=self.DEFAULT_FIGSIZE)
        sns.lineplot(**self.plot_kwargs)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        title = f"Line plot of {self.x_label} vs. {self.y_label}"
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{base64_img}"

    def get_plot_kwargs(self) -> tuple:
        df: pd.DataFrame = self.data.get_df()
        plot_kwargs = {}
        x_label = "Index"
        y_label = "Count"

        if self.axis_x_name and self.axis_x_name in df.columns and not self.axis_y_name:
            plot_kwargs["y"] = df[self.axis_x_name]
            plot_kwargs["x"] = df.index
            y_label = self.axis_x_name

        elif (
            self.axis_y_name and self.axis_y_name in df.columns and not self.axis_x_name
        ):
            plot_kwargs["y"] = df[self.axis_y_name]
            plot_kwargs["x"] = df.index
            y_label = self.axis_y_name

        elif (
            self.axis_x_name
            and self.axis_x_name in df.columns
            and self.axis_y_name
            and self.axis_y_name in df.columns
        ):
            plot_kwargs["y"] = df[self.axis_y_name]
            plot_kwargs["x"] = df[self.axis_x_name]
            x_label = self.axis_x_name
            y_label = self.axis_y_name

        if self.group_by_name and self.group_by_name in df.columns:
            plot_kwargs["hue"] = df[self.group_by_name]

        return plot_kwargs, x_label, y_label
