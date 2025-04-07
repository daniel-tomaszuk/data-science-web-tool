import base64
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class LinePlotHandler:

    def __init__(
        self,
        data: "models.Data",
        axis_x_name: str,
        axis_y_name: str,
        group_by_name: str,
    ):
        self.data = data
        self.axis_x_name = axis_x_name
        self.axis_y_name = axis_y_name
        self.group_by_name = group_by_name

    def create_image(self) -> str:
        df: pd.DataFrame = self.data.get_df()
        plot_kwargs = {}

        if (
            self.axis_x_name
            and self.axis_x_name in df.columns
            and self.axis_y_name
            and self.axis_y_name in df.columns
        ):
            plot_kwargs["x"] = df[self.axis_x_name]
            plot_kwargs["y"] = df[self.axis_y_name]

        elif self.axis_x_name and self.axis_x_name in df.columns:
            plot_kwargs["data"] = df[self.axis_x_name]

        elif self.axis_y_name and self.axis_y_name in df.columns:
            plot_kwargs["data"] = df[self.axis_y_name]

        else:
            return ""

        plt.figure(figsize=(8, 5))
        sns.lineplot(**plot_kwargs)
        plt.xlabel(self.axis_x_name)
        plt.ylabel(self.axis_y_name)
        plt.title(f"Line plot of {self.axis_x_name} vs. {self.axis_y_name}")
        plt.grid(True)

        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{base64_img}"
