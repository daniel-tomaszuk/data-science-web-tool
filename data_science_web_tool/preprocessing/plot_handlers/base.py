import pandas as pd


class BasePlotHandler:
    SELECTOR_EMPTY_CHOICE_VALUE = "-"
    DEFAULT_FIGSIZE = (8, 5)

    def __init__(
        self,
        data: "models.Data",
        axis_x_name: str,
        axis_y_name: str,
        group_by_name: str,
    ):
        self.data = data
        self.axis_x_name = (
            axis_x_name
            if axis_x_name and axis_x_name != self.SELECTOR_EMPTY_CHOICE_VALUE
            else None
        )
        self.axis_y_name = (
            axis_y_name
            if axis_y_name and axis_y_name != self.SELECTOR_EMPTY_CHOICE_VALUE
            else None
        )
        self.group_by_name = (
            group_by_name
            if group_by_name and group_by_name != self.SELECTOR_EMPTY_CHOICE_VALUE
            else None
        )

    def get_plot_kwargs(self) -> dict:
        df: pd.DataFrame = self.data.get_df()
        plot_kwargs = {}

        if self.axis_x_name and self.axis_x_name in df.columns and not self.axis_y_name:
            plot_kwargs["y"] = df[self.axis_x_name]
            plot_kwargs["x"] = df.index

        elif (
            self.axis_y_name and self.axis_y_name in df.columns and not self.axis_x_name
        ):
            plot_kwargs["y"] = df[self.axis_y_name]
            plot_kwargs["x"] = df.index
        elif (
            self.axis_x_name
            and self.axis_x_name in df.columns
            and self.axis_y_name
            and self.axis_y_name in df.columns
        ):
            plot_kwargs["y"] = df[self.axis_y_name]
            plot_kwargs["x"] = df[self.axis_x_name]

        if self.group_by_name and self.group_by_name in df.columns:
            plot_kwargs["hue"] = df[self.group_by_name]

        return plot_kwargs
