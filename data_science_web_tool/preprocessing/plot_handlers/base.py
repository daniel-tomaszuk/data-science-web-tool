class BasePlotHandler:
    SELECTOR_EMPTY_CHOICE_VALUE = "-"
    DEFAULT_FIGSIZE = (12, 6)

    def __init__(
        self,
        data: "models.Data",
        axis_x_name: str,
        axis_y_name: str,
        group_by_name: str,
    ):
        self.data = data
        self.axis_x_name = axis_x_name if axis_x_name and axis_x_name != self.SELECTOR_EMPTY_CHOICE_VALUE else None
        self.axis_y_name = axis_y_name if axis_y_name and axis_y_name != self.SELECTOR_EMPTY_CHOICE_VALUE else None
        self.group_by_name = (
            group_by_name if group_by_name and group_by_name != self.SELECTOR_EMPTY_CHOICE_VALUE else None
        )
        self.plot_kwargs, self.x_label, self.y_label = self.get_plot_kwargs()

    def get_plot_kwargs(self):
        raise NotImplementedError()
