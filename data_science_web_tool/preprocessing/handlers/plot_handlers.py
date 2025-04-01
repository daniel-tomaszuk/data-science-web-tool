import base64
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class LinePlotHandler:

    def __init__(self, data: "models.Data", column_name: str):
        self.data = data
        self.column_name = column_name

    def create_image(self) -> str:
        df: pd.DataFrame = self.data.get_df()

        # TODO: move me to the get_df
        df.set_index("Date", inplace=True)
        df = df.sort_index()

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df[self.column_name])
        plt.xlabel("Date")
        plt.ylabel(self.column_name)
        plt.title(f"Line plot of {self.column_name}")
        plt.grid(True)

        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{base64_img}"
