def tmp():


    train_index = []
    if train_data:
        train_index = list(train_data.keys())
        train_index_dates = index[int(train_index[0]):int(train_index[-1]) + 1]
        train_values = list(train_data.values())
        # train_values = train_values[0:len(train_index_dates)]
        sns.lineplot(
            x=train_index_dates, y=train_values, label=f"Train set data of {target_column}",
        )

    val_index = []
    val_predicted_data_plot = val_predicted_data
    if len(val_predicted_data):
        val_index = list(val_predicted_data.keys())
        if train_data and train_index:
            # connect plots so transition is smooth - last train point with first validation point
            last_train_point_index = train_index[-1]
            val_index = [last_train_point_index] + val_index
            val_predicted_data_plot = {
                last_train_point_index: train_data[last_train_point_index],
                **val_predicted_data,
            }

        val_index_dates = index[int(val_index[0]):int(val_index[-1]) + 1]
        val_values = list(val_predicted_data_plot.values())
        sns.lineplot(
            x=val_index_dates, y=val_values, label=f"Validation Predicted data of {target_column}",
        )

    test_values = []
    test_predicted_data_plot = test_predicted_data
    if len(test_predicted_data):
        test_index = list(test_predicted_data.keys())
        if val_predicted_data and val_index:
            # connect plots so transition is smooth - last validation point with first test point
            last_validation_point_index = val_index[-1]
            test_index = [last_validation_point_index] + test_index
            test_predicted_data_plot = {
                last_validation_point_index: val_predicted_data[last_validation_point_index],
                **test_predicted_data,
            }

        test_index_dates = index[int(test_index[0])::]
        test_values_plot = list(test_predicted_data_plot.values())[0:len(test_index)]
        sns.lineplot(
            x=test_index_dates, y=test_values_plot, label=f"Test Predicted data of {target_column}",
        )

    if len(forecast_horizon_data):
        step = (index[-1] - index[-2]) if len(index) > 1 else pd.Timedelta(days=1)
        forecast_index = [index[-1] + step * (i + 1) for i in range(len(forecast_horizon_data))]
        if test_values:
            # connect plots so transition is smooth - last test point with first forecast point
            forecast_horizon_data = [test_values[-1]] + forecast_horizon_data
            forecast_index = [index[-1]] + forecast_index

        sns.lineplot(
            x=forecast_index,
            y=forecast_horizon_data,
            label=f"Forecast data of {target_column}",
            linestyle="--",
        )

    if linear_regression_result.slope is not None and linear_regression_result.intercept is not None:
        base_date = index[0]
        base_ordinal = base_date.toordinal()

        ordinal_index = [d.toordinal() - base_ordinal for d in index]
        linear_space = np.linspace(min(ordinal_index), max(ordinal_index), 100)
        y_regression_values = linear_regression_result.slope * linear_space + linear_regression_result.intercept
        date_space = [base_date + timedelta(days=int(d)) for d in linear_space]

        sns.lineplot(x=date_space, y=y_regression_values, label="Linear Regression Line", linestyle=":")

    plt.xticks(rotation=45)
    plt.ylim(top=800)
    plt.gcf().autofmt_xdate()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as a base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
    plt.close()
    return f"data:image/png;base64,{base64_img}"
