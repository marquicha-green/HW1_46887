import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Data Explorer and Predictor", layout="wide")
st.title("Data Explorer and Predictor")
st.caption("Upload a CSV file, explore numerical data, and make predictions using linear regression.")

# The shape is the same as the first app.
# The user uploads a file, and everything else sits inside an if.
uploaded_file = st.file_uploader("Upload a CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    #Show a preview of the dataset
    st.subheader("Preview of the Dataset")
    st.dataframe(df.head())

    # This time we need three numeric columns:
    # one to use as the input and one to predict.
    # Two select boxes let the user choose.
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
        st.error("This file needs at least three numeric columns.")
        st.stop()

    # two inputs and one target
    col1, col2, col3 = st.columns(3)
    x_col1 = col1.selectbox("Predictor 1: (X-axis)", numeric_cols)

    x_options2 = [c for c in numeric_cols if c != x_col1]
    x_col2 = col2.selectbox("Predictor 2: (Color)", x_options2)

    y_options = [c for c in numeric_cols
                 if c != x_col1 and c != x_col2]
    y_col = col2.selectbox("Target: (Y-axis)", y_options)

    data = df[[x_col1, x_col2, y_col]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with values in both columns.")
        st.stop()

    # Then the model.
    # We keep the two columns, fit the line,
    # and draw the dots with the line through them.
    x = data[[x_col1, x_col2]]
    y = data[y_col]
    model = LinearRegression()
    model.fit(x, y)

    # display model coefficients and R squared.
    st.subheader("Understanding the Model Results")
    m1, m2, m3= st.columns(3)
    m1.metric(f"{x_col1} Coefficient", f"{model.coef_[0]:,.2f}" )
    m2.metric(f"{x_col2} Coefficient", f"{model.coef_[1]:,.2f}")
    m3.metric("R-sqaured", f"{model.score(x, y):.3f}")

    st.write(
        f"For every one-unit increase in {x_col1}, "
        f"the predicted {y_col} changes by "
        f"{model.coef_[0]:,.2f}, holding {x_col2} constant."
    )

    st.write(
        f"For every one-unit increase in {x_col2}, "
        f"the predicted {y_col} changes by "
        f"{model.coef_[1]:,.2f}, holding {x_col1} constant."
    )

    st.write(
        f"The R-squared value indicates that the model "
        f"explains approximately {model.score(x, y) * 100:.1f}% "
        f"of the variation in {y_col} in the uploaded dataset."
    )

    # user enters values to make a prediction
    st.subheader("Try it")

        # Automatically detect whole numbers or decimals

    if (data[x_col1] % 1 == 0).all():
        value1 = st.number_input(
            f"Enter a value for {x_col1}",
            value=int(round(data[x_col1].mean())),
            step=1,
            key=f"whole_input1_{x_col1}"
        )
    else:
        value1 = st.number_input(
            f"Enter a value for {x_col1}",
            value=round(float(data[x_col1].mean()),2),
            step=0.01,
            format="%.2f",
            key=f"decimal_input1_{x_col1}"
        )

    if (data[x_col2] % 1 == 0).all():
        value2 = st.number_input(
            f"Enter a value for {x_col2}",
            value=int(round(data[x_col2].mean())),
            step=1,
            key=f"whole_input2_{x_col2}"
        )
    else:
        value2 = st.number_input(
            f"Enter a value for {x_col2}",
            value=round(float(data[x_col2].mean()), 2),
            step=0.01,
            format="%.2f",
            key=f"decimal_input2_{x_col2}"
        )

    # check whether inputs are within the observed data range
    low1, high1 = data[x_col1].min(), data[x_col1].max()
    low2, high2 = data[x_col2].min(), data[x_col2].max()

    valid1 = low1 <= value1 <= high1
    valid2 = low2 <= value2 <= high2

    if not valid1:
        st.warning(
            f"The selected {x_col1} value ({value1:.2f}) "
            f"is outside the observed range of "
            f"{low1:.2f} to {high1:.2f}."
        )

    if not valid2:
        st.warning(
            f"The selected {x_col2} value ({value2:.2f}) "
            f"is outside the observed range of "
            f"{low2:.2f} to {high2:.2f}."
        )

    if valid1 and valid2:
        # use the same column names and order as training
        new_person = pd.DataFrame({
            x_col1: [value1],
            x_col2: [value2]
        })

        prediction = model.predict(new_person)[0]

        # Format prediction based on the target column
        if (data[y_col] % 1 == 0).all():
            st.metric(
                f"Predicted {y_col}",
                f"{prediction:,.0f}"
            )

            st.caption(
                "Prediction rounded to a whole number because "
                "the selected target contains whole-number values."
            )
        else:
            st.metric(
                f"Predicted {y_col}",
                f"{prediction:,.2f}"
            )

            st.caption(
                "This is an estimate based on the uploaded data, "
                "not a guaranteed outcome."
            )
    else:
        st.error(
            "Prediction unavailable. Please enter values "
            "within the observed data ranges to generate "
            "an estimate."
        )


    # chart selection
    # lets user select the column and chart type
    st.subheader("Data Visualization")

    chart_type = st.selectbox("Choose a chart type", ["Histogram", "Scatter Plot" , "Line"])

    if chart_type == "Histogram":
        fig = px.histogram(data, x=y_col, title=f"Distribution of {y_col}")

    elif chart_type == "Scatter Plot":
            fig = px.scatter(data, x=x_col1, y=y_col, color=x_col2, title=f"{y_col} vs {x_col1}")
    else:
            chart_data = data.sort_values(x_col1)
            fig = px.line(chart_data, x=x_col1, y=y_col, title=f"{y_col} vs {x_col1}")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Purpose and Limitations")
    st.write(
        "This linear regression model uses two selected numerical "
        "variables to predict a third variable, and inaccurate "
        "predictions could lead to poor decisions when users "
        "rely on estimated results."
    )

else:
    st.info("Upload a CSV file to get started. It needs at least three numeric columns.")
