import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Data Explorer and Predictor", layout="wide")
st.title("Data Explorer and Predictor")
st.caption("Upload a CSV file, explore numerical data, and make predictions using linear regression.")

# The shape is the same as the first app.
# The user uploads a file, and everything else sits inside an if.
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    #Show a preview of the dataset
    st.subheader("Preview of the dataset")
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
    x_col1 = col1.selectbox("First Input: what you know", numeric_cols)

    x_options2 = [c for c in numeric_cols if c != x_col1]
    x_col2 = col2.selectbox("Second Input: what you know", x_options2)

    y_options = [c for c in numeric_cols
                 if c != x_col1 and c != x_col2]
    y_col = col2.selectbox("Target: what you want to predict", y_options)

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
    m1, m2, m3= st.columns(3)
    m1.metric(f"Each extra {x_col1} adds", f"{model.coef_[0]:,.2f}" )
    m2.metric(f"Each extra {x_col2} adds", f"{model.coef_[1]:,.2f}")
    m3.metric("R2 on this data", f"{model.score(x, y):.2f}")

    # The last part is the prediction.
    # The user types a value, and the app reads the answer from the model.
    # Notice the table we give to predict: it has the same column name
    # as the training data. A model expects inputs in the same form
    # it was trained on.
    st.subheader("Try it")
    value1 = st.number_input(f"Enter a value for {x_col1}", value=float(data[x_col1].mean()))
    value2 = st.number_input(f"Enter a value for {x_col2}", value=float(data[x_col2].mean()))

    # use the same column name and order as training
    new_person = pd.DataFrame({x_col1: [value1], x_col2: [value2]})

    prediction = model.predict(new_person)[0]
    st.metric(f"Predicted {y_col}", f"{prediction:,.2f}")

    # The last lines are a warning.
    # If the user types a value outside the range of the data, the app says so.

    low1, high1 = data[x_col1].min(), data[x_col1].max()
    low2, high2 = data[x_col2].min(), data[x_col2].max()

    if value1 < low1 or value1 > high1:
        st.warning(f"{x_col1} is outside the range the model learned from ({low1:,.2f} to {high1:,.2f}).")
    if value2 < low2 or value2 > high2:
        st.warning(f"{x_col2} is outside the range the model learned from ({low2:,.2f} to {high2:,.2f}).")

    # chart selection
    # lets user select the column and chart type
    st.subheader("Data Visualization")

    chart_col = st.selectbox("Choose a numeric column", numeric_cols)
    chart_type = st.selectbox("Choose a chart type", ["Histogram", "Scatter Plot" , "Line"])

    if chart_type == "Histogram":
        fig = px.histogram(df, x=chart_col, title=f"Distribution of {chart_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        other_col = [c for c in numeric_cols if c != chart_col]
        second_col = st.selectbox("Choose a second column", other_col)

        chart_data = df[[chart_col, second_col]].dropna()

        if chart_type == "Scatter Plot":
            fig = px.scatter(chart_data, x=chart_col, y=second_col, title=f"{chart_col} vs {second_col}")
        else:
            chart_data = chart_data.sort_values(chart_col)
            fig = px.line(chart_data, x=chart_col, y=second_col, title=f"{chart_col} vs {second_col}")

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
