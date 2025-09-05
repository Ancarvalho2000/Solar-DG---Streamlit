import streamlit as st
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

st.title('Solar DG - Time Series Analysis')

# Initialize panel flags in session_state
for tool in ['adf_panel', 'arima_panel', 'plot_panel']:
    if tool not in st.session_state:
        st.session_state[tool] = False

def adf_test(series, title=''):
    st.write(f'**Augmented Dickey-Fuller Test:** {title}')
    result = adfuller(series.dropna(), autolag='AIC')

    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pd.Series(result[0:4], index=labels)

    for key, val in result[4].items():
        out[f'critical value ({key})'] = val

    st.write(out)

    if result[1] <= 0.05:
        st.markdown("Strong evidence against the null hypothesis<br>"
                    "Reject the null hypothesis<br>"
                    "Data has no unit root and is **Stationary**",
                    unsafe_allow_html=True)
    else:
        st.markdown("Weak evidence against the null hypothesis<br>"
                    "Fail to reject the null hypothesis<br>"
                    "Data has a unit root and is **Non-Stationary**",
                    unsafe_allow_html=True)

def plot_graph(df):
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Define the range from first to last
    start_date = df.index.min()
    end_date = df.index.max()

    # Create tick locations
    tick_locs = pd.date_range(start=start_date, end=end_date, freq='7D')

    # Force x-axis label to start and end with the first and last date
    if start_date not in tick_locs:
        tick_locs = tick_locs.insert(0, start_date)
    if end_date not in tick_locs:
        tick_locs = tick_locs.append(pd.DatetimeIndex([end_date]))

    st.write("### Select columns to plot:")
    selected_cols = []

    for col_name in numeric_cols:
        checked = st.checkbox(col_name, key=f'chk_{col_name}')
        if checked:
            selected_cols.append(col_name)

    if selected_cols:
        fig, ax = plt.subplots(figsize=(12, 6))

        for col in selected_cols:
            df[col].plot(ax=ax, legend=True)

        plt.xticks(tick_locs, rotation=0)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        plt.xlim(start_date, end_date)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    else:
        st.info("No columns selected for plotting.")


def get_last_week(df):
    # Guarantee DATE_TIME is a proper datetime index
    df.index = pd.to_datetime(df.index, format='%d-%m-%Y %H:%M')

    start_date = df.index.max() - pd.Timedelta(days=7)

    last_week_df = df.loc[start_date:]

    len_last_week = len(last_week_df)

    # st.write(f"Number of rows in the last week: {len_last_week}")

    return len_last_week


def arima_test(df, col):
    col = str(col)

    len_last_week = get_last_week(df)

    train = df.iloc[:-len_last_week]
    test = df.iloc[-len_last_week:]

    # Resample only the selected column and handle missing values
    df_min = df[[col]].resample('min').mean().dropna()

    # fig, ax = plt.subplots()
    # df_min[col].plot(ax=ax)
    # st.pyplot(fig)

    # Fit auto_arima on entire (resampled) selected column
    model = auto_arima(df_min[col], seasonal=False)
    # summary_obj = model.summary()
    # try:
    #     summary_str = summary_obj.as_text()
    # except AttributeError:
    #     summary_str = str(summary_obj)
    # st.text(summary_str)

    p, d, q = model.order
    # st.write(f"ARIMA order: p={p}, d={d}, q={q}")

    # Predict on test portion
    start = len(train)
    end = start + len(test) - 1

    results_arima = ARIMA(df[col], order=(p, d, q)).fit()
    # summary_obj = results_arima.summary()
    # try:
    #     summary_str = summary_obj.as_text()
    # except AttributeError:
    #     summary_str = str(summary_obj)
    # st.text(summary_str)

    predictions_arima = results_arima.predict(start=start, end=end, typ='levels').rename('ARIMA Prediction')

    # Plot actual vs predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    test[col].plot(ax=ax, legend=True)
    predictions_arima.plot(ax=ax, legend=True)
    st.pyplot(fig, use_container_width=True)

    # st.write(f"test[{col}] mean: {test[col].mean()}")
    # st.write(f"predictions mean: {predictions_arima.mean()}")

    # Calculate error metrics
    mae_arima = mean_absolute_error(test[col], predictions_arima)
    st.write(f"MAE: {mae_arima}")

    error_arima = (mae_arima / test[col].mean()) * 100
    accuracy_arima = 100 - error_arima
    st.write(f"Accuracy: {accuracy_arima}%")



uploaded_file = st.sidebar.file_uploader("Upload a file", type="csv")


@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    first_col = df.columns[0]
    df.set_index(first_col, inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.divider()

    # Button ADF Test
    if st.button("Perform ADF Test"):
        st.session_state['adf_panel'] = not st.session_state['adf_panel']

    cols = st.columns([1, 5])
    if st.session_state['adf_panel']:
        with cols[1]:
            st.write("### Select a column to perform the test on:")
            for col_name in df.columns:
                if st.button(col_name, key=f"adf_btn_{col_name}"):
                    st.session_state['adf_selected_column'] = col_name
            if 'adf_selected_column' in st.session_state:
                selected_col = st.session_state['adf_selected_column']

                try:
                    adf_test(df[selected_col], title=selected_col)
                except Exception as e:
                    st.error(f"Could not perform ADF test on column '{selected_col}': {e}" )

    st.divider()

    # Button Plot graph
    if st.button("Plot graph"):
        st.session_state['plot_panel'] = not st.session_state['plot_panel']

    cols = st.columns([1, 5])
    if st.session_state['plot_panel']:
        with cols[1]:
            plot_graph(df)

    st.divider()

    # Button ARIMA
    if st.button("Perform ARIMA for last week"):
        st.session_state['arima_panel'] = not st.session_state['arima_panel']

    cols = st.columns([1, 5])
    if st.session_state['arima_panel']:
        with cols[1]:
            st.write("### Select a column to perform ARIMA on:")
            for col_name in df.columns:
                if st.button(col_name, key=f"arima_btn_{col_name}"):
                    st.session_state['arima_selected_column'] = col_name
            if 'arima_selected_column' in st.session_state:
                selected_col = st.session_state['arima_selected_column']
                arima_test(df, selected_col)
