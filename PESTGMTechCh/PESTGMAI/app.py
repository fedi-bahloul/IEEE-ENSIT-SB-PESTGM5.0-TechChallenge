import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import time

# Load your data
df = pd.read_csv('EXTRACT.csv')

# Preprocess data if needed
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
home_arima_df = df[0:1000]

# Initialize Dash app
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/darkly/bootstrap.min.css''styles.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(className='container-fluid', children=[
    html.H1(className='header', children="Intelligent Energy Management System Dashboard"),

    dcc.Tabs([
        dcc.Tab(label='Random Forest Prediction', children=[
            html.Div(className='tab-content', children=[
                html.Div(className='graph-container', children=[
                    dcc.Graph(id='rf-prediction-graph'),
                ]),
                html.P(className='text-label', children="Train/Test Ratio:"),
                dcc.Slider(
                    id='train-test-ratio-slider',
                    min=0.1,
                    max=0.9,
                    step=0.1,
                    value=0.8,
                    marks={i: f"{i * 100}%" for i in np.arange(0.1, 1, 0.1)}
                )
            ])
        ]),

        dcc.Tab(label='Time Series Forecasting', children=[
            html.Div(className='tab-content', children=[
                html.Div(className='graph-container', children=[
                    dcc.Graph(id='ts-forecast-graph'),
                ]),
                html.P(className='text-label', children="Time Series Column:"),
                dcc.Dropdown(
                    id='ts-column-dropdown',
                    options=[{'label': col, 'value': col} for col in df.columns if col != 'time'],
                    value='House overall',
                    className='dropdown-container'
                ),
                html.P(className='text-label', children="Train/Test Ratio:"),
                dcc.Slider(
                    id='ts-train-test-ratio-slider',
                    min=0.1,
                    max=0.9,
                    step=0.1,
                    value=0.8,
                    marks={i: f"{i * 100}%" for i in np.arange(0.1, 1, 0.1)},
                    className='slider-container'
                )
            ])
        ]),

        dcc.Tab(label='Anomaly Detection', children=[
            html.Div(className='tab-content', children=[
                html.Div(className='graph-container', children=[
                    dcc.Graph(id='anomaly-detection-graph'),
                ]),
                html.P(className='text-label', children="Anomaly Detection Column:"),
                dcc.Dropdown(
                    id='anomaly-column-dropdown',
                    options=[{'label': col, 'value': col} for col in df.columns if col != 'time'],
                    value='House overall',
                    className='dropdown-container'
                ),
                html.P(className='text-label', children="Resample Frequency:"),
                dcc.Dropdown(
                    id='resample-frequency-dropdown',
                    options=[
                        {'label': 'Daily', 'value': 'D'},
                        {'label': 'Weekly', 'value': 'W'},
                        {'label': 'Monthly', 'value': 'M'}
                    ],
                    value='D',
                    className='dropdown-container'
                )
            ])
        ]),

        dcc.Tab(label='ARIMA Model', children=[
            html.Div(className='tab-content', children=[
                html.Div(className='graph-container', children=[
                    dcc.Graph(id='arima-model-graph'),
                ]),
                html.P(className='text-label', children="Select Column:"),
                dcc.Dropdown(
                    id='arima-column-dropdown',
                    options=[{'label': col, 'value': col} for col in home_arima_df.columns],
                    value=home_arima_df.columns[0],
                    className='dropdown-container'
                ),
                html.P(className='text-label', children="Model Order (p,d,q):"),
                dcc.Input(id='arima-p', type='number', value=5, className='input-field'),
                dcc.Input(id='arima-d', type='number', value=1, className='input-field'),
                dcc.Input(id='arima-q', type='number', value=0, className='input-field')
            ])
        ])
    ])
])


@app.callback(
    Output('rf-prediction-graph', 'figure'),
    [Input('train-test-ratio-slider', 'value')]
)
def update_rf_prediction_graph(tt_ratio):
    x = df.drop('House overall', axis=1)
    y = df['House overall']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - tt_ratio, random_state=52)
    rf_reg = RandomForestRegressor(random_state=52)
    rf_model = rf_reg.fit(x_train, y_train)
    y_pred_rf = rf_model.predict(x_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_test.index, y=y_test, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=x_test.index, y=y_pred_rf, mode='lines', name='Predicted'))

    fig.update_layout(title='Random Forest Prediction', xaxis_title='Time', yaxis_title='Energy Usage')
    return fig


@app.callback(
    Output('ts-forecast-graph', 'figure'),
    [Input('ts-column-dropdown', 'value'), Input('ts-train-test-ratio-slider', 'value')]
)
def update_ts_forecast_graph(col, tt_ratio):
    data = df[col].resample('W').mean()

    X = data.values
    size = int(len(X) * tt_ratio)
    train, test = X[:size], X[size:]
    history = list(train)
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test[t])

    preds = np.append(train, predictions)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=data.index, y=preds, mode='lines', name='Predicted'))
    fig.add_shape(type="line", x0=data.index[size], x1=data.index[size], y0=min(data), y1=max(data),
                  line=dict(color="Red", width=2))

    fig.update_layout(title=f'Time Series Forecasting for {col}', xaxis_title='Time', yaxis_title='Energy Usage')
    return fig


@app.callback(
    Output('anomaly-detection-graph', 'figure'),
    [Input('anomaly-column-dropdown', 'value'), Input('resample-frequency-dropdown', 'value')]
)
def update_anomaly_detection_graph(col, freq):
    data = df[col].resample(freq).sum()
    isolation_forest = IsolationForest(n_estimators=100)
    isolation_forest.fit(data.values.reshape(-1, 1))
    data['scores'] = isolation_forest.decision_function(data.values.reshape(-1, 1))
    data['anomaly'] = isolation_forest.predict(data.values.reshape(-1, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=data.index[data['anomaly'] == -1], y=data[col][data['anomaly'] == -1], mode='markers',
                             name='Anomaly', marker=dict(color='red', size=10)))

    fig.update_layout(title=f'Anomaly Detection for {col}', xaxis_title='Time', yaxis_title='Energy Usage')
    return fig


@app.callback(
    Output('arima-model-graph', 'figure'),
    [Input('arima-column-dropdown', 'value'), Input('arima-p', 'value'), Input('arima-d', 'value'),
     Input('arima-q', 'value')]
)
def update_arima_model_graph(column, p, d, q):
    array = np.array(home_arima_df[column]).reshape(-1, 1)
    size = int(len(array) * 0.70)
    train, test = array[0:size], array[size:len(array)]
    history = [x for x in train]
    predictions = list()
    start_time = time.time()

    for t in range(len(test)):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    error = mean_squared_error(test, predictions)
    end_time = time.time()
    time_taken = end_time - start_time

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=test.flatten(), mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted'))
    fig.update_layout(
        title=f'ARIMA Model Prediction for {column} (p={p}, d={d}, q={q})',
        xaxis_title='Time',
        yaxis_title='Values',
        annotations=[
            dict(
                x=0.5,
                y=-0.15,
                showarrow=False,
                text=f'Time taken: {time_taken:.2f} seconds | MSE: {error:.3f}',
                xref='paper',
                yref='paper'
            )
        ]
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
