import dash
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
import numpy as np
import plotly.graph_objects as go
from tensorflow import keras
import pandas as pd
from tensorflow.keras import layers
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from database import fetch_all_voltage_as_df


# Define the dash app first
app = dash.Dash(__name__)


# Define component functions


def page_header():
    """
    Returns the page header
    """
    return html.Div(id='header', children=[
        html.Div([html.H3('​Tri-Cities Load Interconnection Capability Visualization')], className='header'),
    ], className="row")


def description():
    """
    Returns overall project description in markdown
    """
    return html.Div(children=[dcc.Markdown('''
        # Tri-Cities Area Near Real-time Monitoring
        ''', className='description_header'),
        html.Div('The Tri-Cities Load Area serves the communities in cities of Richland, Kennewick, and Pasco. \
            It includes Franklin, Walla Walla, Grant, and Benton Counties. Load in this area peaks in the summer season. \
            The load in this area is residential, commercial, agricultural, and industrial. The generation \
            interconnection queue in the Tri-Cities includes a 3 MW solar project with a 1 MW battery and 1,110 MW \
            of Wind.', className='description'),])


def static_stacked_trend_graph(stack=False):
    """
    Line plot of utility voltage and Ashe main bus voltage.
    If `stack` is `True`, the 4 lines are stacked together.
    """
    colors = ['#6b5b95', '#b1cbbb', '#eea29a', '#c94c4c']
    df = fetch_all_voltage_as_df()
    if df is None:
        return go.Figure()
    sources = ['vol_value', 'Import', 'Load', 'Generation']
    x = df['Datetime']
    fig = go.Figure()
    for i, s in enumerate(sources):
        if s == 'vol_value':
            fig.add_trace(go.Scatter(x=x, y=df[s], mode='lines', name='main bus',
                                    line={'width': 2, 'color': colors[i]},
                                    stackgroup='stack' if stack else None))
        else:
            fig.add_trace(go.Scatter(x=x, y=df[s], mode='lines', name=s,
                                    line={'width': 2, 'color': colors[i]},
                                    stackgroup='stack' if stack else None))
    title = 'Tri-Cities Load Interconnection Capability & Recent Project'
    if stack:
        title += ' [Stacked]'

    fig.update_layout(title=title,
                      plot_bgcolor='#deeaee',
                      paper_bgcolor='#deeaee',
                      yaxis_title='Voltage Value',
                      xaxis_title='Date/Time')
    return fig


def slider_plot_description():
    """
    Describe the interactive plot with sliders - the interactive component
    """
    return html.Div(children=[
        dcc.Markdown('''
        # Utilities and Recent Project in the Tri-Cities Load Area
        ''', className='description_header'),
        html.Div('Please choose the utilities or recent project (Ashe 230kV Main Bus Voltage) to explore and compare \
            the voltage value.', className='description')
    ])


def what_if_tool():
    """
    Returns the What-If tool as a dash `html.Div`. The view is a 8:3 division between
    demand-supply plot and rescale sliders.
    """
    return html.Div(children=[
        html.Div(children=[dcc.Graph(id='slider-figure')]),

        html.Div(children=[
            html.H5("Choose the utility or Ashe main bus voltage", style={'marginTop': '2rem'}, className='description_header'),
            html.Div(children=[
                dcc.Slider(id='project-slider', min=0, max=1, step=None, marks={0: 'utility', 1: 'Ashe'}, value=0)
            ], className='p_slider', style={'marginTop': '5rem'}),
        ], className='columns', style={'marginLeft': 5, 'marginTop': '10%'}),
    ])


def pred_description():
    """
    Returns prediction description in markdown
    """
    return html.Div(children=[dcc.Markdown('''
        # Voltage Predictions
        ''', className='description_header'),
        html.Div('In the prediction part, we use two methods to predict the \'voltage value\' from a \
            specific day (which can be chosen by users). We create two prediction functions, the first \
            is the Naïve approach and the second is an improved method. To visualize the comparisons between \
            the prediction values and the real values, we use different colors to represent the different lines.', className='description'),
        html.Div('If we want to forecast the voltage value for the next day, we can simply take the last day \
            value and estimate the same value for the next day. Such forecasting technique which assumes that \
            the next expected point is equal to the last observed point is called Naive Method.', className='description'),
        html.Div('Hence we need a method that takes into account both trend and seasonality to forecast \
            future voltage values. One such algorithm that we can use in such a scenario is Holt\'s Winter \
            method. The idea behind triple exponential smoothing(Holt’s Winter) is to apply exponential smoothing \
            to the seasonal components in addition to level and trend. So, we can see the predictions are more accurate \
            than the baseline one.', className='description'),])


def pred_tool():
    """
    The function gives a option tool for users to choose from previous data used for prediction and future data they want
    to predict.
    """
    return html.Div(children=[
        html.Div(children=[dcc.Graph(id='pred-figure')]),

        html.Div(children=[
            html.Div(children=[
                html.Div('Future Data:', className='description'),
                dcc.Slider(id='future-slider', min=4, max=6, step=1, value=5,
                           tooltip={"placement": "bottom", "always_visible": True},),
            ], className='p_slider', style={'marginTop': '5rem'}),
        ], className='columns', style={'marginLeft': 5, 'marginTop': '10%'}),
    ])

def ad_description():
    """
    Returns anormaly detection description in markdown
    """
    return html.Div(children=[dcc.Markdown('''
        # Anomaly Detection
        ''', className='description_header'),
        html.Div('The anomaly detection tool demonstrates how you can use a reconstruction \
                 convolutional autoencoder model to detect anomalies in timeseries data. We have a \
                 value for every 5 mins for 7 days. Then we use MAE loss on training samples to find \
                 max MAE loss value. This is the worst our model has performed trying to reconstruct\
                 a sample. We will make this the threshold for anomaly detection. If the reconstruction\
                 loss for a sample is greater than this threshold value then we can infer that the \
                model is seeing a pattern that it is not familiar with. We will label this sample as \
                an anomaly.', className='description'),
        html.Div('With the anomaly samples, the corresponding timestamps are returned from the \
                 original test data by using the following method. All except the initial and the \
                final time_steps-1 data values, will appear in \'time_steps\' number of samples. If \
                the samples [(3, 4, 5), (4, 5, 6), (5, 6, 7)] are anomalies, the data point 5 is an \
                anomaly. Then the anomalies are overlaid on the original test data plot. Here is \
                the interactive plot. If the red line doesn\'t show up, there is no anomalies in the data.\
                Please select the interested time period to check the anomaly \
                values.', className='description'),])

def ad_tool():
    """
    Returns the dropdown options of the perid for the anormaly detection
    """
    return html.Div([
        dcc.Dropdown(
            id='period-dropdown',
            options=[
                {'label': 'day 1', 'value': 1},
                {'label': 'day 2', 'value': 2},
                {'label': 'day 3', 'value': 3},
                {'label': 'day 4', 'value': 4},
                {'label': 'day 5', 'value': 5},
                {'label': 'day 6', 'value': 6},
                {'label': 'day 7', 'value': 7}
            ],
            value=[2, 7],
            multi=True
        ),
        html.Div(children=[dcc.Graph(id='ad-figure')]),
    ], className='columns',)

def summary():
    """
    Returns the text and image of architecture summary of the project.
    """
    return html.Div(children=[
        dcc.Markdown('''
            # Architecture Summary
        ''', className='description_header'),

        html.Div('This is a summary of our website. We join two live datasets, and \
                  upsert the data in the MangoDB database. By using appropriate \
                  queries to the database, we retrieve the data to achieve our \
                  functionality.', className='description'),

        html.Div(children=[
            html.Img(id='archi-img', src='/assets/Data_Flow.png', height=600),
        ], style={'textAlign': 'center'}),
    ])


# Dynamic Page
def dynamic_layout():
    return html.Div([
        page_header(),
        html.Hr(),
        description(),
        dcc.Graph(id='trend-graph', figure=static_stacked_trend_graph(stack=False)),
        slider_plot_description(),
        what_if_tool(),
        pred_description(),
        pred_tool(),
        ad_description(),
        ad_tool(),
        summary(),
    ], className='content', id='content')

app.layout = dynamic_layout

# callbacks for the interactive plot
@app.callback(
    dash.dependencies.Output('slider-figure', 'figure'),
    [dash.dependencies.Input('project-slider', 'value')])

def slider_plot(index):
    # update the plot based on the slider value
    df = fetch_all_voltage_as_df(allow_cached=True)
    x = df['Datetime']
    sources = ['Import', 'Load', 'Generation']
    colors = ['#92a8d1', '#034f84', '#f7cac9', '#f7786b']

    fig = go.Figure()
    if index == 0:
        for i, s in enumerate(sources):
            fig.add_trace(go.Scatter(x=x, y=df[s], mode='lines', name=s,
                          line={'width': 2, 'color': colors[i]}, stackgroup=True))
    
    else:
        fig.add_trace(go.Scatter(x=x, y=df['vol_value'], mode='lines', name='main bus voltage', line={'width': 2, 'color': colors[-1]},
                      fill='tozeroy', stackgroup=False))
        fig.update_layout(yaxis_range=[150,300])

    fig.update_layout(title='Utility/Ashe Main Bus Voltage',
                      plot_bgcolor='#deeaee',
                      paper_bgcolor='#deeaee',
                      yaxis_title='Value',
                      xaxis_title='Date/Time')
    return fig

@app.callback(
    dash.dependencies.Output('pred-figure', 'figure'),
    dash.dependencies.Input('future-slider', 'value'))

def base_imp(day):
    # day should be 4,5,6
    df = fetch_all_voltage_as_df(allow_cached=True)
    df_pred = df[["Datetime","vol_value"]]
    train=df_pred[0:day*288] 
    test=df_pred[day*288:]

    #Aggregating the dataset at daily level
    df_pred.Timestamp = pd.to_datetime(df_pred.Datetime,format='%d-%m-%Y %H:%M') 
    df_pred.index = df_pred.Timestamp 
    df_pred = df_pred.resample('H').mean()
    train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
    train.index = train.Timestamp 
    train = train.resample('H').mean() 
    test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
    test.index = test.Timestamp 
    test = test.resample('H').mean()
    
    dd= np.asarray(train.vol_value)
    y_hat = test.copy()
    y_hat['naive'] = dd[len(dd)-1]
    
    y_hat_avg = test.copy()
    fit1 = ExponentialSmoothing(np.asarray(train['vol_value']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
    y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))

    colors = ['#FFBF00', '#194A8D', '#DE3163', '#B676B1']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['vol_value'], mode='lines', name='Train',
                            line={'width': 2, 'color': colors[0]}))
    fig.add_trace(go.Scatter(x=test.index, y=test['vol_value'], mode='lines', name='Test',
                            line={'width': 2, 'color': colors[1]}))
    fig.add_trace(go.Scatter(x=y_hat.index, y=y_hat['naive'], mode='lines', name='Naive Forecast',
                            line={'width': 2, 'color': colors[2]}))
    fig.add_trace(go.Scatter(x=y_hat_avg.index, y=y_hat_avg['Holt_Winter'], mode='lines', name='Holt_Winter',
                            line={'width': 2, 'color': colors[3]}))
    fig.update_layout(title="Voltage Forecast",
                      plot_bgcolor='#deeaee',
                      paper_bgcolor='#deeaee',
                      yaxis_title='Value',
                      xaxis_title="Time-Step")
    
    return fig


@app.callback(
    dash.dependencies.Output('ad-figure', 'figure'),
    [dash.dependencies.Input('period-dropdown', 'value')])

def detect_abn(period):
    if len(period) < 2:
        raise PreventUpdate
    df = fetch_all_voltage_as_df(allow_cached=True)
    first_day = int(period[0])
    last_day = int(period[1])
    df_ab = df[["Datetime","Generation"]]
    df_small_time = df[["Datetime"]]
    df_small_gene = pd.concat([df[["Generation"]][:200]]*5, ignore_index=True)
    df_small_noise = pd.concat([df_small_time[0:1000],df_small_gene], axis=1)
    df_small_noise = df_small_noise.set_index('Datetime')
    df_daily_jumpsup = df_ab[(first_day-1)*288:(last_day-1)*288].set_index('Datetime')
    # Normalize and save the mean and std we get,
    # for normalizing test data.
    training_mean = df_small_noise.mean()
    training_std = df_small_noise.std()
    df_training_value = (df_small_noise - training_mean) / training_std
    
    TIME_STEPS = 288

    # Generated training sequences for use in the model.
    def create_sequences(values, time_steps=TIME_STEPS):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i : (i + time_steps)])
        return np.stack(output)


    x_train = create_sequences(df_training_value.values)
    
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    
    history = model.fit(
        x_train,
        x_train,
        epochs=10,
        batch_size=128,
        verbose=0,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )
    
    # Get train MAE loss.
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
    
    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    
    df_test_value = (df_daily_jumpsup - training_mean) / training_std

    # Create sequences from test values.
    x_test = create_sequences(df_test_value.values)

    # Get test MAE loss.
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))

    # Detect all the samples which are anomalies.
    anomalies = test_mae_loss > threshold
    
    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_indices = []
    for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
        if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)
            
    df_subset = df_daily_jumpsup.iloc[anomalous_data_indices]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="Normal Generation",
        x=df_daily_jumpsup.index,
        y=df_daily_jumpsup['Generation'],
        marker=dict(
            color="blue"
        ),
        showlegend=True
    ))

    # add line / trace 2 to figure
    fig.add_trace(go.Scatter(
        name="Abnormalies",
        x=df_subset.index,
        y=df_subset['Generation'],
        marker=dict(
            color="red"
        ),
        showlegend=True
    ))
    
    fig.update_layout(title='Main Bus Voltage Anormaly Detection',
                      plot_bgcolor='#deeaee',
                      paper_bgcolor='#deeaee',
                      yaxis_title='Value',
                      xaxis_title='Date/Time')
    
    return fig




if __name__ == '__main__':
    app.run_server(debug=True, port=1050, host='0.0.0.0')
