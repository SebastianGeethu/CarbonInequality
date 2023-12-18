import geopandas as gpd
import json
import os
from dash import Dash
import plotly.express as px
from dash.dependencies import Input, Output
from dash import html, dcc
import dash_bootstrap_components as dbc
from functools import reduce
import pandas as pd
from turfpy.measurement import bbox
import plotly.graph_objects as go

# Load your cleaned and combined dataset
df = pd.read_csv('CombinedDoc_new.csv')

# Create a Dash web application
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# def read_lad_geojson(country):
#     country_jsonfile = country + "_LAD_Boundaries.json"
#     if not os.path.exists(country_jsonfile):
#         lad_jsonfile = country + "_LAD_DEC_2022_UK_BFC.json"
#         if not os.path.exists(lad_jsonfile):
#             shapefile = "lad_small.shp"
#             ladgdf = gpd.read_file(shapefile)
#             # Simplify geometry
#             ladgdf.geometry = ladgdf.geometry.simplify(0.001, preserve_topology=True)
#
#             # Select necessary columns
#             ladgdf = ladgdf[['LAD22CD', 'geometry']]
#
#             ladgdf.to_crs(epsg=4326, inplace=True)
#             ladgdf.to_file(lad_jsonfile, driver='GeoJSON')
#             print(lad_jsonfile, " file created")
#         with open(lad_jsonfile) as f:
#             census_lads = json.load(f)
#         os.remove(lad_jsonfile)
#         # lads = census_lads
#         census_lads['features'] = list(filter(
#             lambda f: f['properties']['LAD22CD'].startswith(country[0]), census_lads['features']))
#         with open(country_jsonfile, 'w') as f:
#             json.dump(census_lads, f)
#     else:
#         with open(country_jsonfile) as f:
#             census_lads = json.load(f)
#     return census_lads


def read_lad_geojson(country):
    country_jsonfile = country + "_LAD_Boundaries.json"
    if not os.path.exists(country_jsonfile):
        print("File does not exist")
    else:
        with open(country_jsonfile) as f:
            census_lads = json.load(f)
    return census_lads

def get_max_value(year, country):
    return df[(df['Year'] == year) & (df['Country'] == country)]["Income_Per_Capita"].max()


def get_min_value(year, country):
    return df[(df['Year'] == year) & (df['Country'] == country)]["Income_Per_Capita"].min()


df_dict = {year: df[df["Year"] == year] for year in range(2015, 2022)}
countries = df["Country"].unique().tolist()

lad_ids = [list(map(lambda f: f['properties']['LAD22CD'], read_lad_geojson(country)["features"])) for country in
           countries]


def blank_fig():
    # Blank figure for initial Dash display
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    return fig


map_controls = dbc.Card(
    [
        dbc.Row([
            dbc.Label("Year", html_for="year"),
            dbc.Col(
                [
                    dbc.RadioItems(
                        id='year',
                        options=[{'label': str(i), 'value': i} for i in range(2015, 2022)],
                        value=2015,
                        inline=True
                    ),
                ],
                width=8
            )
        ]),

        dbc.Row([
            dbc.Label('Country',
                      html_for="country"),
            dbc.Col(
                [
                    dbc.Select(
                        id='country',
                        options=[{'label': i, 'value': i} for i in countries],
                        value=countries[0],
                    )
                ],
                width=8
            )
        ]),
    ],
    body=True  # Use body=True to remove the default Card border
)

graph_controls = dbc.Card([
    dbc.Row([
        dbc.Label('Country', html_for="country_choice"),
        dcc.Dropdown(
            id='country_choice',
            options=[{'label': cn, 'value': cn} for cn in sorted(df['Country'].unique())],
            value=sorted(df['Country'].unique())[0],
            multi=True  # Allow multiple selections
        ),
        dbc.Label('Local Authority', html_for="local-authority"),
        dcc.Dropdown(
            id='local-authority',
            options=[],  # Options will be dynamically populated based on the selected country
            value=[sorted(df[df['Country'] == sorted(df['Country'].unique())[0]]['Local_Authority'].unique())[0]],
            multi=True  # Allow multiple selections
        ),
    ]),
], body=True)


@app.callback(
    Output('local-authority', 'options'),
    Input('country_choice', 'value')
)
def update_local_authorities(selected_countries):
    # Convert the input to a list if it's a string
    if isinstance(selected_countries, str):
        selected_countries = [selected_countries]
    # Handle case when no country is selected
    if not selected_countries:
        return []

    # Filter the DataFrame based on selected countries
    filtered_df = df[df['Country'].isin(selected_countries)]

    # Get unique local authorities from the filtered DataFrame
    local_authorities = sorted(filtered_df['Local_Authority'].unique())

    # Create options for the Local Authority dropdown
    options = [{'label': la, 'value': la} for la in local_authorities]

    return options


app.layout = dbc.Container(
    [
        html.H1("Household Carbon Emission vs Income"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(graph_controls, width=2),
                dbc.Col([
                    html.Div([dcc.Graph(id='income-vs-years-plot')], style={'display': 'inline-block', 'width': '50%'}),
                    html.Div([dcc.Graph(id='total-usage-vs-years-plot')],
                             style={'display': 'inline-block', 'width': '50%'}),
                    html.Div([dcc.Graph(id='gas-usage-vs-years-plot')],
                             style={'display': 'inline-block', 'width': '50%'}),
                    html.Div([dcc.Graph(id='electricity-usage-vs-years-plot')],
                             style={'display': 'inline-block', 'width': '50%'}),
                    html.Div([dcc.Graph(id='other-usage-vs-years-plot')],
                             style={'display': 'inline-block', 'width': '50%'}),

                ], width=10),

            ],
        ),
        dbc.Row(
            [
                dbc.Col([
                    dbc.Col(map_controls, width=2),
                    dbc.Col(html.Div([dcc.Graph(id='map', figure=blank_fig())]))
                ], width=10),
            ]
        ),
    ],
    fluid=True,
)


# def update_local_authorities_callback(selected_countries):
#     return update_local_authorities(selected_countries)


@app.callback(
    Output('map', 'figure'),

    [Input('year', 'value'),
     Input('country', 'value')]
)
def update_graph_and_local_authorities(year, country):
    lad_df = df_dict[year]
    lad_max_value = get_max_value(year, country)
    lad_min_value = get_min_value(year, country)
    gj = read_lad_geojson(country)
    gj_bbox = reduce(lambda b1, b2: [min(b1[0], b2[0]), min(b1[1], b2[1]),
                                     max(b1[2], b2[2]), max(b1[3], b2[3])],
                     map(lambda f: bbox(f['geometry']), gj['features']))

    fig = px.choropleth(lad_df,
                        geojson=gj,
                        locations="Local_Authority_Code",
                        color="Income_Per_Capita",
                        color_continuous_scale=['#CCFFCC', '#3399FF', '#000066'],
                        range_color=(lad_min_value, lad_max_value),
                        featureidkey="properties.LAD22CD",
                        scope='europe',
                        hover_data=["Local_Authority", "Income Group"],
                        title=f"Per Capita Income by Local Authority ({year})",
                        # projection="natural earth"
                        )

    fig.update_geos(
        center_lon=(gj_bbox[0] + gj_bbox[2]) / 2.0,
        center_lat=(gj_bbox[1] + gj_bbox[3]) / 2.0,
        lonaxis_range=[gj_bbox[0], gj_bbox[2]],
        lataxis_range=[gj_bbox[1], gj_bbox[3]],
        visible=False
    )

    # fig.update_traces(hoverinfo="location+z")
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>" +
                                    "Income: %{z:.2f}<br>" +
                                    "Income Group: %{customdata[1]}<extra></extra>",
                      selector=dict(type="choropleth"))

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                      title_x=0.5,
                      width=1200, height=600)

    return fig


@app.callback(
    [
        Output('income-vs-years-plot', 'figure'),
        Output('gas-usage-vs-years-plot', 'figure'),
        Output('electricity-usage-vs-years-plot', 'figure'),
        Output('other-usage-vs-years-plot', 'figure'),
        Output('total-usage-vs-years-plot', 'figure')],
    Input('local-authority', 'value')

)
def update_plots(selected_local_authorities):
    filtered_data = df[df['Local_Authority'].isin(selected_local_authorities)]
    income_vs_years_plot = px.line(filtered_data, x='Year', y='Income_Per_Capita', color='Local_Authority')
    gas_usage_vs_years_plot = px.line(filtered_data, x='Year', y='Gas_Per_Capita', color='Local_Authority')
    electricity_usage_vs_years_plot = px.line(filtered_data, x='Year', y='Electricity_Per_Capita',
                                              color='Local_Authority')
    other_usage_vs_years_plot = px.line(filtered_data, x='Year', y='Other_Per_Capita', color='Local_Authority')
    total_usage_vs_years_plot = px.line(filtered_data, x='Year', y='Total_Per_Capita', color='Local_Authority')

    income_vs_years_plot.update_layout(
        yaxis_title='Per Capita Income'
    )
    gas_usage_vs_years_plot.update_layout(
        yaxis_title='Per Capita Gas Emission(CO2)'
    )
    electricity_usage_vs_years_plot.update_layout(
        yaxis_title='Per Capita Electricity Emission(CO2)'
    )
    other_usage_vs_years_plot.update_layout(
        yaxis_title='Per Capita Emission- Others(CO2)'
    )
    total_usage_vs_years_plot.update_layout(
        yaxis_title='Per Capita Total Emission(CO2)'
    )

    return income_vs_years_plot, gas_usage_vs_years_plot, electricity_usage_vs_years_plot, other_usage_vs_years_plot, total_usage_vs_years_plot


# Run the Dash application
if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False, host = "0.0.0.0", port = 8080)
