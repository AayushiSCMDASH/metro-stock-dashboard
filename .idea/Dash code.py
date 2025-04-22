# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================

from dash import ctx
import dash
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import base64
from dash import dcc, html, dash_table, Input, Output

import matplotlib
matplotlib.use('Agg')  # 🔵 Critical Fix!
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# =============================================================================
# 2. LOAD DATA
# =============================================================================
file_path = r"C:\Users\aayushi.trivedi\Desktop\SAP_Consolidated_Output.csv"

df = pd.read_csv(file_path, low_memory=False)
df.columns = df.columns.str.strip()

for col in ["Site", "Vendor Name", "Article", "RM"]:
    df[col] = df[col].astype(str).str.strip()

df["Article Description"] = df.get("Article Desc", "No Description")
df = df[df["RM"] == "KADARAPPA"].copy()

for col in ["DMS-30", "Stock", "In-Transit Stock", "On_Order_qty", "MOQ qualification", "PO QTY -EA", "GRN QTY-EA", "Lead time"]:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

for col in ["PO Creation Date", "Last GRN Date_x"]:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# =============================================================================
# 3. PRE-CALCULATIONS
# =============================================================================
df["14_Days_Demand"] = df["DMS-30"] * 14
df["Supply_Available"] = df["Stock"] + df["In-Transit Stock"] + df["On_Order_qty"]

df["is_Ordering_Gap"] = (df["14_Days_Demand"] >= df["Supply_Available"]) & (df["Gap Article"] == 1)
df["MOQ_Qualification_%"] = df["MOQ qualification"] * 100
df["is_MOQ_Issue"] = (df["MOQ_Qualification_%"] < 70) & (df["Gap Article"] == 1)
df["Fill_Rate_%"] = np.where(df["PO QTY -EA"] > 0, (df["GRN QTY-EA"] / df["PO QTY -EA"]) * 100, np.nan)
df["is_Fill_Rate_Issue"] = (df["Fill_Rate_%"] < 100) & (df["Gap Article"] == 1)
df["Actual_Lead_Time"] = (df["Last GRN Date_x"] - df["PO Creation Date"]).dt.days
df["is_Lead_Time_Issue"] = (df["Actual_Lead_Time"] > df["Lead time"]) & (df["Gap Article"] == 1)
df["is_Others"] = ~(df["is_Ordering_Gap"] | df["is_MOQ_Issue"] | df["is_Fill_Rate_Issue"] | df["is_Lead_Time_Issue"]) & (df["Gap Article"] == 1)

conditions = [
    df["is_Ordering_Gap"],
    df["is_MOQ_Issue"],
    df["is_Fill_Rate_Issue"],
    df["is_Lead_Time_Issue"]
]
choices = ["Ordering Gap", "MOQ Issue", "Fill Rate", "Lead Time"]
df["Final_Reason"] = np.select(conditions, choices, default="Others")

# =============================================================================
# 4. KPI & CHART PREPARATION
# =============================================================================
total_articles = len(df)
out_of_stock_count = df["Gap Article"].sum()
availability_percent = (total_articles - out_of_stock_count) / total_articles * 100

# Pie Chart
pie_fig = px.pie(
    names=["Available", "Out-of-Stock"],
    values=[total_articles - out_of_stock_count, out_of_stock_count],
    hole=0.4
).update_traces(textinfo='percent+label', pull=[0, 0.1]).update_layout(title_x=0.5)

# Reason Bar Chart
reason_counts = df[df["Gap Article"] == 1].groupby("Final_Reason").size().reset_index(name="Count")
reason_counts["% of Total"] = reason_counts["Count"] / total_articles * 100
reason_order = ["Ordering Gap", "MOQ Issue", "Fill Rate", "Lead Time", "Others"]
reason_counts["Final_Reason"] = pd.Categorical(reason_counts["Final_Reason"], categories=reason_order, ordered=True)
reason_counts = reason_counts.sort_values("Final_Reason")

bar_fig = px.bar(
    reason_counts,
    x="Final_Reason",
    y="% of Total",
    color="% of Total",
    color_continuous_scale="Reds"
).update_layout(
    title_x=0.5,
    xaxis_title="Out of Stock Reason",
    yaxis_title="% of Total Articles",
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)

site_options = [{'label': site, 'value': site} for site in df['Site'].dropna().unique()]
vendor_options = [{'label': vendor, 'value': vendor} for vendor in df['Vendor Name'].dropna().unique()]


# =============================================================================
# 5. DASH APP SETUP
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Metro Cash & Carry - Stock-Out Insights"


# =============================================================================
# 6. LAYOUT
# =============================================================================
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Metro Cash & Carry: Stock-Out Insights", style={
            "backgroundColor": "#003366",
            "color": "white",
            "padding": "10px 20px",
            "borderRadius": "10px",
            "fontWeight": "bold",
            "fontSize": "1.5rem",
            "width": "fit-content"
        }), width="auto")
    ], justify="start", style={"marginBottom": "20px"}),

    # KPI Cards
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Total Articles", style={"textAlign": "center", "fontWeight": "bold"}),
            html.H3(f"{total_articles}", style={"color": "#17a2b8", "textAlign": "center"})
        ]), style={
            "border": "2px solid #ced4da",
            "borderRadius": "12px",
            "boxShadow": "2px 2px 8px rgba(0,0,0,0.1)",
            "transition": "transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out",
            "cursor": "pointer"
        })),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Out-of-Stock Articles", style={"textAlign": "center", "fontWeight": "bold"}),
            html.H3(f"{out_of_stock_count}", style={"color": "#dc3545", "textAlign": "center"})
        ]), style={
            "border": "2px solid #ced4da",
            "borderRadius": "12px",
            "boxShadow": "2px 2px 8px rgba(0,0,0,0.1)",
            "transition": "transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out",
            "cursor": "pointer"
        })),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Availability %", style={"textAlign": "center", "fontWeight": "bold"}),
            html.H3(f"{availability_percent:.2f}%", style={"color": "#28a745", "textAlign": "center"})
        ]), style={
            "border": "2px solid #ced4da",
            "borderRadius": "12px",
            "boxShadow": "2px 2px 8px rgba(0,0,0,0.1)",
            "transition": "transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out",
            "cursor": "pointer"
        }))
    ], className="mb-4"),

    # Overall Stock Section
    html.H3("Overall Stock Status", style={"textAlign": "left", "fontWeight": "bold"}),
    html.Hr(),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            dcc.Graph(figure=pie_fig)
        ]), style={
            "border": "2px solid #ced4da",
            "borderRadius": "12px",
            "padding": "10px",
            "boxShadow": "2px 2px 8px rgba(0,0,0,0.1)",
            "transition": "transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out",
            "cursor": "pointer"
        }), md=6),

        dbc.Col(dbc.Card(dbc.CardBody([
            dcc.Graph(figure=bar_fig)
        ]), style={
            "border": "2px solid #ced4da",
            "borderRadius": "12px",
            "padding": "10px",
            "boxShadow": "2px 2px 8px rgba(0,0,0,0.1)",
            "transition": "transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out",
            "cursor": "pointer"
        }), md=6)
    ], className="mb-4"),

    html.H3("Out of Stock Buckets", style={"textAlign": "left", "fontWeight": "bold"}),
    html.Hr(),

    # Main Tabs
    dcc.Tabs(
        id="main-tabs",
        value="Inventory Planning & MOQ",
        children=[
            dcc.Tab(label="📦 Inventory Planning & MOQ", value="Inventory Planning & MOQ"),
            dcc.Tab(label="🔥 Demand & Pricing", value="Demand & Pricing"),
            dcc.Tab(label="🤝 Vendor Commitment", value="Vendor Commitment"),
            dcc.Tab(label="🛠 Operational & Quality", value="Operational & Quality")
        ]
    ),
    html.Div(id="tabs-content")
], fluid=True)

@app.callback(
    Output("tabs-content", "children"),
    [Input("main-tabs", "value")]
)
def update_main_tab(selected_main_tab):
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="site-filter", options=site_options, placeholder="Filter by Site", multi=True), md=6),
            dbc.Col(dcc.Dropdown(id="vendor-filter", options=vendor_options, placeholder="Filter by Vendor", multi=True), md=6)
        ], className="mb-4"),

        html.Div(id="sub-tabs-content", style={"border": "2px solid #ced4da", "borderRadius": "8px", "padding": "20px", "backgroundColor": "white"})
    ])

@app.callback(
    Output("sub-tabs-content", "children"),
    [Input("main-tabs", "value"),
     Input("site-filter", "value"),
     Input("vendor-filter", "value")]
)
def render_sub_tabs(selected_main_tab, selected_sites, selected_vendors):
    df_filtered = df.copy()

    if selected_sites:
        df_filtered = df_filtered[df_filtered["Site"].isin(selected_sites)]
    if selected_vendors:
        df_filtered = df_filtered[df_filtered["Vendor Name"].isin(selected_vendors)]

    if selected_main_tab == "Inventory Planning & MOQ":
        return inventory_moq_tab(df_filtered)
    elif selected_main_tab == "Demand & Pricing":
        return demand_pricing_tab(df_filtered)
    elif selected_main_tab == "Vendor Commitment":
        return vendor_commitment_tab(df_filtered)
    elif selected_main_tab == "Operational & Quality":
        return operational_quality_tab(df_filtered)
    else:
        return html.Div("No data available.")

def inventory_moq_tab(df_filtered):
    # Create Sets
    ordering_gap_set = set(df_filtered[df_filtered["is_Ordering_Gap"]]["Article"])
    moq_issue_set = set(df_filtered[df_filtered["is_MOQ_Issue"]]["Article"])

    # Define Venn Category
    def get_venn_category(article):
        if article in ordering_gap_set and article in moq_issue_set:
            return "🔀 Ordering Gap linked to MOQ Issue"
        elif article in ordering_gap_set:
            return "📦 Pure Ordering Gap"
        elif article in moq_issue_set:
            return "📏 Pure MOQ Issue"
        else:
            return "Others"

    df_filtered["Venn Category"] = df_filtered["Article"].apply(get_venn_category)

    # Prepare filtered DataFrames
    ordering_gap_df = df_filtered[df_filtered["is_Ordering_Gap"]]
    moq_issue_df = df_filtered[df_filtered["is_MOQ_Issue"]]

    ordering_gap_count = len(ordering_gap_df)
    moq_issue_count = len(moq_issue_df)

    # Venn Diagram
    plt.figure(figsize=(2.8, 2.8))
    venn2([ordering_gap_set, moq_issue_set], set_labels=('Ordering Gap', 'MOQ Issue'))
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close()
    venn_img = f"data:image/png;base64,{base64.b64encode(buf.getbuffer()).decode('ascii')}"

    return html.Div([
        html.Div([
            html.Img(src=venn_img, style={"display": "block", "margin": "auto", "width": "30%", "height": "30%"})
        ], style={"marginTop": "-20px"}),
        html.Hr(),

        dcc.Tabs([
            dcc.Tab(label=f"📦 Ordering Gap ({ordering_gap_count})", children=[
                html.H4([
                    "📦 Ordering Gap: ",
                    html.Span(f"{ordering_gap_count} articles", style={"color": "red"})
                ], style={"textAlign": "center", "marginBottom": "20px", "fontWeight": "bold"}),

                generate_table(ordering_gap_df, reason_type="Ordering Gap"),

                html.Button(
                    "⬇️ Download Ordering Gap Data",
                    id="btn-download-ordering-gap",
                    n_clicks=0,
                    style={"margin": "20px auto", "display": "block", "backgroundColor": "#003366", "color": "white", "padding": "10px", "borderRadius": "8px"}
                )
            ]),

            dcc.Tab(label=f"📏 MOQ Issue ({moq_issue_count})", children=[
                html.H4([
                    "📏 MOQ Issue: ",
                    html.Span(f"{moq_issue_count} articles", style={"color": "red"})
                ], style={"textAlign": "center", "marginBottom": "20px", "fontWeight": "bold"}),

                generate_table(moq_issue_df, reason_type="MOQ Issue"),

                html.Button(
                    "⬇️ Download MOQ Issue Data",
                    id="btn-download-moq-issue",
                    n_clicks=0,
                    style={"margin": "20px auto", "display": "block", "backgroundColor": "#003366", "color": "white", "padding": "10px", "borderRadius": "8px"}
                )
            ])
        ]),

        dcc.Download(id="download-dataframe-xlsx")
    ])


def demand_pricing_tab(df_filtered):
    return html.Div([
        html.H4("Demand and Pricing Analysis", style={"textAlign": "center", "fontWeight": "bold"}),
        html.P("Calculation logic for High Sale and Pricing Issues will be added later.", style={"textAlign": "center"})
    ])

def vendor_commitment_tab(df_filtered):
    fill_rate_df = df_filtered[df_filtered["is_Fill_Rate_Issue"]]
    lead_time_df = df_filtered[df_filtered["is_Lead_Time_Issue"]]

    fill_fig = px.bar(fill_rate_df.groupby("Vendor Name").size().reset_index(name="Count"),
                      x="Vendor Name", y="Count", color="Count", color_continuous_scale="Reds")

    bullet_fig = go.Figure()
    bullet_fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=lead_time_df["Actual_Lead_Time"].mean() if not lead_time_df.empty else 0,
        delta={"reference": lead_time_df["Lead time"].mean() if not lead_time_df.empty else 0},
        title={"text": "Avg Actual Lead Time vs System Lead Time"},
        gauge={
            "axis": {"range": [0, max(30, lead_time_df["Actual_Lead_Time"].max() if not lead_time_df.empty else 30)]},
            "bar": {"color": "blue"}
        }
    ))

    return html.Div([
        dcc.Tabs([
            dcc.Tab(label=f"📈 Fill Rate ", children=[
                html.H4([
                    "📈 Fill Rate Issues: ",
                    html.Span(f"{len(fill_rate_df)} articles", style={"color": "red"})
                ], style={"textAlign": "center", "marginBottom": "20px", "fontWeight": "bold"}),
                dcc.Graph(figure=fill_fig),
                generate_table(fill_rate_df, reason_type="Fill Rate")
            ]),
            dcc.Tab(label=f"⏱ Lead Time ", children=[
                html.H4([
                    "⏱ Lead Time Issues: ",
                    html.Span(f"{len(lead_time_df)} articles", style={"color": "red"})
                ], style={"textAlign": "center", "marginBottom": "20px", "fontWeight": "bold"}),
                dcc.Graph(figure=bullet_fig),
                generate_table(lead_time_df, reason_type="Lead Time")
            ])
        ])
    ])

def operational_quality_tab(df_filtered):
    others_df = df_filtered[df_filtered["is_Others"]]

    if others_df.empty:
        return html.H5("No articles found under Operational & Quality.", style={"textAlign": "center", "color": "red"})

    return html.Div([
        html.H4([
            "🛠 Operational & Quality Issues: ",
            html.Span(f"{len(others_df)} articles", style={"color": "red"})
        ], style={"textAlign": "center", "marginBottom": "20px", "fontWeight": "bold"}),
        generate_table(others_df, reason_type="Others")
    ])

def generate_table(dataframe, reason_type):
    base_columns = ["Site", "Article", "Article Description", "Area", "Buyer Name", "Vendor", "Vendor Name", "ABC Class", "Venn Category"]

    extra_columns = []
    if reason_type == "Ordering Gap":
        extra_columns = ["14_Days_Demand", "Supply_Available"]
    elif reason_type == "MOQ Issue":
        extra_columns = ["MOQ_Qualification_%"]
    elif reason_type == "Fill Rate":
        extra_columns = ["PO QTY -EA", "GRN QTY-EA", "Fill_Rate_%"]
    elif reason_type == "Lead Time":
        extra_columns = ["PO Creation Date", "Last GRN Date_x", "Lead time", "Actual_Lead_Time"]

    selected_columns = base_columns + extra_columns
    selected_columns = [col for col in selected_columns if col in dataframe.columns]

    df_display = dataframe[selected_columns].copy()
    for col in df_display.select_dtypes(include=[np.number]).columns:
        df_display[col] = df_display[col].round(2)

    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df_display.columns],
        data=df_display.to_dict('records'),
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"padding": "8px", "fontSize": "1rem", "textAlign": "center"},
        style_header={
            "backgroundColor": "#003366",
            "color": "white",
            "fontWeight": "bold",
            "fontSize": "1rem"
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
            {'if': {'state': 'active'}, 'backgroundColor': '#e9ecef', 'border': '1px solid #adb5bd'}
        ],
        filter_action="native",
        sort_action="native"
    )

@app.callback(
    Output("download-dataframe-xlsx", "data"),
    [
        Input("btn-download-ordering-gap", "n_clicks"),
        Input("btn-download-moq-issue", "n_clicks"),
    ],
    [
        dash.dependencies.State("site-filter", "value"),
        dash.dependencies.State("vendor-filter", "value"),
    ],
    prevent_initial_call=True,
)
def download_filtered_data(ordering_gap_click, moq_issue_click, selected_sites, selected_vendors):
    df_filtered = df.copy()

    if selected_sites:
        df_filtered = df_filtered[df_filtered["Site"].isin(selected_sites)]
    if selected_vendors:
        df_filtered = df_filtered[df_filtered["Vendor Name"].isin(selected_vendors)]

    triggered_id = dash.ctx.triggered_id

    if triggered_id == "btn-download-ordering-gap":
        data_to_download = df_filtered[df_filtered["is_Ordering_Gap"]]
        filename = "Ordering_Gap_Articles.xlsx"
    elif triggered_id == "btn-download-moq-issue":
        data_to_download = df_filtered[df_filtered["is_MOQ_Issue"]]
        filename = "MOQ_Issue_Articles.xlsx"
    else:
        return dash.no_update

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        data_to_download.to_excel(writer, index=False, sheet_name="Data")
    output.seek(0)

    return dcc.send_bytes(output.read(), filename)


if __name__ == "__main__":
    app.run(debug=True, port=8026)
