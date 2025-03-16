import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import multiprocessing as mp
from datetime import datetime
from model_simulation import (
    CreditCardParams,
    simular_escenarios_paralelo,
    guardar_resultados
)

# Set page configuration
st.set_page_config(
    page_title="Credit Card Portfolio Simulator",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 500;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def format_currency(value):
    return f"${value:,.2f}"

def format_percentage(value):
    return f"{value:.2%}"

def create_monthly_metrics_df(results):
    """Create a DataFrame with monthly metrics for all scenarios."""
    all_data = []
    
    for scenario, result in results.items():
        for month_data in result['resultados_mes']:
            month_dict = {
                'Scenario': scenario,
                'Month': month_data['mes_global'],
                'Year': month_data['anio'],
                'Month of Year': month_data['mes'],
                'Commission Income': month_data['ingresos_comisiones'],
                'Interest Income': month_data['ingresos_intereses'],
                'Total Income': month_data['ingresos'],
                'Expenses': month_data['gastos'],
                'Losses': month_data['perdidas'],
                'Net Profit': month_data['ingresos'] - month_data['gastos'] - month_data['perdidas'],
                'Active Clients': month_data['clientes_activos'],
                'Delinquent Clients': month_data['clientes_morosos']
            }
            all_data.append(month_dict)
    
    return pd.DataFrame(all_data)

def create_summary_df(results):
    """Create a summary DataFrame for all scenarios."""
    summary = []
    
    for scenario, result in results.items():
        summary.append({
            'Scenario': scenario,
            'Total Income': result['ingresos_totales'],
            'Total Expenses': result['gastos_totales'],
            'Total Losses': result['perdidas_totales'],
            'Net Profit': result['ganancia_neta'],
            'Delinquency Rate': result['tasa_morosidad'],
            'Final Active Clients': result['clientes_activos_final'],
            'Initial Clients': result['parametros']['num_clientes']
        })
    
    return pd.DataFrame(summary)

def plot_monthly_metric(df, metric, title, y_label):
    """Create a line chart for a monthly metric across all scenarios."""
    fig = px.line(
        df, 
        x='Month', 
        y=metric, 
        color='Scenario',
        title=title,
        labels={'Month': 'Month', metric: y_label},
        markers=True
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    return fig

def plot_scenario_comparison(df, metric, title, y_label):
    """Create a bar chart comparing a metric across scenarios."""
    fig = px.bar(
        df, 
        x='Scenario', 
        y=metric,
        title=title,
        labels={'Scenario': 'Scenario', metric: y_label},
        color='Scenario'
    )
    fig.update_layout(height=500)
    return fig

def plot_income_composition(df):
    """Create a stacked bar chart showing income composition by scenario."""
    # Melt the dataframe to get it in the right format for a stacked bar chart
    melted_df = pd.melt(
        df, 
        id_vars=['Scenario'], 
        value_vars=['Commission Income', 'Interest Income'],
        var_name='Income Type', 
        value_name='Amount'
    )
    
    # Group by scenario and income type to get totals
    grouped_df = melted_df.groupby(['Scenario', 'Income Type'])['Amount'].sum().reset_index()
    
    fig = px.bar(
        grouped_df, 
        x='Scenario', 
        y='Amount', 
        color='Income Type',
        title='Income Composition by Scenario',
        labels={'Amount': 'Total Income', 'Scenario': 'Scenario'},
        barmode='stack'
    )
    fig.update_layout(height=500)
    return fig

def plot_client_distribution(df):
    """Create a stacked bar chart showing client distribution by scenario."""
    # Create a dataframe with active and delinquent clients
    client_df = df[['Scenario', 'Final Active Clients']].copy()
    client_df['Delinquent Clients'] = df['Initial Clients'] - df['Final Active Clients']
    
    # Melt the dataframe
    melted_df = pd.melt(
        client_df, 
        id_vars=['Scenario'], 
        value_vars=['Final Active Clients', 'Delinquent Clients'],
        var_name='Client Type', 
        value_name='Count'
    )
    
    fig = px.bar(
        melted_df, 
        x='Scenario', 
        y='Count', 
        color='Client Type',
        title='Client Distribution by Scenario',
        labels={'Count': 'Number of Clients', 'Scenario': 'Scenario'},
        barmode='stack'
    )
    fig.update_layout(height=500)
    return fig

def plot_profit_waterfall(scenario_data):
    """Create a waterfall chart showing profit breakdown for a scenario."""
    fig = go.Figure(go.Waterfall(
        name="Profit Breakdown",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Total Income", "Expenses", "Losses", "Net Profit"],
        textposition="outside",
        text=[
            format_currency(scenario_data['ingresos_totales']),
            format_currency(-scenario_data['gastos_totales']),
            format_currency(-scenario_data['perdidas_totales']),
            format_currency(scenario_data['ganancia_neta'])
        ],
        y=[
            scenario_data['ingresos_totales'],
            -scenario_data['gastos_totales'],
            -scenario_data['perdidas_totales'],
            0
        ],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Profit Breakdown",
        showlegend=False,
        height=500
    )
    return fig

def plot_monthly_income_breakdown(scenario_data):
    """Create a stacked area chart showing monthly income breakdown."""
    months = [m['mes_global'] for m in scenario_data['resultados_mes']]
    commission = [m['ingresos_comisiones'] for m in scenario_data['resultados_mes']]
    interest = [m['ingresos_intereses'] for m in scenario_data['resultados_mes']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=commission,
        mode='lines',
        line=dict(width=0.5, color='rgb(30, 136, 229)'),
        stackgroup='one',
        name='Commission Income'
    ))
    fig.add_trace(go.Scatter(
        x=months, y=interest,
        mode='lines',
        line=dict(width=0.5, color='rgb(255, 193, 7)'),
        stackgroup='one',
        name='Interest Income'
    ))
    
    fig.update_layout(
        title="Monthly Income Breakdown",
        xaxis_title="Month",
        yaxis_title="Income",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    return fig

def plot_client_evolution(scenario_data):
    """Create a line chart showing the evolution of active and delinquent clients."""
    months = [m['mes_global'] for m in scenario_data['resultados_mes']]
    active = [m['clientes_activos'] for m in scenario_data['resultados_mes']]
    delinquent = [m['clientes_morosos'] for m in scenario_data['resultados_mes']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=active,
        mode='lines+markers',
        name='Active Clients',
        line=dict(color='rgb(76, 175, 80)')
    ))
    fig.add_trace(go.Scatter(
        x=months, y=delinquent,
        mode='lines+markers',
        name='Delinquent Clients',
        line=dict(color='rgb(244, 67, 54)')
    ))
    
    fig.update_layout(
        title="Client Evolution Over Time",
        xaxis_title="Month",
        yaxis_title="Number of Clients",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    return fig

# Initialize session state variables
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'monthly_metrics_df' not in st.session_state:
    st.session_state.monthly_metrics_df = None
if 'summary_df' not in st.session_state:
    st.session_state.summary_df = None
if 'last_simulation_time' not in st.session_state:
    st.session_state.last_simulation_time = None

# Main app header
st.markdown('<div class="main-header">Credit Card Portfolio Simulator</div>', unsafe_allow_html=True)
st.markdown("""
This dashboard allows you to simulate and analyze a credit card portfolio under different scenarios.
Set your parameters, run the simulation, and explore the results in detail.
""")

# Sidebar for parameters
st.sidebar.markdown('<div class="section-header">Simulation Parameters</div>', unsafe_allow_html=True)

# Portfolio size
st.sidebar.markdown('<div class="subsection-header">Portfolio Size</div>', unsafe_allow_html=True)
num_clientes_opt = st.sidebar.number_input("Optimistic Number of Clients", min_value=1, max_value=100000, value=1000, step=1000)
num_clientes_neut = st.sidebar.number_input("Neutral Number of Clients", min_value=1, max_value=100000, value=500, step=1000)
num_clientes_pes = st.sidebar.number_input("Pessimistic Number of Clients", min_value=1, max_value=100000, value=100, step=1000)

# Customer behavior
st.sidebar.markdown('<div class="subsection-header">Customer Behavior</div>', unsafe_allow_html=True)
perc_totaleros_opt = st.sidebar.slider("Optimistic % of Full Payers", min_value=0.1, max_value=0.9, value=0.6, step=0.05, format="%.2f")
perc_totaleros_neut = st.sidebar.slider("Neutral % of Full Payers", min_value=0.1, max_value=0.9, value=0.75, step=0.05, format="%.2f")
perc_totaleros_pes = st.sidebar.slider("Pessimistic % of Full Payers", min_value=0.1, max_value=0.9, value=0.9, step=0.05, format="%.2f")

perc_morosidad_opt = st.sidebar.slider("Optimistic Monthly Delinquency Rate", min_value=0.0, max_value=0.1, value=0.03, step=0.005, format="%.3f")
perc_morosidad_neut = st.sidebar.slider("Neutral Monthly Delinquency Rate", min_value=0.0, max_value=0.1, value=0.05, step=0.005, format="%.3f")
perc_morosidad_pes = st.sidebar.slider("Pessimistic Monthly Delinquency Rate", min_value=0.0, max_value=0.1, value=0.07, step=0.005, format="%.3f")

# Credit line parameters
st.sidebar.markdown('<div class="subsection-header">Credit Line Parameters</div>', unsafe_allow_html=True)
linea_credito_opt = st.sidebar.number_input("Optimistic Average Credit Line", min_value=0, max_value=1000000, value=50000, step=5000)
linea_credito_neut = st.sidebar.number_input("Neutral Average Credit Line", min_value=0, max_value=1000000, value=25000, step=5000)
linea_credito_pes = st.sidebar.number_input("Pessimistic Average Credit Line", min_value=0, max_value=1000000, value=10000, step=5000)

# Utilization parameters
st.sidebar.markdown('<div class="subsection-header">Utilization Parameters</div>', unsafe_allow_html=True)
util_totaleros_opt = st.sidebar.slider("Optimistic Utilization % (Full Payers)", min_value=0.0, max_value=0.9, value=0.4, step=0.05, format="%.2f")
util_totaleros_neut = st.sidebar.slider("Neutral Utilization % (Full Payers)", min_value=0.0, max_value=0.9, value=0.3, step=0.05, format="%.2f")
util_totaleros_pes = st.sidebar.slider("Pessimistic Utilization % (Full Payers)", min_value=0.0, max_value=0.9, value=0.2, step=0.05, format="%.2f")

# Beta distribution parameters for revolving clients
st.sidebar.markdown("Beta Distribution Parameters for Revolving Clients:")
st.sidebar.markdown("""
These parameters control how revolving clients use their credit:

**Alpha (α)**: Controls credit usage behavior near 0%
- Higher values: Clients are less likely to use very little credit
- Lower values: More clients may use very little credit

**Beta (β)**: Controls credit usage behavior near 100%
- Higher values: Clients are less likely to use all their credit
- Lower values: More clients may use most of their credit

Together, they determine the average credit utilization pattern:
- Higher α/β ratio = Higher average credit usage
- Lower α/β ratio = Lower average credit usage
""")

alpha_opt = st.sidebar.slider("Optimistic Alpha", min_value=0.5, max_value=5.0, value=2.0, step=0.1, format="%.1f")
alpha_neut = st.sidebar.slider("Neutral Alpha", min_value=0.5, max_value=5.0, value=1.8, step=0.1, format="%.1f")
alpha_pes = st.sidebar.slider("Pessimistic Alpha", min_value=0.5, max_value=5.0, value=1.5, step=0.1, format="%.1f")

beta_opt = st.sidebar.slider("Optimistic Beta", min_value=1.0, max_value=10.0, value=4.0, step=0.1, format="%.1f")
beta_neut = st.sidebar.slider("Neutral Beta", min_value=1.0, max_value=10.0, value=3.6, step=0.1, format="%.1f")
beta_pes = st.sidebar.slider("Pessimistic Beta", min_value=1.0, max_value=10.0, value=3.0, step=0.1, format="%.1f")

# Payment parameters
st.sidebar.markdown('<div class="subsection-header">Payment Parameters</div>', unsafe_allow_html=True)
pago_minimo_opt = st.sidebar.slider("Optimistic Minimum Payment %", min_value=0.0, max_value=0.5, value=0.025, step=0.01, format="%.2f")
pago_minimo_neut = st.sidebar.slider("Neutral Minimum Payment %", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f")
pago_minimo_pes = st.sidebar.slider("Pessimistic Minimum Payment %", min_value=0.0, max_value=0.5, value=0.075, step=0.01, format="%.2f")

prob_pago_minimo_opt = st.sidebar.slider("Optimistic Probability of Minimum Payment", min_value=0.1, max_value=0.9, value=0.7, step=0.05, format="%.2f")
prob_pago_minimo_neut = st.sidebar.slider("Neutral Probability of Minimum Payment", min_value=0.1, max_value=0.9, value=0.6, step=0.05, format="%.2f")
prob_pago_minimo_pes = st.sidebar.slider("Pessimistic Probability of Minimum Payment", min_value=0.1, max_value=0.9, value=0.5, step=0.05, format="%.2f")

# Financial parameters
st.sidebar.markdown('<div class="subsection-header">Financial Parameters</div>', unsafe_allow_html=True)
tasa_interes_opt = st.sidebar.slider("Optimistic Annual Interest Rate %", min_value=10.0, max_value=90.0, value=60.0, step=1.0, format="%.1f")
tasa_interes_neut = st.sidebar.slider("Neutral Annual Interest Rate %", min_value=10.0, max_value=90.0, value=45.0, step=1.0, format="%.1f")
tasa_interes_pes = st.sidebar.slider("Pessimistic Annual Interest Rate %", min_value=10.0, max_value=90.0, value=30.0, step=1.0, format="%.1f")

comision_venta_opt = st.sidebar.slider("Optimistic Interchange Fee %", min_value=0.0, max_value=0.05, value=0.015, step=0.005, format="%.3f")
comision_venta_neut = st.sidebar.slider("Neutral Interchange Fee %", min_value=0.0, max_value=0.05, value=0.009, step=0.005, format="%.3f")
comision_venta_pes = st.sidebar.slider("Pessimistic Interchange Fee %", min_value=0.0, max_value=0.05, value=0.005, step=0.005, format="%.3f")

costo_emision_opt = st.sidebar.number_input("Optimistic Cost per Card", min_value=0, max_value=2000, value=20, step=50)
costo_emision_neut = st.sidebar.number_input("Neutral Cost per Card", min_value=0, max_value=2000, value=60, step=50)
costo_emision_pes = st.sidebar.number_input("Pessimistic Cost per Card", min_value=0, max_value=2000, value=100, step=50)

# Simulation parameters
st.sidebar.markdown('<div class="subsection-header">Simulation Parameters</div>', unsafe_allow_html=True)
semilla_aleatoria = st.sidebar.number_input("Random Seed", min_value=1, max_value=1000, value=42, step=1)
num_years = st.sidebar.slider("Number of Years to Simulate", min_value=1, max_value=10, value=3, step=1)
num_processes = st.sidebar.slider("Number of Processes", min_value=1, max_value=mp.cpu_count(), value=mp.cpu_count(), step=1)

# Create parameter object
params = CreditCardParams(
    # Portfolio size
    num_clientes=(num_clientes_opt, num_clientes_neut, num_clientes_pes),
    
    # Customer behavior
    perc_totaleros=(perc_totaleros_opt, perc_totaleros_neut, perc_totaleros_pes),
    perc_morosidad=(perc_morosidad_opt, perc_morosidad_neut, perc_morosidad_pes),
    
    # Credit line parameters
    linea_credito_prom=(linea_credito_opt, linea_credito_neut, linea_credito_pes),
    
    # Utilization parameters
    util_credito_totaleros=(util_totaleros_opt, util_totaleros_neut, util_totaleros_pes),
    util_credito_revolventes_alpha=(alpha_opt, alpha_neut, alpha_pes),
    util_credito_revolventes_beta=(beta_opt, beta_neut, beta_pes),
    
    # Payment parameters
    pago_minimo_perc=(pago_minimo_opt, pago_minimo_neut, pago_minimo_pes),
    prob_pago_minimo=(prob_pago_minimo_opt, prob_pago_minimo_neut, prob_pago_minimo_pes),
    
    # Financial parameters
    tasa_interes=(tasa_interes_opt, tasa_interes_neut, tasa_interes_pes),
    comision_venta=(comision_venta_opt, comision_venta_neut, comision_venta_pes),
    costo_emision=(costo_emision_opt, costo_emision_neut, costo_emision_pes),
    
    # Simulation parameters
    semilla_aleatoria=semilla_aleatoria
)

# Run simulation button
st.sidebar.markdown('<div class="subsection-header">Run Simulation</div>', unsafe_allow_html=True)
save_results = st.sidebar.checkbox("Save Results to Disk", value=True)
save_directory = st.sidebar.text_input("Save Directory", value="resultados_simulacion")

if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Running simulation... This may take a few minutes."):
        start_time = time.time()
        
        # Run simulation
        results = simular_escenarios_paralelo(params, num_years=num_years, num_procesos=num_processes)
        
        # Save results if requested
        if save_results:
            guardar_resultados(results, save_directory)
        
        # Store results in session state
        st.session_state.simulation_results = results
        st.session_state.monthly_metrics_df = create_monthly_metrics_df(results)
        st.session_state.summary_df = create_summary_df(results)
        
        end_time = time.time()
        st.session_state.last_simulation_time = end_time - start_time
    
    st.success(f"Simulation completed in {st.session_state.last_simulation_time:.2f} seconds!")

# Display results if available
if st.session_state.simulation_results:
    results = st.session_state.simulation_results
    monthly_df = st.session_state.monthly_metrics_df
    summary_df = st.session_state.summary_df
    
    # Create tabs for different views
    tabs = st.tabs([
        "Summary", 
        "Scenario Comparison", 
        "Monthly Analysis", 
        "Optimistic Scenario", 
        "Neutral Scenario", 
        "Pessimistic Scenario",
        "Raw Data"
    ])
    
    # Summary Tab
    with tabs[0]:
        st.markdown('<div class="section-header">Simulation Summary</div>', unsafe_allow_html=True)
        
        # Display key metrics in a grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Best Net Profit</div>', unsafe_allow_html=True)
            best_profit = summary_df['Net Profit'].max()
            best_scenario = summary_df.loc[summary_df['Net Profit'].idxmax(), 'Scenario']
            st.markdown(f'<div class="metric-value">{format_currency(best_profit)}</div>', unsafe_allow_html=True)
            st.markdown(f'Scenario: {best_scenario}', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Worst Net Profit</div>', unsafe_allow_html=True)
            worst_profit = summary_df['Net Profit'].min()
            worst_scenario = summary_df.loc[summary_df['Net Profit'].idxmin(), 'Scenario']
            st.markdown(f'<div class="metric-value">{format_currency(worst_profit)}</div>', unsafe_allow_html=True)
            st.markdown(f'Scenario: {worst_scenario}', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Profit Range</div>', unsafe_allow_html=True)
            profit_range = best_profit - worst_profit
            st.markdown(f'<div class="metric-value">{format_currency(profit_range)}</div>', unsafe_allow_html=True)
            st.markdown(f'Difference: {format_percentage(profit_range/abs(worst_profit))}', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display summary charts
        st.markdown('<div class="subsection-header">Key Metrics by Scenario</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_scenario_comparison(summary_df, 'Net Profit', 'Net Profit by Scenario', 'Net Profit')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = plot_scenario_comparison(summary_df, 'Delinquency Rate', 'Delinquency Rate by Scenario', 'Delinquency Rate')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_income_composition(monthly_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = plot_client_distribution(summary_df)
            st.plotly_chart(fig, use_container_width=True)
    
    # Scenario Comparison Tab
    with tabs[1]:
        st.markdown('<div class="section-header">Scenario Comparison</div>', unsafe_allow_html=True)
        
        # Display summary table
        st.markdown('<div class="subsection-header">Summary Metrics</div>', unsafe_allow_html=True)
        
        # Format the summary dataframe for display
        display_df = summary_df.copy()
        for col in ['Total Income', 'Total Expenses', 'Total Losses', 'Net Profit']:
            display_df[col] = display_df[col].apply(format_currency)
        display_df['Delinquency Rate'] = display_df['Delinquency Rate'].apply(format_percentage)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Monthly metrics comparison
        st.markdown('<div class="subsection-header">Monthly Metrics Comparison</div>', unsafe_allow_html=True)
        
        metric_to_plot = st.selectbox(
            "Select Metric to Plot",
            options=[
                'Total Income', 
                'Commission Income', 
                'Interest Income', 
                'Net Profit', 
                'Active Clients', 
                'Delinquent Clients'
            ],
            index=3
        )
        
        fig = plot_monthly_metric(
            monthly_df, 
            metric_to_plot, 
            f'{metric_to_plot} Over Time by Scenario', 
            metric_to_plot
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative metrics
        st.markdown('<div class="subsection-header">Cumulative Metrics</div>', unsafe_allow_html=True)
        
        # Calculate cumulative metrics
        cumulative_df = monthly_df.copy()
        cumulative_df['Cumulative Income'] = cumulative_df.groupby('Scenario')['Total Income'].cumsum()
        cumulative_df['Cumulative Expenses'] = cumulative_df.groupby('Scenario')['Expenses'].cumsum()
        cumulative_df['Cumulative Losses'] = cumulative_df.groupby('Scenario')['Losses'].cumsum()
        cumulative_df['Cumulative Profit'] = (
            cumulative_df['Cumulative Income'] - 
            cumulative_df['Cumulative Expenses'] - 
            cumulative_df['Cumulative Losses']
        )
        
        cumulative_metric = st.selectbox(
            "Select Cumulative Metric to Plot",
            options=[
                'Cumulative Income', 
                'Cumulative Expenses', 
                'Cumulative Losses', 
                'Cumulative Profit'
            ],
            index=3
        )
        
        fig = plot_monthly_metric(
            cumulative_df, 
            cumulative_metric, 
            f'{cumulative_metric} Over Time by Scenario', 
            cumulative_metric
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Analysis Tab
    with tabs[2]:
        st.markdown('<div class="section-header">Monthly Analysis</div>', unsafe_allow_html=True)
        
        # Select scenario and metrics
        col1, col2 = st.columns(2)
        
        with col1:
            selected_scenario = st.selectbox(
                "Select Scenario",
                options=monthly_df['Scenario'].unique(),
                index=2  # Default to neutral scenario
            )
        
        with col2:
            selected_metrics = st.multiselect(
                "Select Metrics to Display",
                options=[
                    'Commission Income', 
                    'Interest Income', 
                    'Total Income', 
                    'Expenses', 
                    'Losses', 
                    'Net Profit'
                ],
                default=['Total Income', 'Net Profit']
            )
        
        # Filter data for selected scenario
        scenario_monthly_df = monthly_df[monthly_df['Scenario'] == selected_scenario]
        
        # Plot selected metrics
        if selected_metrics:
            fig = go.Figure()
            
            for metric in selected_metrics:
                fig.add_trace(go.Scatter(
                    x=scenario_monthly_df['Month'],
                    y=scenario_monthly_df[metric],
                    mode='lines+markers',
                    name=metric
                ))
            
            fig.update_layout(
                title=f'Monthly Metrics for {selected_scenario} Scenario',
                xaxis_title='Month',
                yaxis_title='Amount',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly client evolution
        st.markdown('<div class="subsection-header">Client Evolution</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=scenario_monthly_df['Month'],
            y=scenario_monthly_df['Active Clients'],
            mode='lines+markers',
            name='Active Clients'
        ))
        fig.add_trace(go.Scatter(
            x=scenario_monthly_df['Month'],
            y=scenario_monthly_df['Delinquent Clients'],
            mode='lines+markers',
            name='Delinquent Clients'
        ))
        
        fig.update_layout(
            title=f'Client Evolution for {selected_scenario} Scenario',
            xaxis_title='Month',
            yaxis_title='Number of Clients',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonality analysis
        st.markdown('<div class="subsection-header">Seasonality Analysis</div>', unsafe_allow_html=True)
        
        # Add month of year to the dataframe
        scenario_monthly_df['Month Label'] = scenario_monthly_df['Year'].astype(str) + '-' + scenario_monthly_df['Month of Year'].astype(str).str.zfill(2)
        
        # Group by month of year
        monthly_avg = scenario_monthly_df.groupby('Month of Year')[['Total Income', 'Net Profit']].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_avg['Month of Year'],
            y=monthly_avg['Total Income'],
            name='Avg. Total Income'
        ))
        fig.add_trace(go.Bar(
            x=monthly_avg['Month of Year'],
            y=monthly_avg['Net Profit'],
            name='Avg. Net Profit'
        ))
        
        fig.update_layout(
            title=f'Monthly Averages for {selected_scenario} Scenario',
            xaxis_title='Month of Year',
            yaxis_title='Amount',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual Scenario Tabs (Optimistic, Neutral, Pessimistic)
    for i, scenario_name in enumerate(['optimistic', 'neutral', 'pessimistic']):
        with tabs[i+3]:
            scenario_data = results[scenario_name]
            
            st.markdown(f'<div class="section-header">{scenario_name.capitalize()} Scenario Analysis</div>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Net Profit</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{format_currency(scenario_data["ganancia_neta"])}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Income</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{format_currency(scenario_data["ingresos_totales"])}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Delinquency Rate</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{format_percentage(scenario_data["tasa_morosidad"])}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Active Clients</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{scenario_data["clientes_activos_final"]}</div>', unsafe_allow_html=True)
                st.markdown(f'of {scenario_data["parametros"]["num_clientes"]} initial', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Profit breakdown
            st.markdown('<div class="subsection-header">Profit Breakdown</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_profit_waterfall(scenario_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = plot_monthly_income_breakdown(scenario_data)
                st.plotly_chart(fig, use_container_width=True)
            
            # Client analysis
            st.markdown('<div class="subsection-header">Client Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_client_evolution(scenario_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Monthly delinquency rate
                monthly_data = pd.DataFrame([
                    {
                        'Month': m['mes_global'],
                        'Delinquency Rate': m['clientes_morosos'] / (m['clientes_activos'] + m['clientes_morosos'])
                    }
                    for m in scenario_data['resultados_mes']
                ])
                
                fig = px.line(
                    monthly_data,
                    x='Month',
                    y='Delinquency Rate',
                    title='Monthly Delinquency Rate',
                    labels={'Month': 'Month', 'Delinquency Rate': 'Delinquency Rate'},
                    markers=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Monthly metrics
            st.markdown('<div class="subsection-header">Monthly Metrics</div>', unsafe_allow_html=True)
            
            monthly_metrics = pd.DataFrame([
                {
                    'Month': m['mes_global'],
                    'Year': m['anio'],
                    'Month of Year': m['mes'],
                    'Commission Income': m['ingresos_comisiones'],
                    'Interest Income': m['ingresos_intereses'],
                    'Total Income': m['ingresos'],
                    'Expenses': m['gastos'],
                    'Losses': m['perdidas'],
                    'Net Profit': m['ingresos'] - m['gastos'] - m['perdidas'],
                    'Active Clients': m['clientes_activos'],
                    'Delinquent Clients': m['clientes_morosos']
                }
                for m in scenario_data['resultados_mes']
            ])
            
            st.dataframe(monthly_metrics, use_container_width=True)
    
    # Raw Data Tab
    with tabs[6]:
        st.markdown('<div class="section-header">Raw Data</div>', unsafe_allow_html=True)
        
        # Select which data to view
        data_view = st.radio(
            "Select Data to View",
            options=["Summary Data", "Monthly Data", "Client Data", "Parameters"],
            horizontal=True
        )
        
        if data_view == "Summary Data":
            st.markdown('<div class="subsection-header">Summary Data</div>', unsafe_allow_html=True)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary Data as CSV",
                data=csv,
                file_name="credit_card_simulation_summary.csv",
                mime="text/csv"
            )
        
        elif data_view == "Monthly Data":
            st.markdown('<div class="subsection-header">Monthly Data</div>', unsafe_allow_html=True)
            
            # Filter by scenario
            scenario_filter = st.selectbox(
                "Filter by Scenario",
                options=["All Scenarios"] + list(monthly_df['Scenario'].unique())
            )
            
            if scenario_filter == "All Scenarios":
                filtered_df = monthly_df
            else:
                filtered_df = monthly_df[monthly_df['Scenario'] == scenario_filter]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Monthly Data as CSV",
                data=csv,
                file_name=f"credit_card_simulation_monthly_{scenario_filter.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        elif data_view == "Client Data":
            st.markdown('<div class="subsection-header">Client Data (Last Month)</div>', unsafe_allow_html=True)
            
            # Select scenario
            scenario = st.selectbox(
                "Select Scenario",
                options=list(results.keys())
            )
            
            # Get client data for the selected scenario
            client_data = pd.DataFrame(results[scenario]['resultados_clientes'])
            
            st.dataframe(client_data, use_container_width=True)
            
            # Download button
            csv = client_data.to_csv(index=False)
            st.download_button(
                label="Download Client Data as CSV",
                data=csv,
                file_name=f"credit_card_simulation_clients_{scenario}.csv",
                mime="text/csv"
            )
        
        elif data_view == "Parameters":
            st.markdown('<div class="subsection-header">Simulation Parameters</div>', unsafe_allow_html=True)
            
            # Select scenario
            scenario = st.selectbox(
                "Select Scenario",
                options=list(results.keys())
            )
            
            # Get parameters for the selected scenario
            param_data = results[scenario]['parametros']
            
            # Convert to DataFrame for better display
            param_df = pd.DataFrame([(k, v) for k, v in param_data.items()], columns=['Parameter', 'Value'])
            
            st.dataframe(param_df, use_container_width=True)
            
            # Download button
            json_str = json.dumps(param_data, indent=4)
            st.download_button(
                label="Download Parameters as JSON",
                data=json_str,
                file_name=f"credit_card_simulation_parameters_{scenario}.json",
                mime="application/json"
            )
else:
    # Display instructions if no simulation has been run
    st.info("Set your parameters in the sidebar and click 'Run Simulation' to start.")
    
    # Display sample images or explanations
    st.markdown('<div class="section-header">How to Use This Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Set Parameters**: Adjust the simulation parameters in the sidebar. You can set optimistic, neutral, and pessimistic values for each parameter.
    
    2. **Run Simulation**: Click the "Run Simulation" button to start the simulation. This may take a few minutes depending on the number of clients and years.
    
    3. **Explore Results**: Once the simulation is complete, you can explore the results in different tabs:
       - **Summary**: Overview of key metrics across all scenarios
       - **Scenario Comparison**: Compare different scenarios side by side
       - **Monthly Analysis**: Analyze monthly trends for a specific scenario
       - **Individual Scenario Tabs**: Detailed analysis of each scenario
       - **Raw Data**: Access the raw simulation data
    
    4. **Save Results**: You can save the simulation results to disk by checking the "Save Results to Disk" option before running the simulation.
    """)
    
    st.markdown('<div class="section-header">About the Model</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This simulation models a credit card portfolio with the following key components:
    
    - **Client Behavior**: Clients are categorized as either full payers (who pay their balance in full each month) or revolvers (who carry a balance).
    
    - **Credit Utilization**: Full payers use a fixed percentage of their credit line, while revolvers' utilization follows a beta distribution.
    
    - **Delinquency**: Each month, clients have a chance to become delinquent, at which point they stop making payments.
    
    - **Income Sources**: The bank earns income from interchange fees on purchases and interest on revolving balances.
    
    - **Expenses**: The bank incurs costs for issuing and maintaining cards.
    
    - **Seasonality**: The model includes seasonal variations in spending patterns throughout the year.
    """)
