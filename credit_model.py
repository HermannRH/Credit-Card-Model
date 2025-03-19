import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import multiprocessing as mp
import time
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model_simulation import (
    CreditCardParams,
    simular_escenarios_secuencial
)

# Set page configuration
st.set_page_config(
    page_title="Credit Card Portfolio Simulator",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions for visualization
def create_box_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """Create a box plot using Plotly."""
    fig = px.box(df, x=x_col, y=y_col, title=title)
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        height=500,
        margin=dict(t=50, b=50)
    )
    return fig

def create_line_plot_with_range(df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    """Create a line plot with confidence intervals using Plotly."""
    fig = go.Figure()
    
    for scenario in df['scenario'].unique():
        scenario_data = df[df['scenario'] == scenario].sort_values('month')
        
        # Add main line
        fig.add_trace(go.Scatter(
            x=scenario_data['month'],
            y=scenario_data[f'{y_label}_mean'],
            name=f"{scenario}",
            mode='lines',
            line=dict(width=2)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=scenario_data['month'],
            y=scenario_data[f'{y_label}_mean'] + 2*scenario_data[f'{y_label}_std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name=f"{scenario} Upper"
        ))
        fig.add_trace(go.Scatter(
            x=scenario_data['month'],
            y=scenario_data[f'{y_label}_mean'] - 2*scenario_data[f'{y_label}_std'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0,0,0,0.1)',
            fill='tonexty',
            showlegend=False,
            name=f"{scenario} Lower"
        ))
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=20,
        height=500,
        hovermode='x unified',
        margin=dict(t=50, b=50),
        xaxis_title="Month",
        yaxis_title=y_label.replace('_', ' ').title()
    )
    return fig

def create_stacked_area_plot(df: pd.DataFrame, title: str) -> go.Figure:
    """Create a stacked area plot using Plotly."""
    fig = go.Figure()
    
    for column in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[column],
            name=column,
            stackgroup='one',
            mode='lines',
            line=dict(width=0.5),
            hovertemplate="%{y:,.2f}"
        ))
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=20,
        height=500,
        hovermode='x unified',
        margin=dict(t=50, b=50),
        xaxis_title="Month",
        yaxis_title="Amount"
    )
    return fig

def create_distribution_plot(df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    """Create a distribution plot using Plotly."""
    fig = go.Figure()
    
    for scenario in df['scenario'].unique():
        scenario_data = df[df['scenario'] == scenario][value_col]
        
        fig.add_trace(go.Violin(
            y=scenario_data,
            name=scenario,
            box_visible=True,
            meanline_visible=True
        ))
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=20,
        height=500,
        margin=dict(t=50, b=50),
        xaxis_title="Scenario",
        yaxis_title=value_col.replace('_', ' ').title()
    )
    return fig

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

def process_simulation_results(results_df):
    """
    Process the raw simulation results into summary and monthly DataFrames
    
    Args:
        results_df: DataFrame with all simulation results
        
    Returns:
        Tuple of (summary_stats_df, monthly_stats_df)
    """
    # Split into summary and monthly results
    summary_df = results_df[results_df['simulation_type'] == 'summary']
    monthly_df = results_df[results_df['simulation_type'] == 'monthly']
    
    # Calculate summary statistics by scenario
    summary_stats = []
    for scenario, group in summary_df.groupby('scenario'):
        stats = {
            'scenario': scenario,
            'net_profit_mean': group['net_profit'].mean(),
            'net_profit_std': group['net_profit'].std(),
            'net_profit_min': group['net_profit'].min(),
            'net_profit_max': group['net_profit'].max(),
            'delinquency_rate_mean': group['delinquency_rate'].mean(),
            'delinquency_rate_std': group['delinquency_rate'].std(),
            'active_clients_final_mean': group['active_clients_final'].mean(),
            'active_clients_final_std': group['active_clients_final'].std(),
            'total_income_mean': group['total_income'].mean(),
            'total_expenses_mean': group['total_expenses'].mean(),
            'total_losses_mean': group['total_losses'].mean(),
            'num_clients_initial': group['num_clients_initial'].iloc[0]
        }
        summary_stats.append(stats)
    
    summary_stats_df = pd.DataFrame(summary_stats)
    
    # Calculate monthly statistics by scenario and month
    monthly_stats = []

    for (scenario, month), group in monthly_df.groupby(['scenario', 'month']):
        stats = {
            'scenario': scenario,
            'month': month,
            'year': group['year'].iloc[0],
            'month_of_year': group['month_of_year'].iloc[0],
            'commission_income_mean': group['commission_income'].mean(),
            'commission_income_std': group['commission_income'].std(),
            'interest_income_mean': group['interest_income'].mean(),
            'interest_income_std': group['interest_income'].std(),
            'total_income_mean': group['total_income'].mean(),
            'total_income_std': group['total_income'].std(),
            'expenses_mean': group['expenses'].mean(),
            'expenses_std': group['expenses'].std(),
            'losses_mean': group['losses'].mean(),
            'losses_std': group['losses'].std(),
            'net_profit_mean': group['net_profit'].mean(),
            'net_profit_std': group['net_profit'].std(),
            'active_clients_mean': group['active_clients'].mean(),
            'active_clients_std': group['active_clients'].std(),
            'delinquent_clients_mean': group['delinquent_clients'].mean(),
            'delinquent_clients_std': group['delinquent_clients'].std(),
            'interes_generado_total': group['interes_generado_total'].mean(),
            'interes_generado_total_std': group['interes_generado_total'].std(),
            'pagos_interes_total': group['pagos_interes_total'].mean(),
            'pagos_interes_total_std': group['pagos_interes_total'].std(),
            'saldo_principal_total': group['saldo_principal_total'].mean(),
            'saldo_principal_total_std': group['saldo_principal_total'].std(),
            'saldo_interes_total': group['saldo_interes_total'].mean(),
            'saldo_interes_total_std': group['saldo_interes_total'].std(),
            
        }
        monthly_stats.append(stats)
    
    monthly_stats_df = pd.DataFrame(monthly_stats)
    
    return summary_stats_df, monthly_stats_df

# Initialize session state variables
if 'simulation_results_df' not in st.session_state:
    st.session_state.simulation_results_df = None
if 'summary_stats_df' not in st.session_state:
    st.session_state.summary_stats_df = None
if 'monthly_stats_df' not in st.session_state:
    st.session_state.monthly_stats_df = None
if 'last_simulation_time' not in st.session_state:
    st.session_state.last_simulation_time = None
if 'csv_path' not in st.session_state:
    st.session_state.csv_path = None

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
num_clientes_opt = st.sidebar.number_input("Optimistic Number of Clients", min_value=1, max_value=100000, value=100, step=1000)
num_clientes_neut = st.sidebar.number_input("Neutral Number of Clients", min_value=1, max_value=100000, value=100, step=1000)
num_clientes_pes = st.sidebar.number_input("Pessimistic Number of Clients", min_value=1, max_value=100000, value=100, step=1000)

# Customer behavior
st.sidebar.markdown('<div class="subsection-header">Customer Behavior</div>', unsafe_allow_html=True)
perc_totaleros_opt = st.sidebar.slider("Optimistic % of Full Payers", min_value=0.0, max_value=0.9, value=0.6, step=0.05, format="%.2f")
perc_totaleros_neut = st.sidebar.slider("Neutral % of Full Payers", min_value=0.0, max_value=0.9, value=0.75, step=0.05, format="%.2f")
perc_totaleros_pes = st.sidebar.slider("Pessimistic % of Full Payers", min_value=0.0, max_value=0.9, value=0.9, step=0.05, format="%.2f")

perc_morosidad_opt = st.sidebar.slider("Optimistic Monthly Delinquency Rate", min_value=0.0, max_value=0.01, value=0.003, step=0.0001, format="%.3f")
perc_morosidad_neut = st.sidebar.slider("Neutral Monthly Delinquency Rate", min_value=0.0, max_value=0.01, value=0.004, step=0.0001, format="%.3f")
perc_morosidad_pes = st.sidebar.slider("Pessimistic Monthly Delinquency Rate", min_value=0.0, max_value=0.01, value=0.005, step=0.0001, format="%.3f")

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
num_years = st.sidebar.slider("Number of Years to Simulate", min_value=1, max_value=10, value=3, step=1)
num_seeds = st.sidebar.number_input("Number of Seeds", min_value=1, max_value=10000, value=100, step=50)
start_up_time = st.sidebar.slider("Start-Up Time (months)", min_value=1, max_value=36, value=12, step=1, 
                                 help="Number of months to reach full customer base. In month 1, 50% of customers are added. The remaining 50% are added in month 2.")

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
    semilla_aleatoria=0,  # This will be overridden by the simulation
    start_up_time=start_up_time  # Add start-up time parameter
)

# Run simulation button
st.sidebar.markdown('<div class="subsection-header">Run Simulation</div>', unsafe_allow_html=True)
save_directory = st.sidebar.text_input("Save Directory", value="resultados_simulacion")

if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner(f"Running {num_seeds} simulations per scenario... This may take a few minutes."):
        start_time = time.time()
        
        # Run simulation with multiple seeds
        st.text("Step 1/3: Running Monte Carlo simulations...")
        csv_path = simular_escenarios_secuencial(params, num_years=num_years, num_seeds=num_seeds, directorio=save_directory)
        st.session_state.csv_path = csv_path
        
        # Load the saved CSV for analysis
        st.text("Step 2/3: Processing results for visualization...")
        results_df = pd.read_csv(csv_path)
        
        # Process results into summary and monthly statistics
        st.text("Step 3/3: Generating visualizations...")
        summary_stats_df, monthly_stats_df = process_simulation_results(results_df)
        
        # Store in session state
        st.session_state.simulation_results_df = results_df
        st.session_state.summary_stats_df = summary_stats_df
        st.session_state.monthly_stats_df = monthly_stats_df
        
        end_time = time.time()
        st.session_state.last_simulation_time = end_time - start_time
    
    st.success(f"Simulation completed in {st.session_state.last_simulation_time:.2f} seconds!")


# Display results if available
if st.session_state.simulation_results_df is not None:
    results_df = st.session_state.simulation_results_df
    summary_df = st.session_state.summary_stats_df
    monthly_df = st.session_state.monthly_stats_df
    
    # Create tabs for different views
    tabs = st.tabs([
        "Summary", 
        "Scenario Comparison", 
        "Monthly Analysis",
        "Interest Analysis",
        "Distribution Analysis",
        "Raw Data"
    ])
    
    # Summary Tab
    with tabs[0]:
        st.markdown('<div class="section-header">Simulation Summary</div>', unsafe_allow_html=True)
        
        # Download button for CSV
        if st.session_state.csv_path:
            with open(st.session_state.csv_path, 'rb') as f:
                st.download_button(
                    label="Download Complete Simulation Results (CSV)",
                    data=f,
                    file_name=os.path.basename(st.session_state.csv_path),
                    mime="text/csv"
                )
        elif st.session_state.simulation_results_df is not None:
            buffer = io.StringIO()
            results_df.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Download Complete Simulation Results (CSV)",
                data=buffer.getvalue(),
                file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Display key metrics in a grid
        col1, col2, col3 = st.columns(3)
        
        # Calculate cumulative net profit for each scenario
        cumulative_profits = {}
        for scenario in monthly_df['scenario'].unique():
            scenario_data = monthly_df[monthly_df['scenario'] == scenario]
            cumulative_profits[scenario] = scenario_data['net_profit_mean'].sum()
        
        # Find best and worst scenarios
        best_scenario = max(cumulative_profits.items(), key=lambda x: x[1])
        worst_scenario = min(cumulative_profits.items(), key=lambda x: x[1])
        
        with col1:
            st.markdown('<div class="metric-label">Best Total Net Profit</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(best_scenario[1])}</div>', unsafe_allow_html=True)
            st.markdown(f'Scenario: {best_scenario[0]}', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-label">Worst Total Net Profit</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(worst_scenario[1])}</div>', unsafe_allow_html=True)
            st.markdown(f'Scenario: {worst_scenario[0]}', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-label">Profit Range</div>', unsafe_allow_html=True)
            profit_range = best_scenario[1] - worst_scenario[1]
            st.markdown(f'<div class="metric-value">{format_currency(profit_range)}</div>', unsafe_allow_html=True)
            st.markdown(f'Difference: {format_percentage(profit_range/abs(worst_scenario[1])) if worst_scenario[1] != 0 else "N/A"}', unsafe_allow_html=True)
        
        # Display summary charts
        st.markdown('<div class="subsection-header">Key Metrics by Scenario</div>', unsafe_allow_html=True)
        
        # Net Profit Box Plot
        net_profit_data = []
        for scenario in results_df[results_df['simulation_type'] == 'summary']['scenario'].unique():
            scenario_data = results_df[
                (results_df['simulation_type'] == 'summary') & 
                (results_df['scenario'] == scenario)
            ]
            net_profit_data.extend([{
                'Scenario': scenario,
                'Net Profit': row['net_profit'],
                'Total Income': row['total_income'],
                'Total Expenses': row['total_expenses'],
                'Total Losses': row['total_losses']
            } for _, row in scenario_data.iterrows()])
        
        net_profit_df = pd.DataFrame(net_profit_data)
        
        # Create box plots for key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_box_plot(net_profit_df, 'Scenario', 'Net Profit', 'Net Profit Distribution by Scenario')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_box_plot(net_profit_df, 'Scenario', 'Total Income', 'Total Income Distribution by Scenario')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_box_plot(net_profit_df, 'Scenario', 'Total Expenses', 'Total Expenses Distribution by Scenario')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_box_plot(net_profit_df, 'Scenario', 'Total Losses', 'Total Losses Distribution by Scenario')
            st.plotly_chart(fig, use_container_width=True)
        
        # Income Composition Over Time
        st.markdown('<div class="subsection-header">Income Composition Over Time</div>', unsafe_allow_html=True)
        
        # Create income composition plot
        income_data = []
        for scenario in monthly_df['scenario'].unique():
            scenario_data = monthly_df[monthly_df['scenario'] == scenario].sort_values('month')
            for _, row in scenario_data.iterrows():
                income_data.append({
                    'Month': row['month'],
                    'Scenario': scenario,
                    'Commission Income': row['commission_income_mean'],
                    'Interest Income': row['interest_income_mean']
                })
        
        income_df = pd.DataFrame(income_data)
        
        for scenario in income_df['Scenario'].unique():
            scenario_data = income_df[income_df['Scenario'] == scenario].set_index('Month')
            fig = create_stacked_area_plot(
                scenario_data[['Commission Income', 'Interest Income']], 
                f'Income Composition - {scenario}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Client Evolution
        st.markdown('<div class="subsection-header">Client Evolution</div>', unsafe_allow_html=True)
        
        fig = create_line_plot_with_range(monthly_df, 'Active Clients Over Time', 'active_clients')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = create_line_plot_with_range(monthly_df, 'Delinquent Clients Over Time', 'delinquent_clients')
        st.plotly_chart(fig, use_container_width=True)
    
    # Scenario Comparison Tab
    with tabs[1]:
        st.markdown('<div class="section-header">Scenario Comparison</div>', unsafe_allow_html=True)
        
        # Display summary table
        st.markdown('<div class="subsection-header">Summary Metrics</div>', unsafe_allow_html=True)
        
        # Create a formatted dataframe for display
        display_df = summary_df.copy()
        display_df['Net Profit Mean'] = display_df['net_profit_mean'].apply(format_currency)
        display_df['Net Profit Min'] = display_df['net_profit_min'].apply(format_currency)
        display_df['Net Profit Max'] = display_df['net_profit_max'].apply(format_currency)
        display_df['Delinquency Rate'] = display_df['delinquency_rate_mean'].apply(format_percentage)
        display_df['Active Clients'] = display_df['active_clients_final_mean'].apply(lambda x: f"{x:,.0f}")
        display_df['Initial Clients'] = display_df['num_clients_initial'].apply(lambda x: f"{x:,.0f}")
        
        # Reorder and rename columns for display
        display_cols = {
            'scenario': 'Scenario',
            'Net Profit Mean': 'Net Profit (Mean)',
            'Net Profit Min': 'Net Profit (Min)',
            'Net Profit Max': 'Net Profit (Max)',
            'Delinquency Rate': 'Delinquency Rate',
            'Active Clients': 'Active Clients (Final)',
            'Initial Clients': 'Initial Clients'
        }
        
        display_df = display_df[list(display_cols.keys())].rename(columns=display_cols)
        st.dataframe(display_df, use_container_width=True)
        
        # Monthly metrics comparison
        st.markdown('<div class="subsection-header">Monthly Metrics Comparison</div>', unsafe_allow_html=True)
        
        # Add scenario selection
        available_scenarios = monthly_df['scenario'].unique()
        selected_scenarios = st.multiselect(
            "Select Scenarios to Display",
            options=available_scenarios,
            default=available_scenarios,
            help="Choose which scenarios to display in the plots below"
        )
        
        # If no scenarios selected, show all
        if not selected_scenarios:
            selected_scenarios = available_scenarios
        
        metric_to_plot = st.selectbox(
            "Select Metric to Plot",
            options=[
                'Net Profit', 
                'Total Income', 
                'Commission Income', 
                'Interest Income', 
                'Active Clients', 
                'Delinquent Clients'
            ],
            index=0
        )
        
        # Map display names to column names
        metric_col_map = {
            'Net Profit': 'net_profit',
            'Total Income': 'total_income',
            'Commission Income': 'commission_income',
            'Interest Income': 'interest_income',
            'Active Clients': 'active_clients',
            'Delinquent Clients': 'delinquent_clients'
        }
        
        metric_col = metric_col_map[metric_to_plot]
        
        # Filter monthly_df for selected scenarios
        filtered_monthly_df = monthly_df[monthly_df['scenario'].isin(selected_scenarios)]
        
        # Create line plot with confidence intervals
        fig = create_line_plot_with_range(filtered_monthly_df, f'{metric_to_plot} Over Time by Scenario', metric_col)
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot for selected metric
        filtered_results_df = results_df[
            (results_df['simulation_type'] == 'monthly') & 
            (results_df['scenario'].isin(selected_scenarios))
        ]
        fig = create_box_plot(
            filtered_results_df,
            'scenario',
            metric_col,
            f'{metric_to_plot} Distribution by Scenario'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative metrics
        st.markdown('<div class="subsection-header">Cumulative Metrics</div>', unsafe_allow_html=True)
        
        cum_metric_to_plot = st.selectbox(
            "Select Cumulative Metric to Plot",
            options=[
                'Cumulative Net Profit', 
                'Cumulative Total Income', 
                'Cumulative Expenses', 
                'Cumulative Losses'
            ],
            index=0
        )
        
        # Map display names to column names
        cum_metric_col_map = {
            'Cumulative Net Profit': 'net_profit',
            'Cumulative Total Income': 'total_income',
            'Cumulative Expenses': 'expenses',
            'Cumulative Losses': 'losses'
        }
        
        cum_metric_col = cum_metric_col_map[cum_metric_to_plot]
        
        # Create cumulative plot
        cum_data = []
        for scenario in monthly_df['scenario'].unique():
            scenario_data = monthly_df[monthly_df['scenario'] == scenario].sort_values('month')
            cumulative_values = scenario_data[f'{cum_metric_col}_mean'].cumsum()
            cumulative_std = np.sqrt((scenario_data[f'{cum_metric_col}_std'] ** 2).cumsum())
            
            for i, (_, row) in enumerate(scenario_data.iterrows()):
                cum_data.append({
                    'Month': row['month'],
                    'Scenario': scenario,
                    'Value': cumulative_values.iloc[i],
                    'Std': cumulative_std.iloc[i]
                })
        
        cum_df = pd.DataFrame(cum_data)
        
        fig = go.Figure()
        
        for scenario in cum_df['Scenario'].unique():
            scenario_data = cum_df[cum_df['Scenario'] == scenario].sort_values('Month')
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=scenario_data['Month'],
                y=scenario_data['Value'],
                name=f"{scenario}",
                mode='lines',
                line=dict(width=2)
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=scenario_data['Month'],
                y=scenario_data['Value'] + 2*scenario_data['Std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name=f"{scenario} Upper"
            ))
            fig.add_trace(go.Scatter(
                x=scenario_data['Month'],
                y=scenario_data['Value'] - 2*scenario_data['Std'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(0,0,0,0.1)',
                fill='tonexty',
                showlegend=False,
                name=f"{scenario} Lower"
            ))
        
        fig.update_layout(
            title=f"{cum_metric_to_plot} Over Time by Scenario",
            title_x=0.5,
            title_font_size=20,
            height=500,
            hovermode='x unified',
            margin=dict(t=50, b=50),
            xaxis_title="Month",
            yaxis_title=cum_metric_to_plot.replace('Cumulative ', '')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Analysis Tab
    with tabs[2]:
        st.markdown('<div class="section-header">Monthly Analysis</div>', unsafe_allow_html=True)
        
        # Select scenario
        selected_scenario = st.selectbox(
            "Select Scenario",
            options=monthly_df['scenario'].unique()
        )
        
        # Filter data for selected scenario
        scenario_monthly = monthly_df[monthly_df['scenario'] == selected_scenario].sort_values('month')
        
        # Display monthly income breakdown
        st.markdown('<div class="subsection-header">Monthly Income Breakdown</div>', unsafe_allow_html=True)
        
        # Create dataframe for area chart
        income_data = []
        for _, row in scenario_monthly.iterrows():
            income_data.append({
                'Month': row['month'],
                'Commission Income': row['commission_income_mean'],
                'Interest Income': row['interest_income_mean']
            })
        
        income_df = pd.DataFrame(income_data).set_index('Month')
        
        fig = create_stacked_area_plot(
            income_df, 
            f'Monthly Income Breakdown - {selected_scenario}'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display interest metrics
        st.markdown('<div class="subsection-header">Interest Metrics</div>', unsafe_allow_html=True)
        
        # Create interest metrics plot
        interest_data = []
        for _, row in scenario_monthly.iterrows():
            interest_data.append({
                'Month': row['month'],
                'Interest Generated': row['interes_generado_total'],
                'Interest Paid': row['pagos_interes_total'],
                'Interest Balance': row['saldo_interes_total'],
                'Principal Balance': row['saldo_principal_total']
            })
        
        interest_df = pd.DataFrame(interest_data).set_index('Month')
        
        # Create subplots for interest metrics
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Interest Generation and Payments', 'Principal and Interest Balances'))
        
        # Add interest generation and payments
        fig.add_trace(go.Scatter(
            x=interest_df.index,
            y=interest_df['Interest Generated'],
            name='Interest Generated',
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=interest_df.index,
            y=interest_df['Interest Paid'],
            name='Interest Paid',
            line=dict(color='green')
        ), row=1, col=1)
        
        # Add balances
        fig.add_trace(go.Scatter(
            x=interest_df.index,
            y=interest_df['Principal Balance'],
            name='Principal Balance',
            line=dict(color='red')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=interest_df.index,
            y=interest_df['Interest Balance'],
            name='Interest Balance',
            line=dict(color='orange')
        ), row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Interest Metrics - {selected_scenario}",
            title_x=0.5,
            title_font_size=20
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display interest collection rate
        st.markdown('<div class="subsection-header">Interest Collection Rate</div>', unsafe_allow_html=True)
        
        # Calculate interest collection rate
        collection_data = []
        for _, row in scenario_monthly.iterrows():
            if row['interes_generado_total'] > 0:
                collection_rate = row['pagos_interes_total'] / row['interes_generado_total']
            else:
                collection_rate = 0
            collection_data.append({
                'Month': row['month'],
                'Collection Rate': collection_rate
            })
        
        collection_df = pd.DataFrame(collection_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=collection_df['Month'],
            y=collection_df['Collection Rate'],
            name='Interest Collection Rate',
            mode='lines+markers',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title=f'Monthly Interest Collection Rate - {selected_scenario}',
            title_x=0.5,
            title_font_size=20,
            height=500,
            hovermode='x unified',
            margin=dict(t=50, b=50),
            xaxis_title="Month",
            yaxis_title="Collection Rate",
            yaxis_tickformat=',.1%'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Interest Analysis Tab
    with tabs[3]:
        st.markdown('<div class="section-header">Interest Analysis</div>', unsafe_allow_html=True)
        
        # Select scenario
        selected_scenario = st.selectbox(
            "Select Scenario for Interest Analysis",
            options=monthly_df['scenario'].unique()
        )
        
        # Filter data for selected scenario
        scenario_monthly = monthly_df[monthly_df['scenario'] == selected_scenario].sort_values('month')
        
        # Calculate cumulative interest metrics
        cumulative_data = []
        cum_interest_generated = 0
        cum_interest_paid = 0
        
        for _, row in scenario_monthly.iterrows():
            cum_interest_generated += row['interes_generado_total']
            cum_interest_paid += row['pagos_interes_total']
            
            cumulative_data.append({
                'Month': row['month'],
                'Cumulative Interest Generated': cum_interest_generated,
                'Cumulative Interest Paid': cum_interest_paid,
                'Cumulative Principal Balance': row['saldo_principal_total'],
                'Cumulative Interest Balance': row['saldo_interes_total']
            })
        
        cumulative_df = pd.DataFrame(cumulative_data)
        
        # Display cumulative interest metrics
        st.markdown('<div class="subsection-header">Cumulative Interest Metrics</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Add cumulative interest generated
        fig.add_trace(go.Scatter(
            x=cumulative_df['Month'],
            y=cumulative_df['Cumulative Interest Generated'],
            name='Cumulative Interest Generated',
            line=dict(color='blue', width=2)
        ))
        
        # Add cumulative interest paid
        fig.add_trace(go.Scatter(
            x=cumulative_df['Month'],
            y=cumulative_df['Cumulative Interest Paid'],
            name='Cumulative Interest Paid',
            line=dict(color='green', width=2)
        ))
        
        # Add cumulative interest balance
        fig.add_trace(go.Scatter(
            x=cumulative_df['Month'],
            y=cumulative_df['Cumulative Interest Balance'],
            name='Cumulative Interest Balance',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title=f'Cumulative Interest Metrics - {selected_scenario}',
            title_x=0.5,
            title_font_size=20,
            height=500,
            hovermode='x unified',
            margin=dict(t=50, b=50),
            xaxis_title="Month",
            yaxis_title="Amount"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display interest efficiency metrics
        st.markdown('<div class="subsection-header">Interest Efficiency Metrics</div>', unsafe_allow_html=True)
        
        # Calculate interest efficiency metrics
        efficiency_data = []
        for _, row in scenario_monthly.iterrows():
            if row['saldo_principal_total'] > 0:
                interest_rate = (row['interes_generado_total'] / row['saldo_principal_total']) * 12 * 100
            else:
                interest_rate = 0
                
            if row['interes_generado_total'] > 0:
                collection_rate = row['pagos_interes_total'] / row['interes_generado_total']
            else:
                collection_rate = 0
                
            efficiency_data.append({
                'Month': row['month'],
                'Effective Interest Rate (%)': interest_rate,
                'Interest Collection Rate': collection_rate,
                'Interest Balance / Principal Balance': row['saldo_interes_total'] / row['saldo_principal_total'] if row['saldo_principal_total'] > 0 else 0
            })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        
        # Create subplots for efficiency metrics
        fig = make_subplots(rows=3, cols=1, 
                           subplot_titles=('Effective Interest Rate', 'Interest Collection Rate', 
                                         'Interest/Principal Balance Ratio'))
        
        # Add effective interest rate
        fig.add_trace(go.Scatter(
            x=efficiency_df['Month'],
            y=efficiency_df['Effective Interest Rate (%)'],
            name='Effective Interest Rate',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Add interest collection rate
        fig.add_trace(go.Scatter(
            x=efficiency_df['Month'],
            y=efficiency_df['Interest Collection Rate'],
            name='Interest Collection Rate',
            line=dict(color='green')
        ), row=2, col=1)
        
        # Add interest/principal ratio
        fig.add_trace(go.Scatter(
            x=efficiency_df['Month'],
            y=efficiency_df['Interest Balance / Principal Balance'],
            name='Interest/Principal Ratio',
            line=dict(color='orange')
        ), row=3, col=1)
        
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text=f"Interest Efficiency Metrics - {selected_scenario}",
            title_x=0.5,
            title_font_size=20
        )
        
        # Update y-axis formats
        fig.update_yaxes(title_text="Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Collection Rate", row=2, col=1)
        fig.update_yaxes(title_text="Ratio", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display interest metrics summary
        st.markdown('<div class="subsection-header">Interest Metrics Summary</div>', unsafe_allow_html=True)
        
        # Calculate summary metrics
        total_interest_generated = cumulative_df['Cumulative Interest Generated'].iloc[-1]
        total_interest_paid = cumulative_df['Cumulative Interest Paid'].iloc[-1]
        final_interest_balance = cumulative_df['Cumulative Interest Balance'].iloc[-1]
        final_principal_balance = cumulative_df['Cumulative Principal Balance'].iloc[-1]
        
        # Create summary metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Interest Generated",
                f"${total_interest_generated:,.2f}",
                f"${total_interest_generated/len(cumulative_df):,.2f}/month"
            )
        
        with col2:
            st.metric(
                "Total Interest Paid",
                f"${total_interest_paid:,.2f}",
                f"${total_interest_paid/len(cumulative_df):,.2f}/month"
            )
        
        with col3:
            st.metric(
                "Final Interest Balance",
                f"${final_interest_balance:,.2f}",
                f"{final_interest_balance/final_principal_balance:.1%} of principal"
            )
    
    # Distribution Analysis Tab
    with tabs[4]:
        st.markdown('<div class="section-header">Distribution Analysis</div>', unsafe_allow_html=True)
        
        # Display distribution plots
        st.markdown('<div class="subsection-header">Net Profit Distribution</div>', unsafe_allow_html=True)
        fig = create_distribution_plot(results_df, 'net_profit', 'Net Profit Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="subsection-header">Total Income Distribution</div>', unsafe_allow_html=True)
        fig = create_distribution_plot(results_df, 'total_income', 'Total Income Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="subsection-header">Total Expenses Distribution</div>', unsafe_allow_html=True)
        fig = create_distribution_plot(results_df, 'total_expenses', 'Total Expenses Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="subsection-header">Total Losses Distribution</div>', unsafe_allow_html=True)
        fig = create_distribution_plot(results_df, 'total_losses', 'Total Losses Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Raw Data Tab
    with tabs[5]:
        st.markdown('<div class="section-header">Raw Data</div>', unsafe_allow_html=True)
        
        # Select which data to view
        data_view = st.radio(
            "Select Data to View",
            options=["Summary Statistics", "Monthly Statistics", "Full Simulation Data"],
            horizontal=True
        )
        
        if data_view == "Summary Statistics":
            st.markdown('<div class="subsection-header">Summary Statistics</div>', unsafe_allow_html=True)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary Statistics as CSV",
                data=csv,
                file_name="credit_card_simulation_summary_stats.csv",
                mime="text/csv"
            )
            
        elif data_view == "Monthly Statistics":
            st.markdown('<div class="subsection-header">Monthly Statistics</div>', unsafe_allow_html=True)
            
            # Filter by scenario
            scenario_filter = st.selectbox(
                "Filter by Scenario",
                options=["All Scenarios"] + list(monthly_df['scenario'].unique())
            )
            
            if scenario_filter == "All Scenarios":
                filtered_df = monthly_df
            else:
                filtered_df = monthly_df[monthly_df['scenario'] == scenario_filter]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Monthly Statistics as CSV",
                data=csv,
                file_name=f"credit_card_simulation_monthly_stats_{scenario_filter.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
        elif data_view == "Full Simulation Data":
            st.markdown('<div class="subsection-header">Full Simulation Data</div>', unsafe_allow_html=True)
            
            # Let user see a sample of the data
            st.write("Showing a sample of the full simulation data")
            st.dataframe(results_df.sample(min(1000, len(results_df))), use_container_width=True)
            
            st.write(f"Full dataset has {len(results_df)} rows and is available for download at the top of the Summary tab.")
            
            # Allow filtering
            show_full_data = st.checkbox("Show filtered full data (may be slow for large datasets)")
            
            if show_full_data:
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    scenario_filter = st.selectbox(
                        "Filter by Scenario (Full Data)",
                        options=["All Scenarios"] + list(results_df['scenario'].unique())
                    )
                
                with col2:
                    sim_type_filter = st.selectbox(
                        "Filter by Simulation Type",
                        options=["All Types", "summary", "monthly"]
                    )
                
                # Apply filters
                filtered_raw_df = results_df.copy()
                
                if scenario_filter != "All Scenarios":
                    filtered_raw_df = filtered_raw_df[filtered_raw_df['scenario'] == scenario_filter]
                
                if sim_type_filter != "All Types":
                    filtered_raw_df = filtered_raw_df[filtered_raw_df['simulation_type'] == sim_type_filter]
                
                st.dataframe(filtered_raw_df, use_container_width=True)
                
                # Download filtered data
                csv = filtered_raw_df.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name=f"credit_card_simulation_filtered_data.csv",
                    mime="text/csv"
                )
else:
    # Display instructions if no simulation has been run
    st.info("Set your parameters in the sidebar and click 'Run Simulation' to start, or upload an existing simulation results file.")
    
    # Display sample images or explanations
    st.markdown('<div class="section-header">How to Use This Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Set Parameters**: Adjust the simulation parameters in the sidebar. You can set optimistic, neutral, and pessimistic values for each parameter.
    
    2. **Run Simulation**: Click the "Run Simulation" button to start the simulation. This may take a few minutes depending on the number of clients and years.
    
    3. **Or Load Existing Results**: Upload a previously saved simulation results CSV file.
    
    4. **Explore Results**: Once the simulation is complete or results are loaded, you can explore the results in different tabs:
       - **Summary**: Overview of key metrics across all scenarios
       - **Scenario Comparison**: Compare different scenarios side by side
       - **Monthly Analysis**: Analyze monthly trends for a specific scenario
       - **Raw Data**: Access the raw simulation data
    
    5. **Save Results**: All simulation results are saved as a single CSV file that you can download from the Summary tab.
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
