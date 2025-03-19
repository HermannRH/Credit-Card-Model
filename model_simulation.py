import numpy as np
import pandas as pd
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import os
import json
from datetime import datetime
import logging
import time
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)

# Constants for resource limits
MAX_MEMORY_PERCENT = 80  # Maximum memory usage percentage
BATCH_SIZE = 5  # Number of simulations to process before saving
SAVE_INTERVAL = 300  # Save results every 5 minutes

@dataclass
class CreditCardParams:
    """Parameters for credit card portfolio simulation with optimistic, neutral, and pessimistic values."""
    
    # Portfolio size
    num_clientes: Tuple[int, int, int]  # (optimistic, neutral, pessimistic)
    
    # Customer behavior
    perc_totaleros: Tuple[float, float, float]  # % of customers who pay in full
    perc_morosidad: Tuple[float, float, float]  # % chance of delinquency per month
    
    # Credit line parameters
    linea_credito_prom: Tuple[float, float, float]  # Average credit line
    
    # Utilization parameters
    util_credito_totaleros: Tuple[float, float, float]  # % utilization for full payers
    util_credito_revolventes_alpha: Tuple[float, float, float]  # Alpha param for beta distribution
    util_credito_revolventes_beta: Tuple[float, float, float]  # Beta param for beta distribution
    
    # Payment parameters
    pago_minimo_perc: Tuple[float, float, float]  # Minimum payment as % of balance
    prob_pago_minimo: Tuple[float, float, float]  # Probability of paying only minimum
    
    # Financial parameters
    tasa_interes: Tuple[float, float, float]  # Annual interest rate (%)
    comision_venta: Tuple[float, float, float]  # Interchange fee (%)
    costo_emision: Tuple[float, float, float]  # Cost per card (includes all operational costs)
    
    # Simulation parameters
    semilla_aleatoria: int  # Random seed for reproducibility
    start_up_time: int  # Number of months to reach full customer base

    def get_scenario_params(self, scenario: str) -> Dict[str, Any]:
        """Get parameter values for a specific scenario."""
        idx_map = {
            'optimistic': 0,
            'neutral': 1,
            'pessimistic': 2,
            'opt_neut': 0.5,
            'pes_neut': 1.5,
        }
        
        idx = idx_map[scenario]
        
        if isinstance(idx, float):
            base_idx = int(idx)
            interp_factor = idx - base_idx
            
            result = {}
            for key, value in self.__dict__.items():
                if key == 'semilla_aleatoria':
                    result[key] = value
                    continue
                    
                if isinstance(value, tuple) and len(value) == 3:
                    val1 = value[base_idx]
                    val2 = value[base_idx + 1]
                    interpolated_value = val1 + interp_factor * (val2 - val1)
                    if key == 'num_clientes':
                        result[key] = int(round(interpolated_value))
                    else:
                        result[key] = interpolated_value
                else:
                    result[key] = value
            
            return result
        
        result = {}
        for key, value in self.__dict__.items():
            if key == 'semilla_aleatoria':
                result[key] = value
                continue
                
            if isinstance(value, tuple) and len(value) == 3:
                result[key] = value[idx]
            else:
                result[key] = value
        
        return result

def generar_escenarios(params: CreditCardParams) -> List[Tuple[str, Dict[str, Any]]]:
    """Generate all possible scenarios based on parameters."""
    escenarios = ['optimistic', 'opt_neut', 'neutral', 'pes_neut', 'pessimistic']
    return [(escenario, params.get_scenario_params(escenario)) for escenario in escenarios]

def simular_escenario(nombre_escenario: str, params: Dict[str, Any], num_years: int) -> Optional[Dict[str, Any]]:
    """
    Simulate a specific scenario with error handling.
    
    Args:
        nombre_escenario: Name of the scenario
        params: Scenario parameters
        num_years: Number of years to simulate
        
    Returns:
        Simulation results or None if simulation failed
    """
    try:
        logging.info(f"Starting simulation for scenario: {nombre_escenario}")
        resultados = simular_cartera(params, num_years)
        logging.info(f"Successfully completed simulation for scenario: {nombre_escenario}")
        return resultados
    except Exception as e:
        logging.error(f"Error in simulation for scenario {nombre_escenario}: {str(e)}")
        return None

def save_batch_results(batch_results: List[Dict[str, Any]], directorio: str, batch_number: int) -> None:
    """Save a batch of results to CSV."""
    try:
        if not os.path.exists(directorio):
            os.makedirs(directorio)
            
        # Process batch results
        processed_results = []
        for result in batch_results:
            if result is None:
                continue
                
            try:
                # Add summary metrics
                summary_row = {
                    'scenario': result['scenario'],
                    'seed': result['seed'],
                    'total_income': result['ingresos_totales'],
                    'total_expenses': result['gastos_totales'],
                    'total_losses': result['perdidas_totales'],
                    'net_profit': result['ganancia_neta'],
                    'delinquency_rate': result['tasa_morosidad'],
                    'active_clients_final': result['clientes_activos_final'],
                    'num_clients_initial': result['parametros']['num_clientes'],
                    'simulation_type': 'summary'
                }
                processed_results.append(summary_row)
                
                # Add monthly results
                for month_data in result['resultados_mes']:
                    month_row = {
                        'scenario': result['scenario'],
                        'seed': result['seed'],
                        'month': month_data['mes_global'],
                        'year': month_data['anio'],
                        'month_of_year': month_data['mes'],
                        'commission_income': month_data['ingresos_comisiones'],
                        'interest_income': month_data['ingresos_intereses'],
                        'total_income': month_data['ingresos'],
                        'expenses': month_data['gastos'],
                        'losses': month_data['perdidas'],
                        'net_profit': month_data['ingresos'] - month_data['gastos'] - month_data['perdidas'],
                        'active_clients': month_data['clientes_activos'],
                        'delinquent_clients': month_data['clientes_morosos'],
                        'interes_generado_total': month_data['interes_generado_total'],
                        'pagos_interes_total': month_data['pagos_interes_total'],
                        'saldo_principal_total': month_data['saldo_principal_total'],
                        'saldo_interes_total': month_data['saldo_interes_total'],
                        'simulation_type': 'monthly'
                    }
                    processed_results.append(month_row)
                    
            except Exception as e:
                logging.error(f"Error processing result: {str(e)}")
                continue
        
        if processed_results:
            # Save batch to CSV
            batch_df = pd.DataFrame(processed_results)
            batch_file = f"{directorio}/simulation_results_batch_{batch_number}.csv"
            batch_df.to_csv(batch_file, index=False)
            logging.info(f"Saved batch {batch_number} to {batch_file}")
            
    except Exception as e:
        logging.error(f"Error saving batch {batch_number}: {str(e)}")

def simular_escenarios_secuencial(params: CreditCardParams, num_years: int = 1, 
                                num_seeds: int = 1000, directorio: str = "resultados_simulacion") -> str:
    """
    Simulate all scenarios sequentially with better error handling and logging.
    
    Args:
        params: CreditCardParams object with optimistic, neutral, and pessimistic values
        num_years: Number of years to simulate
        num_seeds: Number of random seeds to simulate
        directorio: Directory to save results
        
    Returns:
        Path to the final results file
    """
    try:
        logging.info("Starting sequential simulation")
        escenarios = generar_escenarios(params)
        all_args = []
        
        # Generate all simulation arguments
        for nombre, params_escenario in escenarios:
            for seed in range(num_seeds):
                params_with_seed = params_escenario.copy()
                params_with_seed['semilla_aleatoria'] = seed
                all_args.append((nombre, params_with_seed, num_years))
        
        total_simulations = len(all_args)
        logging.info(f"Total simulations to run: {total_simulations}")
        
        # Process simulations in batches
        batch_results = []
        batch_number = 1
        last_save_time = time.time()
        
        with tqdm(total=total_simulations, desc="Running simulations") as pbar:
            for i, (nombre, params, years) in enumerate(all_args):
                try:
                    # Run simulation
                    result = simular_escenario(nombre, params, years)
                    if result is not None:
                        result['scenario'] = nombre
                        result['seed'] = params['semilla_aleatoria']
                        batch_results.append(result)
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Save batch if needed
                    current_time = time.time()
                    if (len(batch_results) >= BATCH_SIZE or 
                        current_time - last_save_time >= SAVE_INTERVAL):
                        save_batch_results(batch_results, directorio, batch_number)
                        batch_results = []
                        batch_number += 1
                        last_save_time = current_time
                        
                        # Force garbage collection
                        gc.collect()
                        
                except Exception as e:
                    logging.error(f"Error in simulation {i}: {str(e)}")
                    continue
        
        # Save any remaining results
        if batch_results:
            save_batch_results(batch_results, directorio, batch_number)
        
        # Combine all batch files
        final_file = f"{directorio}/simulation_results.csv"
        batch_files = sorted([f for f in os.listdir(directorio) if f.startswith("simulation_results_batch_")])
        
        if batch_files:
            # Read and combine all batch files
            dfs = []
            for batch_file in batch_files:
                try:
                    df = pd.read_csv(os.path.join(directorio, batch_file))
                    dfs.append(df)
                except Exception as e:
                    logging.error(f"Error reading batch file {batch_file}: {str(e)}")
                    continue
            
            if dfs:
                final_df = pd.concat(dfs, ignore_index=True)
                final_df.to_csv(final_file, index=False)
                logging.info(f"Successfully combined all results into {final_file}")
                
                # Clean up batch files
                for batch_file in batch_files:
                    try:
                        os.remove(os.path.join(directorio, batch_file))
                    except Exception as e:
                        logging.error(f"Error removing batch file {batch_file}: {str(e)}")
        
        logging.info("Simulation completed successfully")
        return final_file
        
    except Exception as e:
        logging.error(f"Fatal error in simulation: {str(e)}")
        raise

def simular_cartera(params: Dict[str, Any], num_years: int = 1) -> Dict[str, Any]:
    """
    Simula el comportamiento de una cartera de tarjetas durante el período especificado
    
    Args:
        params: Diccionario con todos los parámetros necesarios para la simulación
        num_years: Número de años a simular
    
    Returns:
        Diccionario con los resultados de la simulación
    """
    random.seed(params['semilla_aleatoria'])
    np.random.seed(params['semilla_aleatoria'])
    
    # Definir factores de estacionalidad
    seasonality_factors = [
        0.9, 0.9, 0.9,      # Months 1-3: 90%
        1.0, 1.0, 1.0,      # Months 4-6: 100%
        1.1,                # Month 7: 110%
        1.0, 1.0, 1.0, 1.0, # Months 8-11: 100%
        1.1                 # Month 12: 110%
    ]
    
    # Inicializar resultados
    resultados_mes = []
    resultados_clientes = []
    ingresos_totales = 0
    gastos_totales = params['num_clientes'] * params['costo_emision'] * 12 * num_years
    perdidas_totales = 0
    
    # Crear lista de clientes (todos los clientes potenciales)
    clientes_potenciales = []
    for i in range(params['num_clientes']):
        es_totalero = random.random() < params['perc_totaleros']
        linea_credito = params['linea_credito_prom']
        cliente = {
            'id': i + 1,
            'es_totalero': es_totalero,
            'es_moroso': False,
            'linea_credito': linea_credito,
            'saldo_principal': 0,  # Principal balance
            'saldo_interes': 0,    # Interest balance
            'linea_disponible': linea_credito
        }
        clientes_potenciales.append(cliente)
    
    # Simulación mes a mes
    for year in range(num_years):
        for month in range(12):
            mes_global = year * 12 + month + 1
            mes_del_anio = month + 1
            factor_estacional = seasonality_factors[month]
            
            # Determinar cuántos clientes activos hay en este mes
            if mes_global <= params['start_up_time']:
                # Durante el período de inicio, agregar clientes gradualmente
                # Calcular cuántos clientes deberían estar activos en este mes
                clientes_por_mes = params['num_clientes'] / params['start_up_time']
                num_clientes_activos = int(round(clientes_por_mes * mes_global))
            else:
                num_clientes_activos = params['num_clientes']
            
            # Seleccionar los clientes activos para este mes
            clientes = clientes_potenciales[:num_clientes_activos]
            
            ingresos_mes = 0
            gastos_mes = num_clientes_activos * params['costo_emision'] / 12
            perdidas_mes = 0
            
            detalles_mes = {
                'mes_global': mes_global,
                'anio': year + 1,
                'mes': mes_del_anio,
                'clientes': [],
                'ingresos_comisiones': 0,
                'ingresos_intereses': 0,
                'gastos': gastos_mes,
                'perdidas': 0,
                'clientes_activos': 0,
                'clientes_morosos': 0,
                'clientes_totales': num_clientes_activos  # Agregar número total de clientes en este mes
            }
            
            # Simular cada cliente
            for cliente in clientes:
                # Si el cliente ya es moroso, saltamos al siguiente
                if cliente['es_moroso']:
                    detalles_mes['clientes_morosos'] += 1
                    continue
                
                detalles_mes['clientes_activos'] += 1
                
                # Variables para el cliente en este mes
                es_totalero = cliente['es_totalero']
                saldo_principal_anterior = cliente['saldo_principal']
                saldo_interes_anterior = cliente['saldo_interes']
                linea_disponible = cliente['linea_disponible']
                
                # Determinar consumo del mes (ajustado por estacionalidad)
                if es_totalero:
                    consumo_base = cliente['linea_credito'] * params['util_credito_totaleros']
                    consumo = consumo_base * factor_estacional
                    linea_disponible = cliente['linea_credito'] - consumo
                else:
                    # Usar distribución beta para modelar utilización de crédito
                    alpha = params['util_credito_revolventes_alpha']
                    beta = params['util_credito_revolventes_beta']
                    porcentaje_utilizacion = np.random.beta(alpha, beta)
                    consumo_base = linea_disponible * porcentaje_utilizacion
                    consumo = consumo_base * factor_estacional
                
                # Aplicar comisión por venta (interchange fee)
                ingreso_comision = consumo * params['comision_venta']
                ingresos_mes += ingreso_comision
                detalles_mes['ingresos_comisiones'] += ingreso_comision
                
                # Calcular nuevo saldo principal después del consumo
                nuevo_saldo_principal = saldo_principal_anterior + consumo
                
                # Verificar si el cliente se vuelve moroso este mes
                es_moroso_nuevo = random.random() < params['perc_morosidad']
                
                # Determinar pago del cliente
                if es_moroso_nuevo:
                    # Cliente se vuelve moroso, no paga nada
                    cliente['es_moroso'] = True
                    pago_principal = 0
                    pago_interes = 0
                    # Registrar pérdida del principal
                    perdida = nuevo_saldo_principal
                    perdidas_mes += perdida
                    
                    # No hay intereses para clientes morosos
                    interes_mensual = 0
                    saldo_principal_final = 0  # El saldo se da por perdido
                    saldo_interes_final = 0    # El interés se da por perdido
                    linea_disponible = 0  # No hay línea disponible para morosos
                    
                elif es_totalero:
                    # Totalero paga todo el consumo del mes
                    pago_principal = consumo
                    pago_interes = 0
                    saldo_principal_final = saldo_principal_anterior
                    saldo_interes_final = 0
                    interes_mensual = 0
                    linea_disponible = cliente['linea_credito'] - saldo_principal_final
                    
                else:
                    # Cliente revolvente
                    saldo_total = nuevo_saldo_principal + saldo_interes_anterior
                    pago_minimo = saldo_total * params['pago_minimo_perc']
                    
                    # Determinar si paga el mínimo o más
                    if random.random() < params['prob_pago_minimo']:
                        # Paga solo el mínimo
                        pago_total = pago_minimo
                    else:
                        # Paga un monto aleatorio entre el mínimo y el total
                        pago_total = random.uniform(pago_minimo, saldo_total)
                    
                    # Asegurar que el pago no exceda el saldo total
                    pago_total = min(pago_total, saldo_total)
                    
                    # Distribuir el pago entre interés y principal
                    # 1. Primero se paga todo el interés pendiente
                    pago_interes = min(pago_total, saldo_interes_anterior)
                    # 2. El resto del pago se aplica al principal
                    pago_principal = pago_total - pago_interes
                    
                    # Calcular saldo después del pago
                    saldo_principal_despues_pago = nuevo_saldo_principal - pago_principal
                    saldo_interes_despues_pago = saldo_interes_anterior - pago_interes
                    
                    # Calcular nuevo interés solo sobre el principal
                    interes_mensual = saldo_principal_despues_pago * (params['tasa_interes'] / 100 / 12)
                    
                    # Actualizar saldos finales
                    saldo_principal_final = saldo_principal_despues_pago
                    saldo_interes_final = saldo_interes_despues_pago + interes_mensual
                    
                    linea_disponible = cliente['linea_credito'] - (saldo_principal_final + saldo_interes_final)
                    
                    # Solo contar como ingreso el interés que se pagó
                    ingresos_mes += pago_interes
                    detalles_mes['ingresos_intereses'] += pago_interes
                
                # Actualizar cliente
                cliente['saldo_principal'] = saldo_principal_final
                cliente['saldo_interes'] = saldo_interes_final
                cliente['linea_disponible'] = max(0, linea_disponible)
                
                # Guardar detalles del cliente en este mes
                detalle_cliente = {
                    'id': cliente['id'],
                    'es_totalero': es_totalero,
                    'consumo': consumo,
                    'saldo_principal_anterior': saldo_principal_anterior,
                    'saldo_interes_anterior': saldo_interes_anterior,
                    'pago_principal': pago_principal,
                    'pago_interes': pago_interes,
                    'saldo_principal_final': cliente['saldo_principal'],
                    'saldo_interes_final': cliente['saldo_interes'],
                    'linea_disponible': cliente['linea_disponible'],
                    'ingreso_comision': ingreso_comision,
                    'ingreso_intereses': pago_interes,
                    'es_moroso': cliente['es_moroso'],
                    'perdida': perdida if es_moroso_nuevo else 0,
                    'interes_generado': interes_mensual if not es_moroso_nuevo else 0,
                    'saldo_total': cliente['saldo_principal'] + cliente['saldo_interes'],
                    'pago_total': pago_principal + pago_interes
                }
                detalles_mes['clientes'].append(detalle_cliente)
            
            # Actualizar totales
            detalles_mes['ingresos'] = detalles_mes['ingresos_comisiones'] + detalles_mes['ingresos_intereses']
            detalles_mes['perdidas'] = perdidas_mes
            detalles_mes['saldo_principal_total'] = sum(c['saldo_principal'] for c in clientes)
            detalles_mes['saldo_interes_total'] = sum(c['saldo_interes'] for c in clientes)
            detalles_mes['pagos_principal_total'] = sum(d['pago_principal'] for d in detalles_mes['clientes'])
            detalles_mes['pagos_interes_total'] = sum(d['pago_interes'] for d in detalles_mes['clientes'])
            detalles_mes['interes_generado_total'] = sum(d['interes_generado'] for d in detalles_mes['clientes'])
            
            ingresos_totales += detalles_mes['ingresos']
            perdidas_totales += perdidas_mes
            
            # Guardar resultados del mes
            resultados_mes.append(detalles_mes)
            
            # Guardar resultados por cliente para el último mes
            if mes_global == num_years * 12:
                for detalle in detalles_mes['clientes']:
                    resultados_clientes.append(detalle)
    
    # Calcular ganancia neta
    ganancia_neta = ingresos_totales - gastos_totales - perdidas_totales
    
    # Calcular métricas de la cartera
    total_clientes = params['num_clientes']
    clientes_activos_final = sum(1 for c in clientes if not c['es_moroso'])
    tasa_morosidad = (total_clientes - clientes_activos_final) / total_clientes
    
    return {
        'resultados_mes': resultados_mes,
        'resultados_clientes': resultados_clientes,
        'ingresos_totales': ingresos_totales,
        'gastos_totales': gastos_totales,
        'perdidas_totales': perdidas_totales,
        'ganancia_neta': ganancia_neta,
        'tasa_morosidad': tasa_morosidad,
        'clientes_activos_final': clientes_activos_final,
        'parametros': params
    }
