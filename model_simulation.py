import numpy as np
import pandas as pd
import random
from tqdm.auto import tqdm
import multiprocessing as mp
from itertools import product
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import os
import json
from datetime import datetime
import concurrent.futures


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
    
    def get_scenario_params(self, scenario: str) -> Dict[str, Any]:
        """
        Get parameter values for a specific scenario.
        
        Args:
            scenario: One of 'optimistic', 'neutral', 'pessimistic', 'opt_neut', or 'pes_neut'
        
        Returns:
            Dictionary of parameter values for the specified scenario
        """
        idx_map = {
            'optimistic': 0,
            'neutral': 1,
            'pessimistic': 2,
            'opt_neut': 0.5,  # Will interpolate between optimistic and neutral
            'pes_neut': 1.5,  # Will interpolate between neutral and pessimistic
        }
        
        idx = idx_map[scenario]
        
        # For interpolated scenarios
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
                    # Ensure num_clientes is always an integer
                    if key == 'num_clientes':
                        result[key] = int(round(interpolated_value))
                    else:
                        result[key] = interpolated_value
                else:
                    result[key] = value
            
            return result
        
        # For direct scenarios
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
    
    # Crear clientes
    clientes = []
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
        clientes.append(cliente)
    
    # Simulación mes a mes
    for year in range(num_years):
        for month in range(12):
            mes_global = year * 12 + month + 1
            mes_del_anio = month + 1
            factor_estacional = seasonality_factors[month]
            
            ingresos_mes = 0
            gastos_mes = params['num_clientes'] * params['costo_emision'] / 12
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
                'clientes_morosos': 0
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


def generar_escenarios(params: CreditCardParams) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Genera todos los escenarios posibles basados en los parámetros.
    
    Args:
        params: Objeto CreditCardParams con los valores optimistas, neutrales y pesimistas
        
    Returns:
        Lista de tuplas (nombre_escenario, parámetros)
    """
    escenarios = ['optimistic', 'opt_neut', 'neutral', 'pes_neut', 'pessimistic']
    return [(escenario, params.get_scenario_params(escenario)) for escenario in escenarios]


def simular_escenario(args: Tuple[str, Dict[str, Any], int]) -> Tuple[str, Dict[str, Any]]:
    """
    Simula un escenario específico.
    
    Args:
        args: Tupla con (nombre_escenario, parámetros, num_years)
        
    Returns:
        Tupla con (nombre_escenario, resultados)
    """
    nombre_escenario, params, num_years = args
    resultados = simular_cartera(params, num_years)
    return (nombre_escenario, resultados)


def simular_escenarios_paralelo(params: CreditCardParams, num_years: int = 1, 
                               num_procesos: Optional[int] = None,
                               num_seeds: int = 1000) -> Dict[str, List[Dict[str, Any]]]:
    """
    Simula todos los escenarios en paralelo con múltiples semillas.
    
    Args:
        params: Objeto CreditCardParams con los valores optimistas, neutrales y pesimistas
        num_years: Número de años a simular
        num_procesos: Número de procesos a utilizar (None = usar todos los disponibles)
        num_seeds: Número de semillas aleatorias a simular
        
    Returns:
        Diccionario con los resultados de cada escenario y semilla
    """
    escenarios = generar_escenarios(params)
    all_args = []
    
    # Generate args for all scenarios and seeds
    for nombre, params_escenario in escenarios:
        for seed in range(num_seeds):
            params_with_seed = params_escenario.copy()
            params_with_seed['semilla_aleatoria'] = seed
            all_args.append((nombre, params_with_seed, num_years))
    
    if num_procesos is None:
        num_procesos = mp.cpu_count()
    
    with mp.Pool(processes=num_procesos) as pool:
        resultados = list(tqdm(pool.imap(simular_escenario, all_args), 
                              total=len(all_args), 
                              desc="Simulando escenarios"))
    
    # Reorganize results by scenario
    resultados_por_escenario = {}
    for nombre, resultado in resultados:
        if nombre not in resultados_por_escenario:
            resultados_por_escenario[nombre] = []
        resultados_por_escenario[nombre].append(resultado)
    
    return resultados_por_escenario


def calcular_estadisticas_escenario(resultados_escenario: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula estadísticas para un escenario basado en múltiples simulaciones.
    
    Args:
        resultados_escenario: Lista de resultados para un escenario
        
    Returns:
        Diccionario con estadísticas del escenario
    """
    # Extract key metrics
    metricas = {
        'ingresos_totales': [],
        'gastos_totales': [],
        'perdidas_totales': [],
        'ganancia_neta': [],
        'tasa_morosidad': [],
        'clientes_activos_final': []
    }
    
    for resultado in resultados_escenario:
        for metrica in metricas:
            metricas[metrica].append(resultado[metrica])
    
    # Calculate statistics
    estadisticas = {}
    for metrica, valores in metricas.items():
        valores_np = np.array(valores)
        estadisticas[metrica] = {
            'mean': np.mean(valores_np),
            'std': np.std(valores_np),
            'min': np.min(valores_np),
            'max': np.max(valores_np),
            'median': np.median(valores_np),
            'q25': np.percentile(valores_np, 25),
            'q75': np.percentile(valores_np, 75),
            'all_values': valores  # Keep all values for detailed analysis
        }
    
    # Add parameters from first simulation (they're the same for all)
    estadisticas['parametros'] = resultados_escenario[0]['parametros']
    
    # Calculate monthly statistics
    monthly_stats = []
    num_months = len(resultados_escenario[0]['resultados_mes'])
    
    for month in range(num_months):
        month_data = {
            'mes_global': resultados_escenario[0]['resultados_mes'][month]['mes_global'],
            'anio': resultados_escenario[0]['resultados_mes'][month]['anio'],
            'mes': resultados_escenario[0]['resultados_mes'][month]['mes'],
            'ingresos_comisiones': [],
            'ingresos_intereses': [],
            'ingresos': [],
            'gastos': [],
            'perdidas': [],
            'clientes_activos': [],
            'clientes_morosos': []
        }
        
        for resultado in resultados_escenario:
            mes_actual = resultado['resultados_mes'][month]
            for key in month_data.keys():
                if key not in ['mes_global', 'anio', 'mes']:
                    month_data[key].append(mes_actual[key])
        
        # Calculate statistics for each metric
        month_stats = {
            'mes_global': month_data['mes_global'],
            'anio': month_data['anio'],
            'mes': month_data['mes']
        }
        
        for key, values in month_data.items():
            if key not in ['mes_global', 'anio', 'mes']:
                values_np = np.array(values)
                month_stats[f'{key}_mean'] = np.mean(values_np)
                month_stats[f'{key}_std'] = np.std(values_np)
                month_stats[f'{key}_q25'] = np.percentile(values_np, 25)
                month_stats[f'{key}_median'] = np.median(values_np)
                month_stats[f'{key}_q75'] = np.percentile(values_np, 75)
        
        monthly_stats.append(month_stats)
    
    estadisticas['resultados_mes'] = monthly_stats
    return estadisticas


def _save_scenario_stats(args: Tuple[str, Dict, str, str]) -> None:
    """
    Helper function to save statistics for a single scenario.
    
    Args:
        args: Tuple containing (scenario_name, stats, directory, timestamp)
    """
    scenario, stats, directorio, timestamp = args
    
    # Convert numpy types to native Python types for JSON serialization
    stats_json = {}
    for key, value in stats.items():
        if key == 'resultados_mes':
            stats_json[key] = value
        elif key == 'parametros':
            stats_json[key] = value
        else:
            stats_json[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else (
                    float(v) if isinstance(v, (np.float32, np.float64)) else (
                        int(v) if isinstance(v, (np.int32, np.int64)) else v
                    )
                )
                for k, v in value.items()
            }
    
    # Save statistics JSON
    with open(f"{directorio}/estadisticas_{scenario}_{timestamp}.json", 'w') as f:
        json.dump(stats_json, f)
    
    # Save monthly results CSV
    datos_mensuales = []
    for mes in stats['resultados_mes']:
        datos_mensuales.append({
            'mes_global': int(mes['mes_global']),
            'anio': int(mes['anio']),
            'mes': int(mes['mes']),
            'ingresos_comisiones_mean': float(mes['ingresos_comisiones_mean']),
            'ingresos_comisiones_std': float(mes['ingresos_comisiones_std']),
            'ingresos_intereses_mean': float(mes['ingresos_intereses_mean']),
            'ingresos_intereses_std': float(mes['ingresos_intereses_std']),
            'ingresos_mean': float(mes['ingresos_mean']),
            'ingresos_std': float(mes['ingresos_std']),
            'gastos_mean': float(mes['gastos_mean']),
            'gastos_std': float(mes['gastos_std']),
            'perdidas_mean': float(mes['perdidas_mean']),
            'perdidas_std': float(mes['perdidas_std']),
            'clientes_activos_mean': float(mes['clientes_activos_mean']),
            'clientes_activos_std': float(mes['clientes_activos_std']),
            'clientes_morosos_mean': float(mes['clientes_morosos_mean']),
            'clientes_morosos_std': float(mes['clientes_morosos_std'])
        })
    
    df_mensual = pd.DataFrame(datos_mensuales)
    df_mensual.to_csv(f"{directorio}/resultados_mensuales_{scenario}_{timestamp}.csv", index=False)


def guardar_resultados(resultados: Dict[str, List[Dict[str, Any]]], directorio: str) -> str:
    """
    Saves simulation results to a single comprehensive CSV file.
    
    Args:
        resultados: Dictionary with results for each scenario and seed
        directorio: Directory where to save the results
        
    Returns:
        Path to the saved CSV file
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    # Create a list to store all results in a flat structure
    all_results = []
    
    # Process each scenario
    for scenario, scenario_results in resultados.items():
        # Process each simulation run (seed)
        for seed_index, result in enumerate(scenario_results):
            # Add summary metrics
            summary_row = {
                'scenario': scenario,
                'seed': seed_index,
                'total_income': result['ingresos_totales'],
                'total_expenses': result['gastos_totales'],
                'total_losses': result['perdidas_totales'],
                'net_profit': result['ganancia_neta'],
                'delinquency_rate': result['tasa_morosidad'],
                'active_clients_final': result['clientes_activos_final'],
                'num_clients_initial': result['parametros']['num_clientes'],
                'simulation_type': 'summary'
            }
            all_results.append(summary_row)
            
            # Add monthly results
            for month_data in result['resultados_mes']:
                month_row = {
                    'scenario': scenario,
                    'seed': seed_index,
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
                all_results.append(month_row)
    
    # Create a DataFrame with all results
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV with fixed name
    csv_path = f"{directorio}/simulation_results.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"Results saved to {csv_path}")
    return csv_path


def crear_visualizaciones_estadisticas(estadisticas: Dict[str, Dict[str, Any]], directorio: str, timestamp: str) -> None:
    """
    Crea visualizaciones estadísticas de los resultados.
    
    Args:
        estadisticas: Diccionario con las estadísticas de cada escenario
        directorio: Directorio donde guardar las visualizaciones
        timestamp: Timestamp para los nombres de archivo
    """
    # Create directory for visualizations
    vis_dir = f"{directorio}/visualizaciones_{timestamp}"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 1. Box plots of net profits by scenario
    plt.figure(figsize=(12, 6))
    data = []
    labels = []
    for escenario, stats in estadisticas.items():
        data.append(stats['ganancia_neta']['all_values'])
        labels.extend([escenario] * len(stats['ganancia_neta']['all_values']))
    
    plt.boxplot(data, labels=list(estadisticas.keys()))
    plt.title('Distribución de Ganancias Netas por Escenario')
    plt.xlabel('Escenario')
    plt.ylabel('Ganancia Neta')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/distribucion_ganancias.png")
    plt.close()
    
    # 2. Monthly income evolution with confidence intervals
    plt.figure(figsize=(15, 8))
    
    for escenario, stats in estadisticas.items():
        meses = [mes['mes_global'] for mes in stats['resultados_mes']]
        ingresos_mean = [mes['ingresos_mean'] for mes in stats['resultados_mes']]
        ingresos_std = [mes['ingresos_std'] for mes in stats['resultados_mes']]
        
        plt.plot(meses, ingresos_mean, label=escenario)
        plt.fill_between(meses, 
                        [m - 2*s for m, s in zip(ingresos_mean, ingresos_std)],
                        [m + 2*s for m, s in zip(ingresos_mean, ingresos_std)],
                        alpha=0.2)
    
    plt.title('Evolución de Ingresos Mensuales con Intervalos de Confianza')
    plt.xlabel('Mes')
    plt.ylabel('Ingresos')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/evolucion_ingresos.png")
    plt.close()


def ejemplo_uso():
    """
    Ejemplo de uso del modelo de simulación de cartera de tarjetas.
    """
    # Definir parámetros (optimista, neutral, pesimista)
    params = CreditCardParams(
        # Portfolio size
        num_clientes=(10000, 8000, 5000),
        
        # Customer behavior
        perc_totaleros=(0.4, 0.3, 0.2),  # % of customers who pay in full
        perc_morosidad=(0.01, 0.03, 0.05),  # % chance of delinquency per month
        
        # Credit line parameters
        linea_credito_prom=(50000, 40000, 30000),  # Average credit line
        
        # Utilization parameters
        util_credito_totaleros=(0.7, 0.6, 0.5),  # % utilization for full payers
        # Beta distribution parameters calibrated to give ~33% mean utilization
        util_credito_revolventes_alpha=(2.0, 1.8, 1.5),
        util_credito_revolventes_beta=(4.0, 3.6, 3.0),
        
        # Payment parameters
        pago_minimo_perc=(0.05, 0.05, 0.05),  # Minimum payment as % of balance
        prob_pago_minimo=(0.7, 0.75, 0.8),  # Probability of paying only minimum
        
        # Financial parameters
        tasa_interes=(36.0, 42.0, 48.0),  # Annual interest rate (%)
        comision_venta=(0.03, 0.025, 0.02),  # Interchange fee (%)
        costo_emision=(500, 600, 700),  # Cost per card (includes all operational costs)
        
        # Simulation parameters
        semilla_aleatoria=42  # Random seed for reproducibility
    )
    
    # Simular escenarios en paralelo
    num_years = 3
    resultados = simular_escenarios_paralelo(params, num_years=num_years)
    
    # Guardar resultados
    guardar_resultados(resultados, "resultados_simulacion")
    
    # Mostrar resumen
    print("\nResumen de resultados:")
    for escenario, resultado in resultados.items():
        print(f"\nEscenario: {escenario}")
        print(f"Ingresos totales: ${resultado['ingresos_totales']:,.2f}")
        print(f"Gastos totales: ${resultado['gastos_totales']:,.2f}")
        print(f"Pérdidas por morosidad: ${resultado['perdidas_totales']:,.2f}")
        print(f"Ganancia neta: ${resultado['ganancia_neta']:,.2f}")
        print(f"Tasa de morosidad final: {resultado['tasa_morosidad']:.2%}")
        print(f"Clientes activos al final: {resultado['clientes_activos_final']} de {params.get_scenario_params(escenario)['num_clientes']}")


if __name__ == "__main__":
    ejemplo_uso()
