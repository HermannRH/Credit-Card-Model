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
            'saldo_actual': 0,
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
                saldo_anterior = cliente['saldo_actual']
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
                    # Esta distribución beta dará un promedio cercano a 0.33 con los parámetros adecuados
                    porcentaje_utilizacion = np.random.beta(alpha, beta)
                    consumo_base = linea_disponible * porcentaje_utilizacion
                    consumo = consumo_base * factor_estacional
                
                # Aplicar comisión por venta (interchange fee)
                ingreso_comision = consumo * params['comision_venta']
                ingresos_mes += ingreso_comision
                detalles_mes['ingresos_comisiones'] += ingreso_comision
                
                # Calcular nuevo saldo después del consumo
                nuevo_saldo = saldo_anterior + consumo
                
                # Verificar si el cliente se vuelve moroso este mes
                es_moroso_nuevo = random.random() < params['perc_morosidad']
                
                # Determinar pago del cliente
                if es_moroso_nuevo:
                    # Cliente se vuelve moroso, no paga nada
                    cliente['es_moroso'] = True
                    pago = 0
                    # Registrar pérdida del principal
                    perdida = nuevo_saldo
                    perdidas_mes += perdida
                    
                    # No hay intereses para clientes morosos
                    interes_mensual = 0
                    saldo_final = 0  # El saldo se da por perdido
                    linea_disponible = 0  # No hay línea disponible para morosos
                    
                elif es_totalero:
                    # Totalero paga todo el consumo del mes
                    pago = consumo
                    saldo_final = saldo_anterior
                    interes_mensual = 0
                    linea_disponible = cliente['linea_credito'] - saldo_final
                    
                else:
                    # Cliente revolvente
                    pago_minimo = nuevo_saldo * params['pago_minimo_perc']
                    
                    # Determinar si paga el mínimo o más
                    if random.random() < params['prob_pago_minimo']:
                        # Paga solo el mínimo
                        pago = pago_minimo
                    else:
                        # Paga un monto aleatorio entre el mínimo y el total
                        pago = random.uniform(pago_minimo, nuevo_saldo)
                    
                    # Asegurar que el pago no exceda el saldo
                    pago = min(pago, nuevo_saldo)
                    
                    # Calcular saldo después del pago e intereses
                    saldo_despues_pago = nuevo_saldo - pago
                    interes_mensual = saldo_despues_pago * (params['tasa_interes'] / 100 / 12)
                    saldo_final = saldo_despues_pago + interes_mensual
                    
                    linea_disponible = cliente['linea_credito'] - saldo_final
                    ingresos_mes += interes_mensual
                    detalles_mes['ingresos_intereses'] += interes_mensual
                
                # Actualizar cliente
                cliente['saldo_actual'] = saldo_final
                cliente['linea_disponible'] = max(0, linea_disponible)
                
                # Guardar detalles del cliente en este mes
                detalle_cliente = {
                    'id': cliente['id'],
                    'es_totalero': es_totalero,
                    'consumo': consumo,
                    'saldo_anterior': saldo_anterior,
                    'pago': pago,
                    'saldo_final': cliente['saldo_actual'],
                    'linea_disponible': cliente['linea_disponible'],
                    'ingreso_comision': ingreso_comision,
                    'ingreso_intereses': interes_mensual if not es_totalero else 0,
                    'es_moroso': cliente['es_moroso'],
                    'perdida': perdida if es_moroso_nuevo else 0
                }
                detalles_mes['clientes'].append(detalle_cliente)
            
            # Actualizar totales
            detalles_mes['ingresos'] = detalles_mes['ingresos_comisiones'] + detalles_mes['ingresos_intereses']
            detalles_mes['perdidas'] = perdidas_mes
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
                               num_procesos: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Simula todos los escenarios en paralelo.
    
    Args:
        params: Objeto CreditCardParams con los valores optimistas, neutrales y pesimistas
        num_years: Número de años a simular
        num_procesos: Número de procesos a utilizar (None = usar todos los disponibles)
        
    Returns:
        Diccionario con los resultados de cada escenario
    """
    escenarios = generar_escenarios(params)
    args = [(nombre, params_escenario, num_years) for nombre, params_escenario in escenarios]
    
    if num_procesos is None:
        num_procesos = mp.cpu_count()
    
    with mp.Pool(processes=num_procesos) as pool:
        resultados = list(tqdm(pool.imap(simular_escenario, args), 
                              total=len(args), 
                              desc="Simulando escenarios"))
    
    return dict(resultados)


def guardar_resultados(resultados: Dict[str, Dict[str, Any]], directorio: str) -> None:
    """
    Guarda los resultados de la simulación en archivos CSV y JSON.
    
    Args:
        resultados: Diccionario con los resultados de cada escenario
        directorio: Directorio donde guardar los resultados
    """
    # Crear directorio si no existe
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar resumen de escenarios
    resumen = []
    for escenario, resultado in resultados.items():
        resumen.append({
            'escenario': escenario,
            'ingresos_totales': resultado['ingresos_totales'],
            'gastos_totales': resultado['gastos_totales'],
            'perdidas_totales': resultado['perdidas_totales'],
            'ganancia_neta': resultado['ganancia_neta'],
            'tasa_morosidad': resultado['tasa_morosidad'],
            'clientes_activos_final': resultado['clientes_activos_final']
        })
    
    df_resumen = pd.DataFrame(resumen)
    df_resumen.to_csv(f"{directorio}/resumen_escenarios_{timestamp}.csv", index=False)
    
    # Guardar resultados mensuales para cada escenario
    for escenario, resultado in resultados.items():
        # Resultados mensuales
        datos_mensuales = []
        for mes in resultado['resultados_mes']:
            datos_mensuales.append({
                'mes_global': mes['mes_global'],
                'anio': mes['anio'],
                'mes': mes['mes'],
                'ingresos_comisiones': mes['ingresos_comisiones'],
                'ingresos_intereses': mes['ingresos_intereses'],
                'ingresos_totales': mes['ingresos'],
                'gastos': mes['gastos'],
                'perdidas': mes['perdidas'],
                'ganancia': mes['ingresos'] - mes['gastos'] - mes['perdidas'],
                'clientes_activos': mes['clientes_activos'],
                'clientes_morosos': mes['clientes_morosos']
            })
        
        df_mensual = pd.DataFrame(datos_mensuales)
        df_mensual.to_csv(f"{directorio}/resultados_mensuales_{escenario}_{timestamp}.csv", index=False)
        
        # Guardar parámetros utilizados
        with open(f"{directorio}/parametros_{escenario}_{timestamp}.json", 'w') as f:
            json.dump(resultado['parametros'], f, indent=4)
    
    # Crear visualizaciones
    crear_visualizaciones(resultados, directorio, timestamp)
    
    print(f"Resultados guardados en {directorio}")
    print(f"Archivos generados con timestamp: {timestamp}")


def crear_visualizaciones(resultados: Dict[str, Dict[str, Any]], directorio: str, timestamp: str) -> None:
    """
    Crea visualizaciones de los resultados.
    
    Args:
        resultados: Diccionario con los resultados de cada escenario
        directorio: Directorio donde guardar las visualizaciones
        timestamp: Timestamp para los nombres de archivo
    """
    # Crear directorio para visualizaciones
    vis_dir = f"{directorio}/visualizaciones_{timestamp}"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 1. Gráfico de ganancias netas por escenario
    plt.figure(figsize=(10, 6))
    escenarios = list(resultados.keys())
    ganancias = [resultado['ganancia_neta'] for resultado in resultados.values()]
    
    plt.bar(escenarios, ganancias)
    plt.title('Ganancia Neta por Escenario')
    plt.xlabel('Escenario')
    plt.ylabel('Ganancia Neta')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/ganancias_netas.png")
    plt.close()
    
    # 2. Gráfico de evolución de ingresos mensuales por escenario
    plt.figure(figsize=(12, 8))
    
    for escenario, resultado in resultados.items():
        meses = [mes['mes_global'] for mes in resultado['resultados_mes']]
        ingresos = [mes['ingresos'] for mes in resultado['resultados_mes']]
        plt.plot(meses, ingresos, label=escenario)
    
    plt.title('Evolución de Ingresos Mensuales')
    plt.xlabel('Mes')
    plt.ylabel('Ingresos')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/evolucion_ingresos.png")
    plt.close()
    
    # 3. Gráfico de tasa de morosidad por escenario
    plt.figure(figsize=(10, 6))
    tasas_morosidad = [resultado['tasa_morosidad'] for resultado in resultados.values()]
    
    plt.bar(escenarios, tasas_morosidad)
    plt.title('Tasa de Morosidad por Escenario')
    plt.xlabel('Escenario')
    plt.ylabel('Tasa de Morosidad')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/tasas_morosidad.png")
    plt.close()
    
    # 4. Gráfico de composición de ingresos (comisiones vs intereses)
    plt.figure(figsize=(12, 8))
    
    for escenario, resultado in resultados.items():
        comisiones = sum(mes['ingresos_comisiones'] for mes in resultado['resultados_mes'])
        intereses = sum(mes['ingresos_intereses'] for mes in resultado['resultados_mes'])
        
        plt.bar(escenario, [comisiones, intereses], label=['Comisiones', 'Intereses'])
    
    plt.title('Composición de Ingresos por Escenario')
    plt.xlabel('Escenario')
    plt.ylabel('Ingresos')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/composicion_ingresos.png")
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
