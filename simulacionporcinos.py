#Este es un desarrollo para monitoreo de un sistema agropecuario desde el análisis de las capacidades,
#flujo de producción tanto reproductivo como de engorda y evaluación de aspectos financieros.

import pandas as pd
import numpy as np
from numpy.random import uniform, poisson

class ModeloPorcinoIntegrado:
    def __init__(self, n_semanas=52):
        self.n_semanas = n_semanas
        self._configurar_parametros()
        
    def _configurar_parametros(self):
        # Condiciones iniciales
        self.condiciones_iniciales = {
            'engorda': 8236,
            'maternidad': 2400,
            'recria': 5753,
            'hembras_gestacion': 1040,
            'hembras_lactantes': 180,
            'hembras_crianza': 280,
            'hembras_pubertad': 80,
            'hembras_pre_monta': 100
        }
        
        # Restricciones del sistema
        self.restricciones = {
            'max_hembras_total': 1300,
            'max_cerdos_engorda': 8000,
            'min_cerdos_engorda': 7000,
            'max_cerdos_recria': 5000,
            'max_venta_semanal': 850,
            'min_venta_semanal': 400,
            'max_venta_mensual': 3200,
            'max_montas_semana': 70,
            'min_montas_semana': 62
        }
        
        # Parámetros productivos
        self.parametros = {
            'tasa_fertilidad': (82, 88),
            'mortalidad_gestacion': (0.77, 2.0),
            'mortalidad_lactancia': (5, 10),
            'mortalidad_recria': (0.32, 4.3),
            'mortalidad_engorda': (70/3200, 90/3200),
            'dias_no_productivos': (3, 7),
            'total_nacidos': (14.20 16.3),
            'peso_venta': (115, 125),
            'costo_kg': (1400, 1600),
            'precio_venta_kg': (1400, 1700)
        }

    def ejecutar_modelo(self):
        """Método principal que ejecuta todo el modelo"""
        df = self._generar_datos_base()
        df = self._calcular_flujo_completo(df)
        df = self._calcular_rentabilidad(df)

#Este apartado genera la condición inicial para una semana determinada "i". Considera tanto el flujo reproductivo como también el flujo de cerdos entre maternidad y recría.
def _generar_datos_base(self):
    """Genera el DataFrame base con todas las variables iniciales"""
    datos = {
        'semana': range(1, self.n_semanas + 1),
        
        # Variables reproductivas
        'montas_semana': uniform(
            self.restricciones['min_montas_semana'],
            self.restricciones['max_montas_semana'],
            self.n_semanas
        ),
        'hembras_destete_semana': uniform(50, 55, self.n_semanas),
        
        # Mortalidades
        'mortalidad_transito_cliente': uniform(4, 7, self.n_semanas),
        'mortalidad_transito_engorda': poisson(4, self.n_semanas),
        
        # Variables productivas
        'lechones_destete_semana': uniform(600, 850, self.n_semanas),
        
        # Variables económicas
        'costo_kg_real': uniform(
            self.parametros['costo_kg'][0],
            self.parametros['costo_kg'][1],
            self.n_semanas
        ),
        'precio_venta_kg': uniform(
            self.parametros['precio_venta_kg'][0],
            self.parametros['precio_venta_kg'][1],
            self.n_semanas
        )
    }
    return pd.DataFrame(datos)

def _calcular_flujo_completo(self, df):
    """Integra los flujos reproductivo y productivo con restricciones"""
    
    # Inicialización de inventarios
    df.loc[0, 'inventario_engorda'] = self.condiciones_iniciales['engorda']
    df.loc[0, 'inventario_recria'] = self.condiciones_iniciales['recria']
    df.loc[0, 'hembras_gestacion'] = self.condiciones_iniciales['hembras_gestacion']
    df.loc[0, 'hembras_lactantes'] = self.condiciones_iniciales['hembras_lactantes']
    
    # Inicializar histórico de montas (16 semanas previas)
    for sem_previa in range(-16, 0):
        df.loc[sem_previa, 'montas_semana'] = uniform(60, 68)
        repeticion_celo = uniform(8, 12) / 100
        df.loc[sem_previa, 'repeticiones'] = df.loc[sem_previa, 'montas_semana'] * repeticion_celo
        df.loc[sem_previa, 'potenciales_gestantes'] = (
            df.loc[sem_previa, 'montas_semana'] * (1 - repeticion_celo)
        )
    
    for semana in range(1, self.n_semanas):
        # 1. Flujo Reproductivo
        # Montas nuevas
        df.loc[semana, 'montas_semana'] = uniform(60, 68)
        repeticion_celo = uniform(8, 12) / 100
        df.loc[semana, 'repeticiones'] = df.loc[semana, 'montas_semana'] * repeticion_celo
        
        # Gestación
        df.loc[semana, 'potenciales_gestantes'] = (
            df.loc[semana, 'montas_semana'] * (1 - repeticion_celo)
        )
        
        # Partos (de montas de hace 16 semanas)
        montas_anteriores = df.loc[semana - 16, 'montas_semana']
        repeticiones_hist = df.loc[semana - 16, 'repeticiones']
        mortalidad_gestacion = uniform(0.77, 2.0) / 100
        secreciones = uniform(1, 3) / 100
        
        df.loc[semana, 'partos_efectivos'] = (
            montas_anteriores - 
            repeticiones_hist - 
            (montas_anteriores * mortalidad_gestacion) -
            (montas_anteriores * secreciones)
        )
        
        # Actualización de inventario de hembras
        df.loc[semana, 'hembras_gestacion'] = (
            df.loc[semana - 1, 'hembras_gestacion'] +
            df.loc[semana, 'potenciales_gestantes'] -
            df.loc[semana, 'partos_efectivos']
        )
        
        df.loc[semana, 'hembras_lactantes'] = (
            df.loc[semana - 1, 'hembras_lactantes'] +
            df.loc[semana, 'partos_efectivos'] -
            df.loc[semana - 3, 'partos_efectivos'].fillna(0)  # Destetes (21 días)
        )
        
        # Lechones nacidos
        total_nacidos = uniform(
            self.parametros['total_nacidos'][0],
            self.parametros['total_nacidos'][1]
        )
        df.loc[semana, 'lechones_nacidos'] = df.loc[semana, 'partos_efectivos'] * total_nacidos
        
        # 2. Flujo Productivo
        # Maternidad y Destete
        mortalidad_lactancia = uniform(
            self.parametros['mortalidad_lactancia'][0],
            self.parametros['mortalidad_lactancia'][1]
        ) / 100
        
        df.loc[semana, 'lechones_destetados'] = (
            df.loc[semana, 'lechones_nacidos'] * (1 - mortalidad_lactancia)
        )
        
        # Resto del flujo productivo igual...
        #[El resto del código continúa igual con recría y engorda]
        # Continuación del método _calcular_flujo_completo...
        # Recría (7 semanas)
        if semana >= 7:
            mortalidad_recria = uniform(
                self.parametros['mortalidad_recria'][0],
                self.parametros['mortalidad_recria'][1]
            ) / 100
            
            df.loc[semana, 'entrada_recria'] = df.loc[semana - 7, 'lechones_destetados']
            df.loc[semana, 'salida_recria'] = (
                df.loc[semana, 'entrada_recria'] * (1 - mortalidad_recria)
            ) - df.loc[semana, 'mortalidad_transito_engorda']
            
            # Actualización inventario recría
            df.loc[semana, 'inventario_recria'] = (
                df.loc[semana - 1, 'inventario_recria'] +
                df.loc[semana, 'lechones_destetados'] -
                df.loc[semana, 'salida_recria']
            )
            
            # Control inventario recría
            if df.loc[semana, 'inventario_recria'] > self.restricciones['max_cerdos_recria']:
                df.loc[semana, 'alerta_recria'] = (
                    f"Exceso: {df.loc[semana, 'inventario_recria'] - self.restricciones['max_cerdos_recria']:.0f}"
                )
        
        # Engorda (16 semanas)
        if semana >= 16:
            mortalidad_engorda = uniform(
                self.parametros['mortalidad_engorda'][0],
                self.parametros['mortalidad_engorda'][1]
            )
            
            df.loc[semana, 'entrada_engorda'] = df.loc[semana - 16, 'salida_recria']
            df.loc[semana, 'disponibles_venta'] = (
                df.loc[semana, 'entrada_engorda'] * (1 - mortalidad_engorda)
            ) - df.loc[semana, 'mortalidad_transito_cliente']
            
            # Control de ventas semanales
            df.loc[semana, 'venta_programada'] = np.clip(
                df.loc[semana, 'disponibles_venta'],
                self.restricciones['min_venta_semanal'],
                self.restricciones['max_venta_semanal']
            )
            
            # Actualización inventario engorda
            df.loc[semana, 'inventario_engorda'] = (
                df.loc[semana - 1, 'inventario_engorda'] +
                df.loc[semana, 'entrada_engorda'] -
                df.loc[semana, 'venta_programada']
            )
            
            # Control inventario engorda
            if df.loc[semana, 'inventario_engorda'] < self.restricciones['min_cerdos_engorda']:
                df.loc[semana, 'alerta_engorda'] = (
                    f"Déficit: {self.restricciones['min_cerdos_engorda'] - df.loc[semana, 'inventario_engorda']:.0f}"
                )
            elif df.loc[semana, 'inventario_engorda'] > self.restricciones['max_cerdos_engorda']:
                df.loc[semana, 'alerta_engorda'] = (
                    f"Exceso: {df.loc[semana, 'inventario_engorda'] - self.restricciones['max_cerdos_engorda']:.0f}"
                )
        
        # Control de venta mensual
        if semana % 4 == 0 and semana >= 3:  # Cada 4 semanas
            ventas_mes = df.loc[semana-3:semana, 'venta_programada'].sum()
            if ventas_mes > self.restricciones['max_venta_mensual']:
                factor_ajuste = self.restricciones['max_venta_mensual'] / ventas_mes
                df.loc[semana-3:semana, 'venta_programada'] *= factor_ajuste
                df.loc[semana, 'ajuste_ventas'] = f"Ajuste mensual: {factor_ajuste:.2f}"
                
                # Recalcular inventario engorda después del ajuste
                for sem_ajuste in range(semana-3, semana+1):
                    df.loc[sem_ajuste, 'inventario_engorda'] = (
                        df.loc[sem_ajuste - 1, 'inventario_engorda'] +
                        df.loc[sem_ajuste, 'entrada_engorda'] -
                        df.loc[sem_ajuste, 'venta_programada']
                    )
        
        # Consolidar alertas
        alertas = []
        for col in ['alerta_recria', 'alerta_engorda', 'ajuste_ventas']:
            if pd.notna(df.loc[semana, col]):
                alertas.append(df.loc[semana, col])
        
        if alertas:
            df.loc[semana, 'alertas'] = ' | '.join(alertas)
        
        # 3. Aplicar restricciones adicionales si es necesario
        df = self._aplicar_restricciones(df, semana)
return df

# Aplicacion de restricciones para el modelo de simulación, considerando las capacidades máximas,
#resultados de inventario inicial, las cantidades máximas que controlan los flujos de producción.

def _aplicar_restricciones(self, df, semana):
    # 1. Control de Inventarios y Capacidades
    def verificar_inventarios():
        alertas = []
        
        # Control Recría
        
        inv_recria = df.loc[semana, 'inventario_recria']
        if inv_recria > self.restricciones['max_cerdos_recria']:
            exceso = inv_recria - self.restricciones['max_cerdos_recria']
            alertas.append(f"Exceso recría: {exceso:.0f}")
            # Ajuste automático si es necesario
            df.loc[semana, 'inventario_recria'] = self.restricciones['max_cerdos_recria']
        
        # Control Engorda
        
        inv_engorda = df.loc[semana, 'inventario_engorda']
        if inv_engorda < self.restricciones['min_cerdos_engorda']:
            deficit = self.restricciones['min_cerdos_engorda'] - inv_engorda
            alertas.append(f"Déficit engorda: {deficit:.0f}")
        elif inv_engorda > self.restricciones['max_cerdos_engorda']:
            exceso = inv_engorda - self.restricciones['max_cerdos_engorda']
            alertas.append(f"Exceso engorda: {exceso:.0f}")
            
        return alertas

    # 2. Control de Ventas
    def ajustar_ventas():
        # Control semanal
        df.loc[semana, 'venta_programada'] = np.clip(
            df.loc[semana, 'disponibles_venta'],
            self.restricciones['min_venta_semanal'],
            self.restricciones['max_venta_semanal']
        )
        
        # Control mensual
        if semana % 4 == 0 and semana >= 3:
            ventas_mes = df.loc[semana-3:semana, 'venta_programada'].sum()
            if ventas_mes > self.restricciones['max_venta_mensual']:
                factor_ajuste = self.restricciones['max_venta_mensual'] / ventas_mes
                df.loc[semana-3:semana, 'venta_programada'] *= factor_ajuste
                return f"Ajuste ventas mensual: {factor_ajuste:.2f}"
        return None

    # 3. Control Población Reproductiva
    def verificar_poblacion_reproductiva():
        total_hembras = sum([
            df.loc[semana, col].fillna(0) for col in [
                'hembras_gestacion', 'hembras_lactantes',
                'hembras_crianza', 'hembras_pubertad',
                'hembras_pre_monta'
            ]
        ])
        
        if total_hembras > self.restricciones['max_hembras_total']:
            return f"Exceso hembras: {total_hembras - self.restricciones['max_hembras_total']:.0f}"
        return None

    # Aplicar todas las restricciones y recopilar alertas
    alertas = verificar_inventarios()
    alerta_ventas = ajustar_ventas()
    alerta_reproductiva = verificar_poblacion_reproductiva()
    
    if alerta_ventas:
        alertas.append(alerta_ventas)
    if alerta_reproductiva:
        alertas.append(alerta_reproductiva)
    
    df.loc[semana, 'alertas'] = ' | '.join(alertas) if alertas else None
    
    return df

def _calcular_rentabilidad(self, df):
    """Cálculos financieros y de rentabilidad"""
    
    # 1. Cálculos por Cerdo
    df['peso_promedio'] = uniform(
        self.parametros['peso_venta'][0],
        self.parametros['peso_venta'][1],
        self.n_semanas
    )
    
    df['kg_producidos'] = df['venta_programada'] * df['peso_promedio']
    
    # 2. Ingresos y Costos
    df['ingreso_total'] = df['kg_producidos'] * df['precio_venta_kg']
    df['costo_total'] = df['kg_producidos'] * df['costo_kg_real']
    
    # 3. Márgenes y Rentabilidad
    df['margen_operacional'] = df['ingreso_total'] - df['costo_total']
    df['margen_por_kg'] = df['precio_venta_kg'] - df['costo_kg_real']
    df['margen_porcentual'] = (df['margen_por_kg'] / df['costo_kg_real']) * 100
    df['roi'] = (df['margen_operacional'] / df['costo_total']) * 100
    
    # 4. Indicadores Acumulados
    df['ingreso_acumulado'] = df['ingreso_total'].cumsum()
    df['costo_acumulado'] = df['costo_total'].cumsum()
    df['margen_acumulado'] = df['margen_operacional'].cumsum()
    
return df


import matplotlib.pyplot as plt
import seaborn as sns

class ModeloPorcinoIntegrado:  # Continuación de la clase
    def _generar_reportes(self, df):
        """Sistema integral de reportes y visualizaciones"""
        self._reporte_productivo(df)
        self._reporte_financiero(df)
        self._reporte_alertas(df)
        self._generar_graficos(df)
    
    def _reporte_productivo(self, df):
        """Reporte detallado de indicadores productivos"""
        print("\n=== REPORTE PRODUCTIVO ===")
        
        # 1. Indicadores Reproductivos
        print("\nIndicadores Reproductivos (Promedios Semanales):")
        print(f"Montas efectivas: {df['montas_efectivas'].mean():.1f}")
        print(f"Partos: {df['partos'].mean():.1f}")
        print(f"Lechones nacidos: {df['lechones_nacidos'].mean():.1f}")
        print(f"Lechones destetados: {df['lechones_destetados'].mean():.1f}")
        
        # 2. Flujo Productivo
        print("\nFlujo Productivo (Promedios Semanales):")
        print(f"Entrada recría: {df['entrada_recria'].mean():.1f}")
        print(f"Salida recría: {df['salida_recria'].mean():.1f}")
        print(f"Entrada engorda: {df['entrada_engorda'].mean():.1f}")
        print(f"Ventas: {df['venta_programada'].mean():.1f}")
        
        # 3. Inventarios
        print("\nInventarios Promedio:")
        print(f"Recría: {df['inventario_recria'].mean():.0f}")
        print(f"Engorda: {df['inventario_engorda'].mean():.0f}")
        
        # 4. Mortalidades
        print("\nMortalidades Promedio:")
        mortalidad_total = (1 - (df['venta_programada'].sum() / 
                               df['lechones_nacidos'].sum())) * 100
        print(f"Mortalidad total sistema: {mortalidad_total:.1f}%")
    
    def _reporte_financiero(self, df):
        """Reporte detallado financiero"""
        print("\n=== REPORTE FINANCIERO ===")
        
        # 1. Indicadores por Kg
        print("\nIndicadores por Kg (Promedios):")
        print(f"Costo/kg: ${df['costo_kg_real'].mean():.0f}")
        print(f"Precio venta/kg: ${df['precio_venta_kg'].mean():.0f}")
        print(f"Margen/kg: ${df['margen_por_kg'].mean():.0f}")
        
        # 2. Resultados Financieros
        print("\nResultados Financieros:")
        print(f"Ingreso total: ${df['ingreso_total'].sum():,.0f}")
        print(f"Costo total: ${df['costo_total'].sum():,.0f}")
        print(f"Margen operacional: ${df['margen_operacional'].sum():,.0f}")
        print(f"ROI promedio: {df['roi'].mean():.1f}%")
        
        # 3. Análisis de Tendencias
        tendencia = np.polyfit(df.index, df['margen_operacional'], 1)[0]
        print(f"\nTendencia margen: {'Positiva' if tendencia > 0 else 'Negativa'}")
    
    def _reporte_alertas(self, df):
        """Reporte de alertas y restricciones"""
        print("\n=== REPORTE DE ALERTAS ===")
        
        alertas = df[df['alertas'].notna()]['alertas']
        if not alertas.empty:
            print("\nSemanas con alertas:")
            for semana, alerta in alertas.items():
                print(f"Semana {semana}: {alerta}")
    
    def _generar_graficos(self, df):
        """Generación de visualizaciones clave"""
        # 1. Flujo Productivo
        plt.figure(figsize=(12, 6))
        plt.plot(df['semana'], df['lechones_destetados'], label='Destetados')
        plt.plot(df['semana'], df['venta_programada'], label='Ventas')
        plt.title('Flujo Productivo Semanal')
        plt.xlabel('Semana')
        plt.ylabel('Cantidad')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 2. Márgenes
        plt.figure(figsize=(12, 6))
        plt.plot(df['semana'], df['margen_operacional'], 'g-')
        plt.title('Margen Operacional Semanal')
        plt.xlabel('Semana')
        plt.ylabel('Margen ($)')
        plt.grid(True)
        plt.show()
        
        # 3. Inventarios
        plt.figure(figsize=(12, 6))
        plt.plot(df['semana'], df['inventario_recria'], label='Recría')
        plt.plot(df['semana'], df['inventario_engorda'], label='Engorda')
        plt.title('Inventarios Semanales')
        plt.xlabel('Semana')
        plt.ylabel('Cantidad')
        plt.legend()
        plt.grid(True)
        plt.show()

    def guardar_resultados(self, df, nombre_archivo):
        """Guarda los resultados en Excel"""
        with pd.ExcelWriter(nombre_archivo) as writer:
            df.to_excel(writer, sheet_name='Datos_Semanales')
            
            # Resumen ejecutivo
            resumen = pd.DataFrame({
                'Indicador': [
                    'Ventas promedio/semana',
                    'Margen total',
                    'ROI promedio',
                    'Mortalidad total'
                ],
                'Valor': [
                    f"{df['venta_programada'].mean():.1f}",
                    f"${df['margen_operacional'].sum():,.0f}",
                    f"{df['roi'].mean():.1f}%",
                    f"{(1 - df['venta_programada'].sum() / df['lechones_nacidos'].sum()) * 100:.1f}%"
                ]
            })
            resumen.to_excel(writer, sheet_name='Resumen_Ejecutivo')
        self._generar_reportes(df)
        return df