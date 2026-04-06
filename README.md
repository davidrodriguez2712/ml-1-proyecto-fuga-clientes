# Predicción de Churn de Clientes — Proyecto de Data Science Orientado a Producción

**-----------------PLANTILLA (NO ADAPTADO AL PROYECTO EN DESARROLLO...)----------**
## Resumen Ejecutivo

Este proyecto desarrolla un modelo de predicción de churn para identificar clientes con alto riesgo de abandono y permitir estrategias de retención proactivas.

El modelo utiliza variables de comportamiento, temporales e interacciones para capturar cambios en el uso del servicio y estimar la probabilidad de churn.

El modelo final es validado usando splits temporales y evaluación out-of-time (OOT) para simular despliegue real.

Resultados principales:

- ROC AUC (OOT): 0.74
- Lift Top 20%: 2.3x
- Captura de churners en decil superior: 65%
- Estabilidad validada con PSI

Impacto de negocio (simulado):

- Reducción potencial de churn: 12–18%
- Mejora en targeting de campañas
- Reducción de costos de retención

---

## Definición del Problema

El objetivo es predecir qué clientes abandonarán el servicio para:

- Priorizar campañas de retención
- Reducir pérdida de ingresos
- Mejorar lifetime value
- Optimizar incentivos

Se modela como un problema de clasificación binaria.

---

## Dataset

Variables incluidas:

- uso del servicio
- actividad del cliente
- historial de quejas
- comportamiento de pago
- features temporales

Split temporal:

Train: 2023–2024  
Test: 2025  
OOT: 2026  

Esto simula un escenario real de producción.

---

## Feature Engineering

Se crearon features de comportamiento:

- tendencia de uso
- caída de actividad
- intensidad de quejas
- features temporales
- interacciones

Ejemplos:

delta_usage  
complaints_per_usage  
active_days_ratio  

---

## Modelado

Modelos evaluados:

- Regresión Logística
- Random Forest
- XGBoost
- LightGBM

Modelo final: LightGBM

Razón:

- captura relaciones no lineales
- maneja features débiles
- detecta interacciones automáticamente

---

## Evaluación

Métricas:

- ROC AUC
- Precision Recall
- Lift
- Gain
- OOT performance
- PSI
- Estabilidad del score

---

## Impacto de Negocio

El modelo permite:

- targeting de retención
- reducción de churn
- optimización de incentivos
- priorización de clientes

Ejemplo:

Top 20% clientes de mayor riesgo  
Captura: 60% churners  

---

## Estructura del Proyecto
src/
data/
features/
models/
evaluation/

notebooks/
docs/
README.md


---

## Reproducibilidad

Instalar dependencias:

pip install -r requirements.txt




