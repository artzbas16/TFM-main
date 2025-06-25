import os
import sys
from datetime import datetime

# Añadir el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_complete_strategic_analysis():
    """Ejecuta análisis completo de aprendizaje estratégico"""
    
    print("🧠 ANÁLISIS COMPLETO DE APRENDIZAJE ESTRATÉGICO")
    print("=" * 50)
    print(f"🕐 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Crear directorios necesarios
    directories = [
        "strategic_analysis",
        "learning_analysis"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directorio creado/verificado: {directory}")
    
    print("\n" + "="*50)
    print("🎯 FASE 1: ANÁLISIS ESTRATÉGICO COMPARATIVO")
    print("="*50)
    
    try:
        from strategic_analysis import StrategicAnalysisSystem
        
        analyzer = StrategicAnalysisSystem()
        analyzer.load_ia_statistics()
        
        # Crear gráficos comparativos
        similarity_matrix = analyzer.create_strategic_comparison_plots()
        
        # Generar informe de evaluación
        report = analyzer.generate_learning_assessment_report()
        
        print("✅ Análisis estratégico completado")
        
    except Exception as e:
        print(f"❌ Error en análisis estratégico: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("📈 FASE 2: ANÁLISIS DE MÉTRICAS DE APRENDIZAJE")
    print("="*50)
    
    try:
        from learning_metrics import LearningMetricsAnalyzer
        
        metrics_analyzer = LearningMetricsAnalyzer()
        
        # Simular datos de entrenamiento para demostración
        import numpy as np
        training_data = {
            'episode_rewards': {
                'jugador_0': np.random.randn(400).cumsum() + np.linspace(-5, 15, 400),
                'jugador_1': np.random.randn(400).cumsum() + np.linspace(-3, 12, 400),
                'jugador_2': np.random.randn(400).cumsum() + np.linspace(-4, 14, 400),
                'jugador_3': np.random.randn(400).cumsum() + np.linspace(-2, 11, 400)
            },
            'episode_lengths': np.random.randint(30, 150, 400),
            'epsilon_values': {
                'jugador_0': np.exp(-np.linspace(0, 6, 400)),
                'jugador_1': np.exp(-np.linspace(0, 6, 400)),
                'jugador_2': np.exp(-np.linspace(0, 6, 400)),
                'jugador_3': np.exp(-np.linspace(0, 6, 400))
            },
            'win_rates': {
                'equipo_1': np.random.choice([0, 1], 400, p=[0.45, 0.55]),
                'equipo_2': np.random.choice([0, 1], 400, p=[0.55, 0.45])
            },
            'phase_success': {
                'GRANDE': np.random.choice([0, 1], 400, p=[0.4, 0.6]),
                'CHICA': np.random.choice([0, 1], 400, p=[0.35, 0.65]),
                'PARES': np.random.choice([0, 1], 400, p=[0.5, 0.5]),
                'JUEGO': np.random.choice([0, 1], 400, p=[0.45, 0.55])
            }
        }
        
        # Ejecutar análisis de convergencia
        metrics_analyzer.analyze_convergence_patterns(training_data)
        
        # Ejecutar evaluación de calidad
        metrics_analyzer.create_learning_quality_assessment(training_data, {})
        
        print("✅ Análisis de métricas completado")
        
    except Exception as e:
        print(f"❌ Error en análisis de métricas: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("🎉 ANÁLISIS COMPLETO TERMINADO")
    print("="*50)
    
    print(f"🕐 Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📋 ARCHIVOS GENERADOS:")
    print("   🎯 strategic_analysis/ - Comparación con jugadores expertos")
    print("   📈 learning_analysis/ - Métricas de convergencia y calidad")
    
    print("\n🧠 INTERPRETACIÓN DE RESULTADOS:")
    print("   📊 Similitud >80% = Aprendizaje excelente")
    print("   📊 Similitud 60-80% = Aprendizaje bueno")
    print("   📊 Similitud 40-60% = Necesita mejoras")
    print("   📊 Similitud <40% = Requiere reentrenamiento")
    
    print("\n🎯 ¡Revisa los gráficos para evaluar el aprendizaje de tus agentes!")


if __name__ == "__main__":
    run_complete_strategic_analysis()
