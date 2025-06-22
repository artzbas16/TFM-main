import os
import sys
import time
from datetime import datetime

# Añadir el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_complete_training_pipeline():
    """Ejecuta el pipeline completo de entrenamiento y evaluación"""
    
    print("🎮 PIPELINE COMPLETO DE ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 55)
    print(f"🕐 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Crear directorios necesarios
    directories = [
        "training_results",
        "trained_models", 
        "evaluation_results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directorio creado/verificado: {directory}")
    
    print("\n" + "="*55)
    print("🚀 FASE 1: ENTRENAMIENTO CON VISUALIZACIÓN")
    print("="*55)
    
    try:
        # Importar y ejecutar entrenamiento
        from enhanced_training import EnhancedTrainingSystem
        
        training_system = EnhancedTrainingSystem(
            max_episodes=400,  # Número de episodios para entrenamiento
            save_interval=50   # Guardar cada 50 episodios
        )
        
        print("🎯 Iniciando entrenamiento mejorado...")
        training_start = time.time()
        
        # Ejecutar entrenamiento con visualización
        training_system.train_with_visualization()
        
        training_time = time.time() - training_start
        print(f"✅ Entrenamiento completado en {training_time/60:.1f} minutos")
        
    except Exception as e:
        print(f"❌ Error en el entrenamiento: {e}")
        print("⚠️ Continuando con la evaluación usando modelos existentes...")
    
    # Pausa entre fases
    print("\n⏸️ Pausa de 5 segundos antes de la evaluación...")
    time.sleep(5)
    
    print("\n" + "="*55)
    print("📊 FASE 2: EVALUACIÓN DE AGENTES ENTRENADOS")
    print("="*55)
    
    try:
        # Importar y ejecutar evaluación
        from evaluation_system import AgentEvaluationSystem
        
        evaluation_system = AgentEvaluationSystem(
            num_evaluation_games=150  # Número de juegos para evaluación
        )
        
        print("🎯 Iniciando evaluación completa...")
        evaluation_start = time.time()
        
        # Ejecutar evaluación
        analysis = evaluation_system.run_full_evaluation()
        
        if analysis:
            # Generar informe y visualizaciones
            evaluation_system.generate_evaluation_report(analysis)
            evaluation_system.create_evaluation_visualizations(analysis)
            
            evaluation_time = time.time() - evaluation_start
            print(f"✅ Evaluación completada en {evaluation_time/60:.1f} minutos")
        else:
            print("❌ Error en la evaluación")
            
    except Exception as e:
        print(f"❌ Error en la evaluación: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*55)
    print("🎉 PIPELINE COMPLETADO")
    print("="*55)
    
    total_time = time.time() - training_start if 'training_start' in locals() else 0
    print(f"⏱️ Tiempo total: {total_time/60:.1f} minutos")
    print(f"🕐 Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📋 ARCHIVOS GENERADOS:")
    print("   📊 training_results/ - Gráficos y métricas de entrenamiento")
    print("   🤖 trained_models/ - Modelos de agentes entrenados")
    print("   📈 evaluation_results/ - Resultados y análisis de evaluación")
    
    print("\n🎯 ¡Revisa los directorios para ver todos los resultados!")


if __name__ == "__main__":
    run_complete_training_pipeline()
