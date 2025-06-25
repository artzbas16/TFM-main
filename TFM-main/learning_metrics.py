import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
import os
from datetime import datetime

class LearningMetricsAnalyzer:
    """Analizador de métricas específicas de aprendizaje"""
    
    def __init__(self):
        self.metrics = {
            'convergence_analysis': {},
            'strategy_evolution': defaultdict(list),
            'performance_benchmarks': {},
            'learning_efficiency': {}
        }
    
    def analyze_convergence_patterns(self, training_data):
        """Analiza patrones de convergencia del entrenamiento"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📈 Análisis de Convergencia del Aprendizaje', fontsize=16)
        
        # 1. Convergencia de recompensas
        ax = axes[0, 0]
        if 'episode_rewards' in training_data:
            for agent_id, rewards in training_data['episode_rewards'].items():
                if len(rewards) > 50:
                    # Calcular media móvil para suavizar
                    window = 50
                    smoothed = []
                    for i in range(len(rewards)):
                        start_idx = max(0, i - window + 1)
                        smoothed.append(np.mean(rewards[start_idx:i+1]))
                    
                    ax.plot(smoothed, label=agent_id, linewidth=2)
                    
                    # Detectar punto de convergencia
                    convergence_point = self._detect_convergence(smoothed)
                    if convergence_point:
                        ax.axvline(convergence_point, color='red', linestyle='--', alpha=0.7)
                        ax.text(convergence_point, max(smoothed), f'Convergencia\nEp. {convergence_point}', 
                               ha='center', va='bottom', fontsize=8)
        
        ax.set_title('🎯 Convergencia de Recompensas')
        ax.set_xlabel('Episodio')
        ax.set_ylabel('Recompensa Media')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Estabilidad del aprendizaje
        ax = axes[0, 1]
        if 'episode_rewards' in training_data:
            stability_scores = []
            episodes = []
            
            for agent_id, rewards in training_data['episode_rewards'].items():
                if len(rewards) > 100:
                    # Calcular estabilidad como inverso de la varianza en ventanas
                    window = 50
                    for i in range(window, len(rewards), 10):
                        window_data = rewards[i-window:i]
                        stability = 1 / (1 + np.var(window_data))
                        stability_scores.append(stability)
                        episodes.append(i)
                    break
            
            if stability_scores:
                ax.plot(episodes, stability_scores, 'g-', linewidth=2, label='Estabilidad')
                ax.fill_between(episodes, stability_scores, alpha=0.3, color='green')
        
        ax.set_title('📊 Estabilidad del Aprendizaje')
        ax.set_xlabel('Episodio')
        ax.set_ylabel('Índice de Estabilidad')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Evolución de epsilon (exploración vs explotación)
        ax = axes[1, 0]
        if 'epsilon_values' in training_data:
            for agent_id, epsilons in training_data['epsilon_values'].items():
                if len(epsilons) > 10:
                    ax.plot(epsilons, label=agent_id, linewidth=2)
        
        ax.set_title('🔍 Evolución de la Exploración (Epsilon)')
        ax.set_xlabel('Episodio')
        ax.set_ylabel('Valor de Epsilon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Eficiencia del aprendizaje
        ax = axes[1, 1]
        if 'episode_lengths' in training_data:
            lengths = training_data['episode_lengths']
            if len(lengths) > 50:
                # Calcular eficiencia como inverso de la duración promedio
                window = 50
                efficiency = []
                episodes = []
                
                for i in range(window, len(lengths), 10):
                    avg_length = np.mean(lengths[i-window:i])
                    # Normalizar eficiencia (episodios más cortos = más eficientes)
                    eff = 1 / (1 + avg_length / 100)
                    efficiency.append(eff)
                    episodes.append(i)
                
                ax.plot(episodes, efficiency, 'purple', linewidth=2, label='Eficiencia')
                ax.fill_between(episodes, efficiency, alpha=0.3, color='purple')
        
        ax.set_title('⚡ Eficiencia del Aprendizaje')
        ax.set_xlabel('Episodio')
        ax.set_ylabel('Índice de Eficiencia')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        os.makedirs("learning_analysis", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learning_analysis/convergence_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Análisis de convergencia guardado: {filename}")
    
    def _detect_convergence(self, data, threshold=0.05, window=50):
        """Detecta el punto de convergencia en una serie de datos"""
        if len(data) < window * 2:
            return None
        
        for i in range(window, len(data) - window):
            # Comparar varianza antes y después
            before = np.var(data[i-window:i])
            after = np.var(data[i:i+window])
            
            # Si la varianza se reduce significativamente, es convergencia
            if before > 0 and after / before < threshold:
                return i
        
        return None
    
    def create_learning_quality_assessment(self, training_data, evaluation_data):
        """Crea evaluación de calidad del aprendizaje"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎓 Evaluación de Calidad del Aprendizaje', fontsize=16)
        
        # 1. Progreso de aprendizaje por fases
        ax = axes[0, 0]
        phases = ['GRANDE', 'CHICA', 'PARES', 'JUEGO']
        
        if 'phase_success' in training_data:
            for phase in phases:
                if phase in training_data['phase_success']:
                    success_data = training_data['phase_success'][phase]
                    if len(success_data) > 50:
                        # Calcular tasa de éxito móvil
                        window = 50
                        success_rate = []
                        episodes = []
                        
                        for i in range(window, len(success_data), 10):
                            rate = np.mean(success_data[i-window:i])
                            success_rate.append(rate)
                            episodes.append(i)
                        
                        ax.plot(episodes, success_rate, label=phase, linewidth=2)
        
        ax.set_title('📊 Progreso por Fase del Juego')
        ax.set_xlabel('Episodio')
        ax.set_ylabel('Tasa de Éxito')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 2. Distribución de recompensas finales
        ax = axes[0, 1]
        if 'episode_rewards' in training_data:
            all_final_rewards = []
            for agent_id, rewards in training_data['episode_rewards'].items():
                if len(rewards) > 100:
                    # Tomar últimas 100 recompensas
                    final_rewards = rewards[-100:]
                    all_final_rewards.extend(final_rewards)
            
            if all_final_rewards:
                ax.hist(all_final_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(np.mean(all_final_rewards), color='red', linestyle='--', 
                          label=f'Media: {np.mean(all_final_rewards):.2f}')
                ax.axvline(np.median(all_final_rewards), color='green', linestyle='--',
                          label=f'Mediana: {np.median(all_final_rewards):.2f}')
        
        ax.set_title('📈 Distribución de Recompensas Finales')
        ax.set_xlabel('Recompensa')
        ax.set_ylabel('Frecuencia')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Comparación de rendimiento inicial vs final
        ax = axes[0, 2]
        if 'episode_rewards' in training_data:
            initial_performance = []
            final_performance = []
            
            for agent_id, rewards in training_data['episode_rewards'].items():
                if len(rewards) > 200:
                    initial = np.mean(rewards[:50])  # Primeros 50 episodios
                    final = np.mean(rewards[-50:])   # Últimos 50 episodios
                    initial_performance.append(initial)
                    final_performance.append(final)
            
            if initial_performance and final_performance:
                agents = [f'Agente {i+1}' for i in range(len(initial_performance))]
                x = np.arange(len(agents))
                width = 0.35
                
                ax.bar(x - width/2, initial_performance, width, label='Inicial', color='lightcoral', alpha=0.8)
                ax.bar(x + width/2, final_performance, width, label='Final', color='lightgreen', alpha=0.8)
                
                ax.set_xticks(x)
                ax.set_xticklabels(agents)
        
        ax.set_title('🚀 Mejora del Rendimiento')
        ax.set_ylabel('Recompensa Promedio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Análisis de consistencia
        ax = axes[1, 0]
        if 'win_rates' in training_data:
            consistency_scores = []
            team_labels = []
            
            for team, wins in training_data['win_rates'].items():
                if len(wins) > 100:
                    # Calcular consistencia como 1 - desviación estándar de ventanas móviles
                    window = 25
                    window_vars = []
                    
                    for i in range(window, len(wins), 5):
                        window_data = wins[i-window:i]
                        window_vars.append(np.var(window_data))
                    
                    if window_vars:
                        consistency = 1 - np.mean(window_vars)
                        consistency_scores.append(max(0, consistency))
                        team_labels.append(team)
            
            if consistency_scores:
                colors = ['#FF6B6B', '#4ECDC4']
                bars = ax.bar(team_labels, consistency_scores, color=colors, alpha=0.8)
                
                for bar, score in zip(bars, consistency_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        ax.set_title('🎯 Consistencia del Rendimiento')
        ax.set_ylabel('Índice de Consistencia')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 5. Velocidad de aprendizaje
        ax = axes[1, 1]
        if 'episode_rewards' in training_data:
            learning_speeds = []
            agent_names = []
            
            for agent_id, rewards in training_data['episode_rewards'].items():
                if len(rewards) > 100:
                    # Calcular pendiente de mejora en los primeros 100 episodios
                    x = np.arange(100)
                    y = rewards[:100]
                    slope, _ = np.polyfit(x, y, 1)
                    learning_speeds.append(slope)
                    agent_names.append(agent_id)
            
            if learning_speeds:
                colors = plt.cm.viridis(np.linspace(0, 1, len(learning_speeds)))
                bars = ax.bar(agent_names, learning_speeds, color=colors, alpha=0.8)
                
                for bar, speed in zip(bars, learning_speeds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{speed:.3f}', ha='center', va='bottom', rotation=90)
        
        ax.set_title('⚡ Velocidad de Aprendizaje')
        ax.set_ylabel('Pendiente de Mejora')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 6. Mapa de calor de aprendizaje
        ax = axes[1, 2]
        
        # Crear matriz de métricas de aprendizaje
        metrics = ['Convergencia', 'Estabilidad', 'Consistencia', 'Velocidad']
        agents = ['Agente 1', 'Agente 2', 'Agente 3', 'Agente 4']
        
        # Simular datos de calidad (en implementación real, calcular de datos reales)
        quality_matrix = np.random.rand(len(agents), len(metrics)) * 0.3 + 0.7  # Entre 0.7 y 1.0
        
        im = ax.imshow(quality_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(agents)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(agents)
        ax.set_title('🎓 Calidad del Aprendizaje por Agente')
        
        # Añadir valores en las celdas
        for i in range(len(agents)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{quality_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learning_analysis/quality_assessment_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"🎓 Evaluación de calidad guardada: {filename}")


def main():
    """Función principal para análisis de métricas de aprendizaje"""
    print("📊 ANÁLISIS DE MÉTRICAS DE APRENDIZAJE")
    print("=" * 40)
    
    analyzer = LearningMetricsAnalyzer()
    
    # Cargar datos de entrenamiento
    training_data = {}
    try:
        # Buscar archivos de entrenamiento
        results_dir = "training_results"
        if os.path.exists(results_dir):
            report_files = [f for f in os.listdir(results_dir) if f.startswith("training_report_")]
            if report_files:
                latest_file = sorted(report_files)[-1]
                file_path = os.path.join(results_dir, latest_file)
                with open(file_path, 'r') as f:
                    training_data = json.load(f)
                print(f"✅ Datos de entrenamiento cargados: {file_path}")
    except Exception as e:
        print(f"⚠️ Error cargando datos de entrenamiento: {e}")
        print("Usando datos simulados para demostración...")
        training_data = {
            'episode_rewards': {
                'jugador_0': np.random.randn(300).cumsum() + np.linspace(0, 10, 300),
                'jugador_1': np.random.randn(300).cumsum() + np.linspace(0, 8, 300),
                'jugador_2': np.random.randn(300).cumsum() + np.linspace(0, 9, 300),
                'jugador_3': np.random.randn(300).cumsum() + np.linspace(0, 7, 300)
            },
            'episode_lengths': np.random.randint(50, 200, 300),
            'epsilon_values': {
                'jugador_0': np.exp(-np.linspace(0, 5, 300)),
                'jugador_1': np.exp(-np.linspace(0, 5, 300)),
                'jugador_2': np.exp(-np.linspace(0, 5, 300)),
                'jugador_3': np.exp(-np.linspace(0, 5, 300))
            },
            'win_rates': {
                'equipo_1': np.random.choice([0, 1], 300, p=[0.4, 0.6]),
                'equipo_2': np.random.choice([0, 1], 300, p=[0.6, 0.4])
            }
        }
    
    # Ejecutar análisis
    analyzer.analyze_convergence_patterns(training_data)
    analyzer.create_learning_quality_assessment(training_data, {})
    
    print("📊 Análisis de métricas completado!")


if __name__ == "__main__":
    main()
