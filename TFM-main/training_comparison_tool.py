import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import glob

class TrainingComparisonTool:
    def __init__(self, results_dir="training_results"):
        self.results_dir = results_dir
        self.training_sessions = {}
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_training_sessions(self):
        """Cargar todas las sesiones de entrenamiento disponibles"""
        print("🔍 Buscando sesiones de entrenamiento...")
        
        # Buscar checkpoints
        checkpoint_dirs = glob.glob(os.path.join(self.results_dir, "checkpoint_ep_*"))
        
        for checkpoint_dir in checkpoint_dirs:
            session_name = os.path.basename(checkpoint_dir)
            metrics_file = os.path.join(checkpoint_dir, "metrics.pkl")
            
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'rb') as f:
                        metrics = pickle.load(f)
                    
                    self.training_sessions[session_name] = {
                        'metrics': metrics,
                        'path': checkpoint_dir,
                        'episode': int(session_name.split('_')[-1])
                    }
                    
                    print(f"✅ Cargada sesión: {session_name}")
                
                except Exception as e:
                    print(f"❌ Error cargando {session_name}: {e}")
        
        # Buscar informes de entrenamiento completos
        report_files = glob.glob(os.path.join(self.results_dir, "executive_summary.txt"))
        
        if report_files:
            print(f"📋 Encontrados {len(report_files)} informes completos")
        
        print(f"📊 Total de sesiones cargadas: {len(self.training_sessions)}")
        return len(self.training_sessions) > 0
    
    def compare_learning_curves(self):
        """Comparar curvas de aprendizaje entre sesiones"""
        if not self.training_sessions:
            print("❌ No hay sesiones para comparar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📈 Comparación de Curvas de Aprendizaje', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.training_sessions)))
        
        # 1. Recompensas promedio
        ax1 = axes[0, 0]
        ax1.set_title('💰 Recompensas Promedio')
        
        for i, (session_name, session_data) in enumerate(self.training_sessions.items()):
            metrics = session_data['metrics']
            if 'episode_rewards' in metrics and 'player_0' in metrics['episode_rewards']:
                # Calcular promedio de todos los agentes
                all_rewards = []
                for agent_rewards in metrics['episode_rewards'].values():
                    all_rewards.extend(agent_rewards)
                
                if all_rewards:
                    # Agrupar por episodios
                    num_agents = len(metrics['episode_rewards'])
                    episodes = len(metrics['episode_rewards']['player_0'])
                    avg_rewards = []
                    
                    for ep in range(episodes):
                        ep_rewards = [metrics['episode_rewards'][agent][ep] 
                                    for agent in metrics['episode_rewards'].keys() 
                                    if ep < len(metrics['episode_rewards'][agent])]
                        avg_rewards.append(np.mean(ep_rewards))
                    
                    # Suavizar
                    smoothed = self._smooth_curve(avg_rewards, window=50)
                    ax1.plot(smoothed, label=f'Ep {session_data["episode"]}', 
                            color=colors[i], alpha=0.8)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Recompensa Promedio')
        ax1.set_xlabel('Episodio')
        
        # 2. Duración de episodios
        ax2 = axes[0, 1]
        ax2.set_title('⏱️ Duración de Episodios')
        
        for i, (session_name, session_data) in enumerate(self.training_sessions.items()):
            metrics = session_data['metrics']
            if 'episode_lengths' in metrics and metrics['episode_lengths']:
                smoothed = self._smooth_curve(metrics['episode_lengths'], window=50)
                ax2.plot(smoothed, label=f'Ep {session_data["episode"]}', 
                        color=colors[i], alpha=0.8)
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('Duración (acciones)')
        ax2.set_xlabel('Episodio')
        
        # 3. Coordinación de equipo
        ax3 = axes[1, 0]
        ax3.set_title('🤝 Coordinación de Equipo')
        
        for i, (session_name, session_data) in enumerate(self.training_sessions.items()):
            metrics = session_data['metrics']
            if 'team_coordination' in metrics and metrics['team_coordination']:
                smoothed = self._smooth_curve(metrics['team_coordination'], window=50)
                ax3.plot(smoothed, label=f'Ep {session_data["episode"]}', 
                        color=colors[i], alpha=0.8)
        
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylabel('Tasa de Coordinación')
        ax3.set_xlabel('Episodio')
        
        # 4. Decisiones estratégicas
        ax4 = axes[1, 1]
        ax4.set_title('🎯 Decisiones Estratégicas')
        
        for i, (session_name, session_data) in enumerate(self.training_sessions.items()):
            metrics = session_data['metrics']
            if 'strategic_decisions' in metrics and metrics['strategic_decisions']:
                smoothed = self._smooth_curve(metrics['strategic_decisions'], window=50)
                ax4.plot(smoothed, label=f'Ep {session_data["episode"]}', 
                        color=colors[i], alpha=0.8)
        
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylabel('Tasa de Decisiones Estratégicas')
        ax4.set_xlabel('Episodio')
        
        plt.tight_layout()
        
        # Guardar comparación
        comparison_path = os.path.join(self.results_dir, f"training_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comparación guardada: {comparison_path}")
    
    def compare_final_performance(self):
        """Comparar rendimiento final entre sesiones"""
        if not self.training_sessions:
            print("❌ No hay sesiones para comparar")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('🏆 Comparación de Rendimiento Final', fontsize=16, fontweight='bold')
        
        session_names = []
        final_rewards = []
        final_coordination = []
        final_strategic = []
        
        for session_name, session_data in self.training_sessions.items():
            metrics = session_data['metrics']
            episode = session_data['episode']
            
            session_names.append(f'Ep {episode}')
            
            # Recompensa final promedio
            if 'episode_rewards' in metrics:
                final_ep_rewards = []
                for agent_rewards in metrics['episode_rewards'].values():
                    if agent_rewards:
                        final_ep_rewards.append(np.mean(agent_rewards[-50:]))  # Últimos 50 episodios
                final_rewards.append(np.mean(final_ep_rewards) if final_ep_rewards else 0)
            else:
                final_rewards.append(0)
            
            # Coordinación final
            if 'team_coordination' in metrics and metrics['team_coordination']:
                final_coordination.append(np.mean(metrics['team_coordination'][-50:]))
            else:
                final_coordination.append(0)
            
            # Decisiones estratégicas finales
            if 'strategic_decisions' in metrics and metrics['strategic_decisions']:
                final_strategic.append(np.mean(metrics['strategic_decisions'][-50:]))
            else:
                final_strategic.append(0)
        
        # 1. Recompensas finales
        bars1 = axes[0].bar(session_names, final_rewards, alpha=0.7, color='green')
        axes[0].set_title('💰 Recompensas Finales')
        axes[0].set_ylabel('Recompensa Promedio')
        axes[0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars1, final_rewards):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_rewards)*0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # 2. Coordinación final
        bars2 = axes[1].bar(session_names, final_coordination, alpha=0.7, color='blue')
        axes[1].set_title('🤝 Coordinación Final')
        axes[1].set_ylabel('Tasa de Coordinación')
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, final_coordination):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_coordination)*0.01,
                        f'{value:.2%}', ha='center', va='bottom')
        
        # 3. Decisiones estratégicas finales
        bars3 = axes[2].bar(session_names, final_strategic, alpha=0.7, color='orange')
        axes[2].set_title('🎯 Decisiones Estratégicas Finales')
        axes[2].set_ylabel('Tasa de Decisiones Estratégicas')
        axes[2].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, final_strategic):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_strategic)*0.01,
                        f'{value:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Guardar comparación
        performance_path = os.path.join(self.results_dir, f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(performance_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comparación de rendimiento guardada: {performance_path}")
        
        # Identificar mejor sesión
        if final_rewards:
            best_idx = np.argmax(final_rewards)
            best_session = session_names[best_idx]
            print(f"🏆 Mejor sesión por recompensas: {best_session} ({final_rewards[best_idx]:.3f})")
        
        if final_coordination:
            best_coord_idx = np.argmax(final_coordination)
            best_coord_session = session_names[best_coord_idx]
            print(f"🤝 Mejor sesión por coordinación: {best_coord_session} ({final_coordination[best_coord_idx]:.2%})")
    
    def analyze_learning_stability(self):
        """Analizar estabilidad del aprendizaje"""
        if not self.training_sessions:
            print("❌ No hay sesiones para analizar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 Análisis de Estabilidad del Aprendizaje', fontsize=16, fontweight='bold')
        
        # 1. Varianza de recompensas
        ax1 = axes[0, 0]
        ax1.set_title('📈 Varianza de Recompensas')
        
        for session_name, session_data in self.training_sessions.items():
            metrics = session_data['metrics']
            if 'reward_variance' in metrics and metrics['reward_variance']:
                smoothed = self._smooth_curve(metrics['reward_variance'], window=20)
                ax1.plot(smoothed, label=f'Ep {session_data["episode"]}', alpha=0.8)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Varianza')
        ax1.set_xlabel('Episodio')
        
        # 2. Progreso de aprendizaje
        ax2 = axes[0, 1]
        ax2.set_title('🚀 Progreso de Aprendizaje')
        
        for session_name, session_data in self.training_sessions.items():
            metrics = session_data['metrics']
            if 'learning_progress' in metrics and metrics['learning_progress']:
                ax2.plot(metrics['learning_progress'], label=f'Ep {session_data["episode"]}', alpha=0.8)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('Mejora en Recompensas')
        ax2.set_xlabel('Episodio')
        
        # 3. Distribución de recompensas finales
        ax3 = axes[1, 0]
        ax3.set_title('📊 Distribución de Recompensas Finales')
        
        all_final_rewards = []
        labels = []
        
        for session_name, session_data in self.training_sessions.items():
            metrics = session_data['metrics']
            if 'episode_rewards' in metrics and 'player_0' in metrics['episode_rewards']:
                # Últimos 100 episodios de todos los agentes
                final_rewards = []
                for agent_rewards in metrics['episode_rewards'].values():
                    if len(agent_rewards) >= 100:
                        final_rewards.extend(agent_rewards[-100:])
                    else:
                        final_rewards.extend(agent_rewards)
                
                if final_rewards:
                    all_final_rewards.append(final_rewards)
                    labels.append(f'Ep {session_data["episode"]}')
        
        if all_final_rewards:
            ax3.boxplot(all_final_rewards, labels=labels)
            ax3.set_ylabel('Recompensa')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Convergencia
        ax4 = axes[1, 1]
        ax4.set_title('🎯 Análisis de Convergencia')
        
        convergence_scores = []
        session_labels = []
        
        for session_name, session_data in self.training_sessions.items():
            metrics = session_data['metrics']
            if 'episode_rewards' in metrics and 'player_0' in metrics['episode_rewards']:
                # Calcular score de convergencia (estabilidad en últimos episodios)
                agent_rewards = metrics['episode_rewards']['player_0']
                if len(agent_rewards) >= 200:
                    last_100 = agent_rewards[-100:]
                    prev_100 = agent_rewards[-200:-100]
                    
                    # Diferencia en varianza (menor = más convergente)
                    var_diff = abs(np.var(last_100) - np.var(prev_100))
                    convergence_score = 1 / (1 + var_diff)  # Normalizar
                    
                    convergence_scores.append(convergence_score)
                    session_labels.append(f'Ep {session_data["episode"]}')
        
        if convergence_scores:
            bars = ax4.bar(session_labels, convergence_scores, alpha=0.7, color='purple')
            ax4.set_ylabel('Score de Convergencia')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars, convergence_scores):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Guardar análisis
        stability_path = os.path.join(self.results_dir, f"stability_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(stability_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Análisis de estabilidad guardado: {stability_path}")
    
    def generate_comparison_report(self):
        """Generar informe completo de comparación"""
        if not self.training_sessions:
            print("❌ No hay sesiones para comparar")
            return
        
        print("📊 Generando informe de comparación...")
        
        # Generar todos los gráficos
        self.compare_learning_curves()
        self.compare_final_performance()
        self.analyze_learning_stability()
        
        # Generar resumen textual
        self._generate_comparison_summary()
        
        print("✅ Informe de comparación completado")
    
    def _generate_comparison_summary(self):
        """Generar resumen textual de la comparación"""
        summary_path = os.path.join(self.results_dir, f"comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("📊 COMPARACIÓN DE SESIONES DE ENTRENAMIENTO\n")
            f.write("=" * 45 + "\n\n")
            
            f.write(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🔢 Sesiones analizadas: {len(self.training_sessions)}\n\n")
            
            # Análisis por sesión
            f.write("📈 ANÁLISIS POR SESIÓN:\n")
            f.write("-" * 22 + "\n")
            
            session_stats = []
            
            for session_name, session_data in self.training_sessions.items():
                metrics = session_data['metrics']
                episode = session_data['episode']
                
                stats = {
                    'episode': episode,
                    'final_reward': 0,
                    'coordination': 0,
                    'strategic': 0,
                    'stability': 0
                }
                
                # Calcular estadísticas
                if 'episode_rewards' in metrics:
                    final_rewards = []
                    for agent_rewards in metrics['episode_rewards'].values():
                        if agent_rewards:
                            final_rewards.append(np.mean(agent_rewards[-50:]))
                    stats['final_reward'] = np.mean(final_rewards) if final_rewards else 0
                
                if 'team_coordination' in metrics and metrics['team_coordination']:
                    stats['coordination'] = np.mean(metrics['team_coordination'][-50:])
                
                if 'strategic_decisions' in metrics and metrics['strategic_decisions']:
                    stats['strategic'] = np.mean(metrics['strategic_decisions'][-50:])
                
                # Estabilidad (inverso de varianza)
                if 'reward_variance' in metrics and metrics['reward_variance']:
                    recent_variance = np.mean(metrics['reward_variance'][-20:])
                    stats['stability'] = 1 / (1 + recent_variance)
                
                session_stats.append(stats)
                
                f.write(f"🎯 Episodio {episode}:\n")
                f.write(f"   💰 Recompensa final: {stats['final_reward']:.3f}\n")
                f.write(f"   🤝 Coordinación: {stats['coordination']:.2%}\n")
                f.write(f"   🎯 Decisiones estratégicas: {stats['strategic']:.2%}\n")
                f.write(f"   📊 Estabilidad: {stats['stability']:.3f}\n\n")
            
            # Ranking de sesiones
            f.write("🏆 RANKING DE SESIONES:\n")
            f.write("-" * 20 + "\n")
            
            # Por recompensa
            best_reward = max(session_stats, key=lambda x: x['final_reward'])
            f.write(f"🥇 Mejor recompensa: Episodio {best_reward['episode']} ({best_reward['final_reward']:.3f})\n")
            
            # Por coordinación
            best_coord = max(session_stats, key=lambda x: x['coordination'])
            f.write(f"🤝 Mejor coordinación: Episodio {best_coord['episode']} ({best_coord['coordination']:.2%})\n")
            
            # Por decisiones estratégicas
            best_strategic = max(session_stats, key=lambda x: x['strategic'])
            f.write(f"🎯 Mejor estrategia: Episodio {best_strategic['episode']} ({best_strategic['strategic']:.2%})\n")
            
            # Por estabilidad
            best_stability = max(session_stats, key=lambda x: x['stability'])
            f.write(f"📊 Más estable: Episodio {best_stability['episode']} ({best_stability['stability']:.3f})\n\n")
            
            # Recomendaciones
            f.write("💡 RECOMENDACIONES:\n")
            f.write("-" * 18 + "\n")
            
            if len(session_stats) >= 2:
                # Tendencia de mejora
                sorted_sessions = sorted(session_stats, key=lambda x: x['episode'])
                first_reward = sorted_sessions[0]['final_reward']
                last_reward = sorted_sessions[-1]['final_reward']
                
                if last_reward > first_reward * 1.1:
                    f.write("✅ Tendencia de mejora positiva detectada\n")
                    f.write("• Continuar con la configuración actual\n")
                elif last_reward < first_reward * 0.9:
                    f.write("⚠️ Tendencia de empeoramiento detectada\n")
                    f.write("• Revisar hiperparámetros\n")
                    f.write("• Considerar reducir learning rate\n")
                else:
                    f.write("📊 Rendimiento estable\n")
                    f.write("• Considerar aumentar episodios de entrenamiento\n")
            
            # Mejor modelo recomendado
            f.write(f"\n🎯 MODELO RECOMENDADO:\n")
            f.write(f"Episodio {best_reward['episode']} (mejor balance general)\n")
            f.write(f"Ubicación: {self.results_dir}/checkpoint_ep_{best_reward['episode']}/\n")
        
        print(f"📋 Resumen de comparación guardado: {summary_path}")
    
    def _smooth_curve(self, data, window=10):
        """Suavizar curva con ventana móvil"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        
        return smoothed
    
    def run_comparison_analysis(self):
        """Ejecutar análisis completo de comparación"""
        print("🔍 Iniciando análisis de comparación de entrenamientos...")
        print("=" * 55)
        
        # Cargar sesiones
        if not self.load_training_sessions():
            print("❌ No se encontraron sesiones de entrenamiento para comparar")
            print("💡 Ejecuta varios entrenamientos primero con diferentes configuraciones")
            return
        
        # Generar informe completo
        self.generate_comparison_report()
        
        print(f"\n📁 Todos los resultados guardados en: {self.results_dir}/")
        
        return self.training_sessions


if __name__ == "__main__":
    comparator = TrainingComparisonTool()
    comparator.run_comparison_analysis()
