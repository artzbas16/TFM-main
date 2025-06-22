import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import json
from datetime import datetime
import torch

# Importar módulos existentes
from mus_env import mus
from marl_agent import MARLAgent

class QuickEvaluationTool:
    def __init__(self, results_dir="training_results"):
        self.results_dir = results_dir
        self.evaluation_results = {}
        
        # Configurar estilo de gráficos
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_trained_agents(self, model_path="best_models"):
        """Cargar agentes entrenados"""
        model_dir = os.path.join(self.results_dir, model_path)
        
        if not os.path.exists(model_dir):
            print(f"❌ No se encontraron modelos en {model_dir}")
            return None
        
        # Crear entorno para obtener dimensiones
        env = mus.env()
        
        # Crear agentes
        agents = {}
        for agent_id in env.possible_agents:
            agent = MARLAgent(
                state_size=env.observation_space(agent_id).shape[0],
                action_size=env.action_space(agent_id).n,
                agent_id=agent_id
            )
            
            # Cargar modelo entrenado
            model_file = os.path.join(model_dir, f"{agent_id}_best.pth")
            if os.path.exists(model_file):
                agent.q_network.load_state_dict(torch.load(model_file))
                agent.epsilon = 0.0  # Sin exploración para evaluación
                agents[agent_id] = agent
                print(f"✅ Modelo cargado para {agent_id}")
            else:
                print(f"⚠️ No se encontró modelo para {agent_id}")
        
        return agents if agents else None
    
    def create_random_agents(self):
        """Crear agentes que actúan aleatoriamente para comparación"""
        env = mus.env()
        
        class RandomAgent:
            def __init__(self, action_size):
                self.action_size = action_size
            
            def act(self, observation):
                return np.random.randint(0, self.action_size)
        
        random_agents = {}
        for agent_id in env.possible_agents:
            random_agents[agent_id] = RandomAgent(env.action_space(agent_id).n)
        
        return random_agents
    
    def evaluate_agents_vs_random(self, trained_agents, num_games=100):
        """Evaluar agentes entrenados contra agentes aleatorios"""
        print(f"🎯 Evaluando agentes entrenados vs aleatorios ({num_games} juegos)...")
        
        env = mus.env()
        random_agents = self.create_random_agents()
        
        results = {
            'trained_wins': 0,
            'random_wins': 0,
            'draws': 0,
            'game_lengths': [],
            'trained_scores': [],
            'random_scores': [],
            'strategic_actions': 0,
            'total_actions': 0
        }
        
        for game in range(num_games):
            env.reset()
            game_length = 0
            strategic_actions = 0
            
            # Alternar entre agentes entrenados y aleatorios
            # Equipo 0: entrenados, Equipo 1: aleatorios
            current_agents = {
                'player_0': trained_agents['player_0'],
                'player_1': random_agents['player_1'],
                'player_2': trained_agents['player_2'],
                'player_3': random_agents['player_3']
            }
            
            for agent_id in env.agent_iter():
                if env.terminations[agent_id] or env.truncations[agent_id]:
                    continue
                
                observation, reward, termination, truncation, info = env.last()
                game_length += 1
                
                if termination or truncation:
                    action = None
                else:
                    action = current_agents[agent_id].act(observation)
                    
                    # Contar acciones estratégicas de agentes entrenados
                    if agent_id in ['player_0', 'player_2']:
                        if self._is_strategic_action(action, info):
                            strategic_actions += 1
                
                env.step(action)
            
            # Analizar resultado
            final_info = info
            scores = final_info.get('scores', [0, 0])
            
            results['game_lengths'].append(game_length)
            results['trained_scores'].append(scores[0])  # Equipo 0 (entrenados)
            results['random_scores'].append(scores[1])   # Equipo 1 (aleatorios)
            results['strategic_actions'] += strategic_actions
            results['total_actions'] += game_length
            
            if scores[0] > scores[1]:
                results['trained_wins'] += 1
            elif scores[1] > scores[0]:
                results['random_wins'] += 1
            else:
                results['draws'] += 1
        
        # Calcular métricas finales
        results['trained_win_rate'] = results['trained_wins'] / num_games
        results['random_win_rate'] = results['random_wins'] / num_games
        results['draw_rate'] = results['draws'] / num_games
        results['avg_game_length'] = np.mean(results['game_lengths'])
        results['strategic_rate'] = results['strategic_actions'] / max(results['total_actions'], 1)
        results['avg_trained_score'] = np.mean(results['trained_scores'])
        results['avg_random_score'] = np.mean(results['random_scores'])
        
        self.evaluation_results['vs_random'] = results
        return results
    
    def evaluate_trained_vs_trained(self, trained_agents, num_games=50):
        """Evaluar agentes entrenados entre sí"""
        print(f"🤖 Evaluando agentes entrenados entre sí ({num_games} juegos)...")
        
        env = mus.env()
        
        results = {
            'team_0_wins': 0,
            'team_1_wins': 0,
            'draws': 0,
            'game_lengths': [],
            'team_scores': {'team_0': [], 'team_1': []},
            'coordination_events': 0,
            'strategic_decisions': 0,
            'total_actions': 0
        }
        
        for game in range(num_games):
            env.reset()
            game_length = 0
            coordination_events = 0
            strategic_decisions = 0
            
            for agent_id in env.agent_iter():
                if env.terminations[agent_id] or env.truncations[agent_id]:
                    continue
                
                observation, reward, termination, truncation, info = env.last()
                game_length += 1
                
                if termination or truncation:
                    action = None
                else:
                    action = trained_agents[agent_id].act(observation)
                    
                    if self._is_strategic_action(action, info):
                        strategic_decisions += 1
                    
                    if self._indicates_coordination(agent_id, action, info):
                        coordination_events += 1
                
                env.step(action)
            
            # Analizar resultado
            final_info = info
            scores = final_info.get('scores', [0, 0])
            
            results['game_lengths'].append(game_length)
            results['team_scores']['team_0'].append(scores[0])
            results['team_scores']['team_1'].append(scores[1])
            results['coordination_events'] += coordination_events
            results['strategic_decisions'] += strategic_decisions
            results['total_actions'] += game_length
            
            if scores[0] > scores[1]:
                results['team_0_wins'] += 1
            elif scores[1] > scores[0]:
                results['team_1_wins'] += 1
            else:
                results['draws'] += 1
        
        # Calcular métricas finales
        results['team_0_win_rate'] = results['team_0_wins'] / num_games
        results['team_1_win_rate'] = results['team_1_wins'] / num_games
        results['draw_rate'] = results['draws'] / num_games
        results['avg_game_length'] = np.mean(results['game_lengths'])
        results['coordination_rate'] = results['coordination_events'] / max(results['total_actions'], 1)
        results['strategic_rate'] = results['strategic_decisions'] / max(results['total_actions'], 1)
        results['balance_score'] = abs(results['team_0_win_rate'] - 0.5)  # Qué tan lejos del equilibrio
        
        self.evaluation_results['trained_vs_trained'] = results
        return results
    
    def _is_strategic_action(self, action, info):
        """Determinar si una acción es estratégica"""
        phase = info.get('phase', 'unknown')
        
        strategic_actions = {
            'mus': [2, 3],  # mus, no mus
            'grande': [1, 6, 7],  # envido, órdago, quiero
            'chica': [1, 6, 7],
            'pares': [1, 6, 7],
            'juego': [1, 6, 7],
            'descarte': list(range(11, 15))  # seleccionar cartas
        }
        
        return action in strategic_actions.get(phase, [])
    
    def _indicates_coordination(self, agent_id, action, info):
        """Detectar indicios de coordinación (simplificado)"""
        phase = info.get('phase', 'unknown')
        
        # Acciones que podrían indicar coordinación en fases de apuesta
        if phase in ['grande', 'chica', 'pares', 'juego']:
            return action in [1, 6, 7]  # envido, órdago, quiero
        
        return False
    
    def generate_evaluation_report(self):
        """Generar informe visual de evaluación"""
        if not self.evaluation_results:
            print("❌ No hay resultados de evaluación para mostrar")
            return
        
        # Crear figura con múltiples subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎯 Informe de Evaluación de Agentes MARL', fontsize=16, fontweight='bold')
        
        # 1. Comparación de tasas de victoria
        if 'vs_random' in self.evaluation_results:
            self._plot_win_rates_comparison(axes[0, 0])
        
        # 2. Distribución de puntuaciones
        if 'vs_random' in self.evaluation_results:
            self._plot_score_distribution(axes[0, 1])
        
        # 3. Duración de juegos
        self._plot_game_lengths(axes[0, 2])
        
        # 4. Métricas de calidad
        self._plot_quality_metrics(axes[1, 0])
        
        # 5. Análisis de coordinación
        if 'trained_vs_trained' in self.evaluation_results:
            self._plot_coordination_analysis(axes[1, 1])
        
        # 6. Resumen de rendimiento
        self._plot_performance_summary(axes[1, 2])
        
        plt.tight_layout()
        
        # Guardar informe
        report_path = os.path.join(self.results_dir, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Informe de evaluación guardado: {report_path}")
        
        # Generar resumen textual
        self._generate_text_summary()
    
    def _plot_win_rates_comparison(self, ax):
        """Gráfico de comparación de tasas de victoria"""
        ax.set_title('🏆 Tasas de Victoria: Entrenados vs Aleatorios', fontweight='bold')
        
        results = self.evaluation_results['vs_random']
        
        categories = ['Entrenados', 'Aleatorios', 'Empates']
        values = [results['trained_win_rate'], results['random_win_rate'], results['draw_rate']]
        colors = ['green', 'red', 'gray']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Tasa de Victoria')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    def _plot_score_distribution(self, ax):
        """Distribución de puntuaciones"""
        ax.set_title('📊 Distribución de Puntuaciones', fontweight='bold')
        
        results = self.evaluation_results['vs_random']
        
        ax.hist(results['trained_scores'], alpha=0.7, label='Entrenados', bins=15, color='green')
        ax.hist(results['random_scores'], alpha=0.7, label='Aleatorios', bins=15, color='red')
        
        ax.axvline(np.mean(results['trained_scores']), color='green', linestyle='--', 
                  label=f'Media Entrenados: {np.mean(results["trained_scores"]):.1f}')
        ax.axvline(np.mean(results['random_scores']), color='red', linestyle='--',
                  label=f'Media Aleatorios: {np.mean(results["random_scores"]):.1f}')
        
        ax.legend()
        ax.set_xlabel('Puntuación')
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3)
    
    def _plot_game_lengths(self, ax):
        """Duración de los juegos"""
        ax.set_title('⏱️ Duración de los Juegos', fontweight='bold')
        
        all_lengths = []
        labels = []
        
        if 'vs_random' in self.evaluation_results:
            all_lengths.append(self.evaluation_results['vs_random']['game_lengths'])
            labels.append('vs Aleatorios')
        
        if 'trained_vs_trained' in self.evaluation_results:
            all_lengths.append(self.evaluation_results['trained_vs_trained']['game_lengths'])
            labels.append('Entrenados vs Entrenados')
        
        if all_lengths:
            ax.boxplot(all_lengths, labels=labels)
            ax.set_ylabel('Duración (acciones)')
            ax.grid(True, alpha=0.3)
    
    def _plot_quality_metrics(self, ax):
        """Métricas de calidad de decisiones"""
        ax.set_title('🎯 Calidad de Decisiones', fontweight='bold')
        
        metrics = []
        values = []
        
        if 'vs_random' in self.evaluation_results:
            metrics.append('Decisiones\nEstratégicas')
            values.append(self.evaluation_results['vs_random']['strategic_rate'])
        
        if 'trained_vs_trained' in self.evaluation_results:
            metrics.extend(['Coordinación\nde Equipo', 'Balance\nCompetitivo'])
            values.extend([
                self.evaluation_results['trained_vs_trained']['coordination_rate'],
                1 - self.evaluation_results['trained_vs_trained']['balance_score']  # Invertir para que mayor sea mejor
            ])
        
        if metrics:
            bars = ax.bar(metrics, values, alpha=0.7, color=['blue', 'orange', 'purple'][:len(metrics)])
            
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Tasa')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
    
    def _plot_coordination_analysis(self, ax):
        """Análisis de coordinación entre equipos"""
        ax.set_title('🤝 Análisis de Coordinación', fontweight='bold')
        
        if 'trained_vs_trained' not in self.evaluation_results:
            ax.text(0.5, 0.5, 'No hay datos\nde coordinación', ha='center', va='center', transform=ax.transAxes)
            return
        
        results = self.evaluation_results['trained_vs_trained']
        
        # Crear gráfico de barras con métricas de coordinación
        metrics = ['Eventos de\nCoordinación', 'Decisiones\nEstratégicas', 'Balance\nCompetitivo']
        values = [
            results['coordination_rate'],
            results['strategic_rate'],
            1 - results['balance_score']
        ]
        
        bars = ax.bar(metrics, values, alpha=0.7, color=['green', 'blue', 'orange'])
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Tasa')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_summary(self, ax):
        """Resumen de rendimiento general"""
        ax.set_title('📈 Resumen de Rendimiento', fontweight='bold')
        
        # Crear radar chart o gráfico de barras con métricas clave
        performance_metrics = {}
        
        if 'vs_random' in self.evaluation_results:
            vs_random = self.evaluation_results['vs_random']
            performance_metrics['Superioridad vs Aleatorios'] = vs_random['trained_win_rate']
            performance_metrics['Decisiones Estratégicas'] = vs_random['strategic_rate']
        
        if 'trained_vs_trained' in self.evaluation_results:
            vs_trained = self.evaluation_results['trained_vs_trained']
            performance_metrics['Coordinación'] = vs_trained['coordination_rate']
            performance_metrics['Balance Competitivo'] = 1 - vs_trained['balance_score']
        
        if performance_metrics:
            metrics = list(performance_metrics.keys())
            values = list(performance_metrics.values())
            
            # Crear gráfico de barras horizontal
            bars = ax.barh(metrics, values, alpha=0.7, color=['green', 'blue', 'orange', 'purple'][:len(metrics)])
            
            for bar, value in zip(bars, values):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.2%}', ha='left', va='center', fontweight='bold')
            
            ax.set_xlabel('Puntuación')
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hay datos\nsuficientes', ha='center', va='center', transform=ax.transAxes)
    
    def _generate_text_summary(self):
        """Generar resumen textual de la evaluación"""
        summary_path = os.path.join(self.results_dir, "evaluation_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("🎯 RESUMEN DE EVALUACIÓN - AGENTES MARL MUS GAME\n")
            f.write("=" * 55 + "\n\n")
            
            f.write(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Evaluación vs agentes aleatorios
            if 'vs_random' in self.evaluation_results:
                results = self.evaluation_results['vs_random']
                f.write("🤖 EVALUACIÓN VS AGENTES ALEATORIOS:\n")
                f.write("-" * 35 + "\n")
                f.write(f"🏆 Tasa de victoria entrenados: {results['trained_win_rate']:.1%}\n")
                f.write(f"🎲 Tasa de victoria aleatorios: {results['random_win_rate']:.1%}\n")
                f.write(f"🤝 Empates: {results['draw_rate']:.1%}\n")
                f.write(f"📊 Puntuación promedio entrenados: {results['avg_trained_score']:.1f}\n")
                f.write(f"📊 Puntuación promedio aleatorios: {results['avg_random_score']:.1f}\n")
                f.write(f"🎯 Tasa de decisiones estratégicas: {results['strategic_rate']:.1%}\n")
                f.write(f"⏱️ Duración promedio de juego: {results['avg_game_length']:.1f} acciones\n\n")
                
                # Análisis de resultados
                if results['trained_win_rate'] > 0.6:
                    f.write("✅ EXCELENTE: Los agentes superan claramente a los aleatorios\n")
                elif results['trained_win_rate'] > 0.55:
                    f.write("✅ BUENO: Los agentes muestran ventaja sobre los aleatorios\n")
                elif results['trained_win_rate'] > 0.45:
                    f.write("⚠️ REGULAR: Los agentes no muestran ventaja clara\n")
                else:
                    f.write("❌ MALO: Los agentes no han aprendido efectivamente\n")
                f.write("\n")
            
            # Evaluación entre agentes entrenados
            if 'trained_vs_trained' in self.evaluation_results:
                results = self.evaluation_results['trained_vs_trained']
                f.write("🤖 EVALUACIÓN ENTRE AGENTES ENTRENADOS:\n")
                f.write("-" * 38 + "\n")
                f.write(f"⚖️ Tasa de victoria Equipo 0: {results['team_0_win_rate']:.1%}\n")
                f.write(f"⚖️ Tasa de victoria Equipo 1: {results['team_1_win_rate']:.1%}\n")
                f.write(f"🤝 Empates: {results['draw_rate']:.1%}\n")
                f.write(f"🤝 Tasa de coordinación: {results['coordination_rate']:.1%}\n")
                f.write(f"🎯 Tasa de decisiones estratégicas: {results['strategic_rate']:.1%}\n")
                f.write(f"📊 Puntuación de balance: {results['balance_score']:.3f}\n")
                f.write(f"⏱️ Duración promedio de juego: {results['avg_game_length']:.1f} acciones\n\n")
                
                # Análisis de balance
                if results['balance_score'] < 0.1:
                    f.write("✅ EXCELENTE: Equipos muy balanceados\n")
                elif results['balance_score'] < 0.2:
                    f.write("✅ BUENO: Equipos razonablemente balanceados\n")
                else:
                    f.write("⚠️ DESBALANCEADO: Un equipo domina al otro\n")
                f.write("\n")
            
            # Conclusiones generales
            f.write("🎯 CONCLUSIONES GENERALES:\n")
            f.write("-" * 25 + "\n")
            
            learning_success = False
            if 'vs_random' in self.evaluation_results:
                if self.evaluation_results['vs_random']['trained_win_rate'] > 0.55:
                    learning_success = True
                    f.write("✅ Los agentes han aprendido estrategias efectivas\n")
                else:
                    f.write("❌ Los agentes necesitan más entrenamiento\n")
            
            if 'trained_vs_trained' in self.evaluation_results:
                coord_rate = self.evaluation_results['trained_vs_trained']['coordination_rate']
                if coord_rate > 0.3:
                    f.write("✅ Buena coordinación entre compañeros de equipo\n")
                elif coord_rate > 0.1:
                    f.write("⚠️ Coordinación moderada entre compañeros\n")
                else:
                    f.write("❌ Poca coordinación entre compañeros\n")
            
            # Recomendaciones
            f.write("\n💡 RECOMENDACIONES:\n")
            f.write("-" * 18 + "\n")
            
            if not learning_success:
                f.write("• Aumentar el número de episodios de entrenamiento\n")
                f.write("• Ajustar hiperparámetros (learning rate, epsilon decay)\n")
                f.write("• Revisar sistema de recompensas\n")
            
            if 'trained_vs_trained' in self.evaluation_results:
                if self.evaluation_results['trained_vs_trained']['coordination_rate'] < 0.2:
                    f.write("• Implementar recompensas específicas para coordinación\n")
                    f.write("• Considerar comunicación entre agentes del mismo equipo\n")
            
            f.write("\n🎮 Para entrenar más, ejecutar: python improved_training_system.py\n")
        
        print(f"📋 Resumen de evaluación guardado: {summary_path}")
    
    def run_quick_evaluation(self):
        """Ejecutar evaluación rápida completa"""
        print("🚀 Iniciando evaluación rápida de agentes MARL...")
        print("=" * 50)
        
        # Cargar agentes entrenados
        trained_agents = self.load_trained_agents()
        
        if not trained_agents:
            print("❌ No se pudieron cargar agentes entrenados")
            print("💡 Ejecuta primero el entrenamiento con: python improved_training_system.py")
            return
        
        print(f"✅ Agentes cargados: {list(trained_agents.keys())}")
        
        # Evaluación vs agentes aleatorios
        print("\n🎲 Evaluando contra agentes aleatorios...")
        vs_random_results = self.evaluate_agents_vs_random(trained_agents, num_games=100)
        
        print(f"📊 Resultados vs Aleatorios:")
        print(f"   🏆 Tasa de victoria: {vs_random_results['trained_win_rate']:.1%}")
        print(f"   🎯 Decisiones estratégicas: {vs_random_results['strategic_rate']:.1%}")
        print(f"   📈 Puntuación promedio: {vs_random_results['avg_trained_score']:.1f}")
        
        # Evaluación entre agentes entrenados
        print("\n🤖 Evaluando agentes entrenados entre sí...")
        vs_trained_results = self.evaluate_trained_vs_trained(trained_agents, num_games=50)
        
        print(f"📊 Resultados Entrenados vs Entrenados:")
        print(f"   ⚖️ Balance competitivo: {1-vs_trained_results['balance_score']:.1%}")
        print(f"   🤝 Coordinación de equipo: {vs_trained_results['coordination_rate']:.1%}")
        print(f"   🎯 Decisiones estratégicas: {vs_trained_results['strategic_rate']:.1%}")
        
        # Generar informe visual
        print("\n📊 Generando informe visual...")
        self.generate_evaluation_report()
        
        # Evaluación final
        print("\n🎯 EVALUACIÓN FINAL:")
        if vs_random_results['trained_win_rate'] > 0.6:
            print("✅ EXCELENTE: Los agentes han aprendido efectivamente")
        elif vs_random_results['trained_win_rate'] > 0.55:
            print("✅ BUENO: Los agentes muestran aprendizaje")
        elif vs_random_results['trained_win_rate'] > 0.45:
            print("⚠️ REGULAR: Aprendizaje limitado")
        else:
            print("❌ MALO: Los agentes necesitan más entrenamiento")
        
        print(f"\n📁 Resultados guardados en: {self.results_dir}/")
        
        return self.evaluation_results


if __name__ == "__main__":
    evaluator = QuickEvaluationTool()
    evaluator.run_quick_evaluation()
