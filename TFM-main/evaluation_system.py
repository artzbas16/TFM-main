import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json
from datetime import datetime
import sys
import time

# Añadir el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AgentEvaluationSystem:
    """Sistema de evaluación completo para agentes entrenados"""
    
    def __init__(self, num_evaluation_games=100):
        self.num_evaluation_games = num_evaluation_games
        self.evaluation_results = {
            'game_results': [],
            'agent_performance': defaultdict(list),
            'team_performance': defaultdict(list),
            'phase_analysis': defaultdict(list),
            'decision_analysis': defaultdict(int),
            'strategy_patterns': defaultdict(list)
        }
        
    def setup_evaluation_environment(self):
        """Configura el entorno para evaluación"""
        try:
            from mus_env import mus
            from marl_agent import MARLAgent
            
            # Crear entorno
            self.env = mus.env()
            
            # Cargar agentes entrenados
            self.agents = {}
            state_size = 21
            action_size = 15
            
            for i, agent_id in enumerate(self.env.possible_agents):
                team = "equipo_1" if i in [0, 2] else "equipo_2"
                self.agents[agent_id] = MARLAgent(
                    state_size=state_size,
                    action_size=action_size,
                    agent_id=i,
                    team=team
                )
                
                # Cargar modelo entrenado
                model_paths = [
                    f"trained_models/model_{agent_id}_final.pth",
                    f"model_{agent_id}.pth",
                    f"trained_models/model_{agent_id}_ep_250.pth"
                ]
                
                loaded = False
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        self.agents[agent_id].load(model_path)
                        print(f"✅ Modelo cargado para {agent_id}: {model_path}")
                        loaded = True
                        break
                
                if not loaded:
                    print(f"⚠️ No se encontró modelo para {agent_id}, usando modelo aleatorio")
                
                # Configurar para evaluación (sin exploración)
                self.agents[agent_id].epsilon = 0.0
            
            print(f"🎮 Entorno de evaluación configurado con {len(self.agents)} agentes")
            return True
            
        except Exception as e:
            print(f"❌ Error configurando evaluación: {e}")
            return False
    
    def process_observation(self, obs):
        """Procesa observación (mismo que en entrenamiento)"""
        try:
            if isinstance(obs, dict):
                cartas_flat = obs.get("cartas", np.zeros((4, 2))).flatten()
                fase_onehot = np.zeros(7)
                if "fase" in obs and obs["fase"] < 7:
                    fase_onehot[obs["fase"]] = 1
                    
                turno_onehot = np.zeros(4)
                if "turno" in obs and obs["turno"] < 4:
                    turno_onehot[obs["turno"]] = 1
                
                apuesta_norm = obs.get("apuesta_actual", 0) / 30.0
                equipo_apostador = obs.get("equipo_apostador", 0) / 2.0
                
                state = np.concatenate([
                    cartas_flat, fase_onehot, turno_onehot,
                    [apuesta_norm], [equipo_apostador]
                ])
                
                return state
            else:
                return np.zeros(21)
                
        except Exception as e:
            return np.zeros(21)
    
    def get_valid_actions(self, agent_id):
        """Obtiene acciones válidas (mismo que en entrenamiento)"""
        try:
            if self.env.fase_actual == "MUS":
                return [2, 3]
            elif self.env.fase_actual == "DESCARTE":
                return [4] + list(range(11, 15))
            elif self.env.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
                if agent_id not in self.env.jugadores_que_pueden_hablar:
                    return [0]
                
                valid = [0, 1]
                
                if hasattr(self.env, 'hay_ordago') and self.env.hay_ordago:
                    if self.env.equipo_de_jugador[agent_id] != self.env.equipo_apostador:
                        return [5, 7]
                    else:
                        return []
                
                if self.env.apuesta_actual > 0:
                    equipo_actual = self.env.equipo_de_jugador[agent_id]
                    if equipo_actual != self.env.equipo_apostador:
                        valid.extend([5, 7])
                        
                valid.append(6)
                return valid
            else:
                return [0]
                
        except Exception as e:
            return [0]
    
    def run_evaluation_game(self, game_num):
        """Ejecuta un juego de evaluación"""
        game_data = {
            'game_num': game_num,
            'winner': None,
            'final_scores': {},
            'game_length': 0,
            'phase_winners': {},
            'agent_decisions': defaultdict(list),
            'critical_moments': []
        }
        
        try:
            self.env.reset()
            max_steps = 300
            step_count = 0
            
            while step_count < max_steps and self.env.fase_actual != "RECUENTO":
                current_agent = self.env.agent_selection
                
                if current_agent not in self.env.agents or self.env.dones.get(current_agent, False):
                    break
                
                # Obtener observación y acción
                obs = self.env.observe(current_agent)
                state = self.process_observation(obs)
                valid_actions = self.get_valid_actions(current_agent)
                action = self.agents[current_agent].act(state, valid_actions)
                
                # Registrar decisión
                game_data['agent_decisions'][current_agent].append({
                    'step': step_count,
                    'phase': self.env.fase_actual,
                    'action': action,
                    'valid_actions': valid_actions.copy()
                })
                
                # Detectar momentos críticos (órdago, apuestas altas)
                if action == 6:  # Órdago
                    game_data['critical_moments'].append({
                        'step': step_count,
                        'agent': current_agent,
                        'action': 'ORDAGO',
                        'phase': self.env.fase_actual
                    })
                elif action == 1 and self.env.apuesta_actual >= 4:  # Envido con apuesta alta
                    game_data['critical_moments'].append({
                        'step': step_count,
                        'agent': current_agent,
                        'action': 'ENVIDO_ALTO',
                        'phase': self.env.fase_actual
                    })
                
                # Ejecutar acción
                self.env.step(action)
                step_count += 1
            
            # Registrar resultados finales
            game_data['game_length'] = step_count
            game_data['final_scores'] = self.env.puntos_equipos.copy()
            game_data['phase_winners'] = self.env.ganadores_fases.copy()
            
            # Determinar ganador
            if self.env.puntos_equipos["equipo_1"] > self.env.puntos_equipos["equipo_2"]:
                game_data['winner'] = "equipo_1"
            elif self.env.puntos_equipos["equipo_2"] > self.env.puntos_equipos["equipo_1"]:
                game_data['winner'] = "equipo_2"
            else:
                game_data['winner'] = "empate"
            
            return game_data
            
        except Exception as e:
            print(f"Error en juego de evaluación {game_num}: {e}")
            return game_data
    
    def run_full_evaluation(self):
        """Ejecuta evaluación completa"""
        print(f"🎯 Iniciando evaluación con {self.num_evaluation_games} juegos...")
        
        if not self.setup_evaluation_environment():
            return None
        
        start_time = time.time()
        
        for game_num in range(self.num_evaluation_games):
            if game_num % 20 == 0:
                elapsed = time.time() - start_time
                eta = elapsed * (self.num_evaluation_games - game_num) / max(game_num, 1)
                print(f"📊 Juego {game_num}/{self.num_evaluation_games} (ETA: {eta/60:.1f}min)")
            
            # Ejecutar juego
            game_data = self.run_evaluation_game(game_num)
            self.evaluation_results['game_results'].append(game_data)
            
            # Actualizar estadísticas
            self._update_evaluation_stats(game_data)
        
        print("✅ Evaluación completada!")
        return self._analyze_results()
    
    def _update_evaluation_stats(self, game_data):
        """Actualiza estadísticas de evaluación"""
        # Rendimiento por equipo
        winner = game_data['winner']
        if winner != "empate":
            self.evaluation_results['team_performance'][winner].append(1)
            other_team = "equipo_2" if winner == "equipo_1" else "equipo_1"
            self.evaluation_results['team_performance'][other_team].append(0)
        
        # Análisis por fase
        for phase, phase_winner in game_data['phase_winners'].items():
            if phase_winner:
                self.evaluation_results['phase_analysis'][phase].append(phase_winner)
        
        # Análisis de decisiones
        for agent_id, decisions in game_data['agent_decisions'].items():
            for decision in decisions:
                action_key = f"{decision['phase']}_{decision['action']}"
                self.evaluation_results['decision_analysis'][action_key] += 1
    
    def _analyze_results(self):
        """Analiza los resultados de evaluación"""
        analysis = {
            'overall_performance': {},
            'phase_performance': {},
            'strategy_analysis': {},
            'critical_decisions': {}
        }
        
        # Rendimiento general
        total_games = len(self.evaluation_results['game_results'])
        for team, wins in self.evaluation_results['team_performance'].items():
            win_rate = np.mean(wins) if wins else 0
            analysis['overall_performance'][team] = {
                'win_rate': win_rate,
                'games_won': sum(wins),
                'total_games': len(wins)
            }
        
        # Rendimiento por fase
        for phase, winners in self.evaluation_results['phase_analysis'].items():
            if winners:
                team_counts = defaultdict(int)
                for winner in winners:
                    team_counts[winner] += 1
                
                analysis['phase_performance'][phase] = dict(team_counts)
        
        # Análisis de estrategias
        total_decisions = sum(self.evaluation_results['decision_analysis'].values())
        for decision_key, count in self.evaluation_results['decision_analysis'].items():
            if count > 0:
                percentage = (count / total_decisions) * 100
                analysis['strategy_analysis'][decision_key] = {
                    'count': count,
                    'percentage': percentage
                }
        
        return analysis
    
    def generate_evaluation_report(self, analysis):
        """Genera informe completo de evaluación"""
        print("\n" + "="*60)
        print("📊 INFORME DE EVALUACIÓN DE AGENTES ENTRENADOS")
        print("="*60)
        
        # Rendimiento general
        print("\n🏆 RENDIMIENTO GENERAL:")
        for team, stats in analysis['overall_performance'].items():
            print(f"   {team}: {stats['win_rate']:.1%} ({stats['games_won']}/{stats['total_games']})")
        
        # Rendimiento por fase
        print("\n🎯 RENDIMIENTO POR FASE:")
        for phase, winners in analysis['phase_performance'].items():
            print(f"   {phase}:")
            total_phase = sum(winners.values())
            for team, wins in winners.items():
                percentage = (wins / total_phase) * 100 if total_phase > 0 else 0
                print(f"      {team}: {percentage:.1f}% ({wins}/{total_phase})")
        
        # Estrategias más comunes
        print("\n🧠 ESTRATEGIAS MÁS COMUNES:")
        sorted_strategies = sorted(
            analysis['strategy_analysis'].items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )[:10]
        
        for strategy, stats in sorted_strategies:
            print(f"   {strategy}: {stats['percentage']:.1f}% ({stats['count']} veces)")
        
        return analysis
    
    def create_evaluation_visualizations(self, analysis):
        """Crea visualizaciones de la evaluación"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📊 Evaluación de Agentes Entrenados - Mus MARL', fontsize=16)
        
        # 1. Tasa de victoria por equipo
        ax = axes[0, 0]
        teams = list(analysis['overall_performance'].keys())
        win_rates = [analysis['overall_performance'][team]['win_rate'] for team in teams]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(teams, win_rates, color=colors, alpha=0.8)
        ax.set_title('🏆 Tasa de Victoria por Equipo')
        ax.set_ylabel('Tasa de Victoria')
        ax.set_ylim(0, 1)
        
        # Añadir valores en las barras
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. Rendimiento por fase
        ax = axes[0, 1]
        phases = list(analysis['phase_performance'].keys())
        if phases:
            team1_wins = []
            team2_wins = []
            
            for phase in phases:
                phase_data = analysis['phase_performance'][phase]
                total = sum(phase_data.values())
                team1_wins.append(phase_data.get('equipo_1', 0) / total if total > 0 else 0)
                team2_wins.append(phase_data.get('equipo_2', 0) / total if total > 0 else 0)
            
            x = np.arange(len(phases))
            width = 0.35
            
            ax.bar(x - width/2, team1_wins, width, label='Equipo 1', color='#FF6B6B', alpha=0.8)
            ax.bar(x + width/2, team2_wins, width, label='Equipo 2', color='#4ECDC4', alpha=0.8)
            
            ax.set_title('🎯 Rendimiento por Fase')
            ax.set_ylabel('Tasa de Victoria')
            ax.set_xticks(x)
            ax.set_xticklabels(phases)
            ax.legend()
            ax.set_ylim(0, 1)
        
        # 3. Distribución de duración de juegos
        ax = axes[1, 0]
        game_lengths = [game['game_length'] for game in self.evaluation_results['game_results']]
        if game_lengths:
            ax.hist(game_lengths, bins=20, alpha=0.7, color='#96CEB4', edgecolor='black')
            ax.set_title('⏱️ Distribución de Duración de Juegos')
            ax.set_xlabel('Pasos por Juego')
            ax.set_ylabel('Frecuencia')
            ax.axvline(np.mean(game_lengths), color='red', linestyle='--', 
                      label=f'Promedio: {np.mean(game_lengths):.1f}')
            ax.legend()
        
        # 4. Top estrategias
        ax = axes[1, 1]
        top_strategies = sorted(
            analysis['strategy_analysis'].items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )[:8]
        
        if top_strategies:
            strategies = [s[0].replace('_', '\n') for s, _ in top_strategies]
            percentages = [data['percentage'] for _, data in top_strategies]
            
            bars = ax.barh(strategies, percentages, color='#FECA57', alpha=0.8)
            ax.set_title('🧠 Estrategias Más Utilizadas')
            ax.set_xlabel('Porcentaje de Uso')
            
            # Añadir valores
            for bar, pct in zip(bars, percentages):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                       f'{pct:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        
        # Guardar visualización
        os.makedirs("evaluation_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results/evaluation_report_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizaciones guardadas: {filename}")
        
        # Guardar datos de evaluación
        data_filename = f"evaluation_results/evaluation_data_{timestamp}.json"
        with open(data_filename, 'w') as f:
            # Convertir numpy arrays a listas para JSON
            json_data = {}
            for key, value in analysis.items():
                if isinstance(value, dict):
                    json_data[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            json_data[key][k] = v.tolist()
                        elif isinstance(v, dict):
                            json_data[key][k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) 
                                               for kk, vv in v.items()}
                        else:
                            json_data[key][k] = v
                else:
                    json_data[key] = value
            
            json.dump(json_data, f, indent=2)
        
        print(f"💾 Datos de evaluación guardados: {data_filename}")


def main():
    """Función principal para ejecutar la evaluación"""
    print("🎯 SISTEMA DE EVALUACIÓN DE AGENTES ENTRENADOS")
    print("=" * 50)
    
    # Crear sistema de evaluación
    evaluation_system = AgentEvaluationSystem(num_evaluation_games=200)
    
    # Ejecutar evaluación completa
    analysis = evaluation_system.run_full_evaluation()
    
    if analysis:
        # Generar informe
        evaluation_system.generate_evaluation_report(analysis)
        
        # Crear visualizaciones
        evaluation_system.create_evaluation_visualizations(analysis)
        
        print("\n🎉 ¡Evaluación completada exitosamente!")
        print("📊 Revisa los gráficos y resultados generados en 'evaluation_results/'")
    else:
        print("❌ Error en la evaluación")


if __name__ == "__main__":
    main()
