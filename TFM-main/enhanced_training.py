import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
from datetime import datetime
import sys
import threading

# Añadir el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_visualizer import TrainingVisualizer

class EnhancedTrainingSystem:
    """Sistema de entrenamiento mejorado con visualización en tiempo real"""
    
    def __init__(self, max_episodes=500, save_interval=50):
        self.max_episodes = max_episodes
        self.save_interval = save_interval
        self.visualizer = TrainingVisualizer(update_interval=5)
        
        # Métricas de entrenamiento
        self.training_metrics = {
            'start_time': None,
            'episode_times': [],
            'convergence_episode': None,
            'best_performance': defaultdict(float)
        }
        
    def setup_environment_and_agents(self):
        """Configura el entorno y los agentes"""
        try:
            # Importar módulos necesarios
            from mus_env import mus
            from marl_agent import MARLAgent
            
            # Crear entorno
            self.env = mus.env()
            self.env.reset()
            
            # Crear agentes MARL
            state_size = 21
            action_size = 15
            self.agents = {}
            
            for i, agent_id in enumerate(self.env.possible_agents):
                team = "equipo_1" if i in [0, 2] else "equipo_2"
                self.agents[agent_id] = MARLAgent(
                    state_size=state_size,
                    action_size=action_size,
                    agent_id=i,
                    team=team
                )
                
                # Cargar modelo preentrenado si existe
                model_path = f"model_{agent_id}.pth"
                if os.path.exists(model_path):
                    self.agents[agent_id].load(model_path)
                    print(f"✅ Modelo cargado para {agent_id}")
                    
            print(f"🎮 Entorno y {len(self.agents)} agentes configurados")
            return True
            
        except Exception as e:
            print(f"❌ Error configurando entorno: {e}")
            return False
    
    def process_observation(self, obs):
        """Procesa la observación del entorno"""
        try:
            if isinstance(obs, dict):
                # Extraer información básica
                cartas_flat = obs.get("cartas", np.zeros((4, 2))).flatten()
                fase_onehot = np.zeros(7)
                if "fase" in obs and obs["fase"] < 7:
                    fase_onehot[obs["fase"]] = 1
                    
                turno_onehot = np.zeros(4)
                if "turno" in obs and obs["turno"] < 4:
                    turno_onehot[obs["turno"]] = 1
                
                # Información adicional
                apuesta_norm = obs.get("apuesta_actual", 0) / 30.0
                equipo_apostador = obs.get("equipo_apostador", 0) / 2.0
                
                # Concatenar features
                state = np.concatenate([
                    cartas_flat,
                    fase_onehot,
                    turno_onehot,
                    [apuesta_norm],
                    [equipo_apostador]
                ])
                
                return state
            else:
                return np.zeros(21)
                
        except Exception as e:
            print(f"Error procesando observación: {e}")
            return np.zeros(21)
    
    def get_valid_actions(self, agent_id):
        """Obtiene acciones válidas para un agente"""
        try:
            if self.env.fase_actual == "MUS":
                return [2, 3]  # Mus o No Mus
            elif self.env.fase_actual == "DESCARTE":
                return [4] + list(range(11, 15))  # OK + selección cartas
            elif self.env.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
                if agent_id not in self.env.jugadores_que_pueden_hablar:
                    return [0]  # Solo paso
                
                valid = [0, 1]  # Paso, Envido
                
                if hasattr(self.env, 'hay_ordago') and self.env.hay_ordago:
                    if self.env.equipo_de_jugador[agent_id] != self.env.equipo_apostador:
                        return [5, 7]  # No quiero, Quiero
                    else:
                        return []
                
                if self.env.apuesta_actual > 0:
                    equipo_actual = self.env.equipo_de_jugador[agent_id]
                    if equipo_actual != self.env.equipo_apostador:
                        valid.extend([5, 7])  # No quiero, Quiero
                        
                valid.append(6)  # Órdago
                return valid
            else:
                return [0]
                
        except Exception as e:
            print(f"Error obteniendo acciones válidas: {e}")
            return [0]
    
    def calculate_rewards(self):
        """Calcula recompensas para todos los agentes"""
        rewards = {}
        
        try:
            for agent_id in self.env.agents:
                equipo = self.env.equipo_de_jugador[agent_id]
                puntos_equipo = self.env.puntos_equipos[equipo]
                puntos_oponente = self.env.puntos_equipos["equipo_2" if equipo == "equipo_1" else "equipo_1"]
                
                # Recompensa base por diferencia de puntos
                reward = (puntos_equipo - puntos_oponente) * 0.1
                
                # Recompensa por participación activa
                if agent_id in self.env.jugadores_que_pueden_hablar:
                    reward += 0.5
                
                # Recompensa por ganar la partida
                if self.env.fase_actual == "RECUENTO":
                    if puntos_equipo >= 30:
                        reward += 20
                    elif puntos_oponente >= 30:
                        reward -= 10
                
                rewards[agent_id] = reward
                
        except Exception as e:
            print(f"Error calculando recompensas: {e}")
            rewards = {agent_id: 0 for agent_id in self.env.agents}
            
        return rewards
    
    def run_training_episode(self, episode_num):
        """Ejecuta un episodio de entrenamiento"""
        episode_start_time = time.time()
        episode_data = {
            'episode': episode_num,
            'rewards': defaultdict(float),
            'length': 0,
            'winner': None,
            'epsilons': {},
            'losses': {},
            'phase_performance': defaultdict(int)
        }
        
        try:
            # Reset del entorno
            self.env.reset()
            max_steps = 200
            step_count = 0
            
            # Estados previos para el aprendizaje
            prev_states = {}
            prev_actions = {}
            
            while step_count < max_steps and self.env.fase_actual != "RECUENTO":
                current_agent = self.env.agent_selection
                
                if current_agent not in self.env.agents or self.env.dones.get(current_agent, False):
                    break
                
                # Obtener observación y procesarla
                obs = self.env.observe(current_agent)
                current_state = self.process_observation(obs)
                
                # Si hay estado previo, guardar experiencia
                if current_agent in prev_states:
                    prev_state = prev_states[current_agent]
                    prev_action = prev_actions[current_agent]
                    rewards = self.calculate_rewards()
                    reward = rewards.get(current_agent, 0.0)
                    done = self.env.dones.get(current_agent, False)
                    
                    self.agents[current_agent].remember(
                        prev_state, prev_action, reward, current_state, done
                    )
                    
                    # Entrenar el agente y capturar pérdida
                    try:
                        loss = self.agents[current_agent].replay()
                        # Algunos agentes pueden devolver la pérdida, otros no
                        if loss is not None and not np.isnan(loss) and loss > 0:
                            episode_data['losses'][current_agent] = float(loss)
                    except Exception as e:
                        # Si hay error en replay, continuar sin registrar pérdida
                        pass
            
            # Obtener acciones válidas y tomar decisión
            valid_actions = self.get_valid_actions(current_agent)
            if len(valid_actions) == 0:
                valid_actions = [0]  # Acción por defecto
                
            action = self.agents[current_agent].act(current_state, valid_actions)
            
            # Registrar epsilon
            epsilon = getattr(self.agents[current_agent], 'epsilon', 0.0)
            episode_data['epsilons'][current_agent] = float(epsilon)
            
            # Ejecutar acción
            self.env.step(action)
            
            # Actualizar estados previos
            prev_states[current_agent] = current_state
            prev_actions[current_agent] = action
            
            step_count += 1
            episode_data['length'] = step_count
            
            # Pequeña pausa para visualización
            time.sleep(0.001)  # Reducir pausa para mejor rendimiento
        
            # Calcular recompensas finales
            final_rewards = self.calculate_rewards()
            for agent_id, reward in final_rewards.items():
                episode_data['rewards'][agent_id] = float(reward)
            
            # Determinar ganador
            equipo1_puntos = self.env.puntos_equipos.get("equipo_1", 0)
            equipo2_puntos = self.env.puntos_equipos.get("equipo_2", 0)
            
            if equipo1_puntos > equipo2_puntos:
                episode_data['winner'] = "equipo_1"
            elif equipo2_puntos > equipo1_puntos:
                episode_data['winner'] = "equipo_2"
            else:
                episode_data['winner'] = "empate"
            
            # Registrar rendimiento por fase
            for fase, ganador in self.env.ganadores_fases.items():
                if ganador is not None:
                    episode_data['phase_performance'][fase] = 1
                else:
                    episode_data['phase_performance'][fase] = 0
            
            episode_time = time.time() - episode_start_time
            self.training_metrics['episode_times'].append(episode_time)
            
            return episode_data
        
        except Exception as e:
            print(f"Error en episodio {episode_num}: {e}")
            # Devolver datos parciales en caso de error
            return episode_data
        
    def train_with_visualization(self):
        """Entrenamiento principal con visualización"""
        print("🚀 Iniciando entrenamiento mejorado con visualización...")
        
        # Configurar entorno y agentes
        if not self.setup_environment_and_agents():
            return None
        
        # Iniciar visualización
        self.visualizer.start_training_visualization()
        self.training_metrics['start_time'] = time.time()
        
        try:
            print(f"🎯 Entrenando por {self.max_episodes} episodios...")
            
            for episode in range(self.max_episodes):
                # Mostrar progreso cada 25 episodios
                if episode % 25 == 0:
                    elapsed_time = time.time() - self.training_metrics['start_time']
                    avg_episode_time = np.mean(self.training_metrics['episode_times'][-10:]) if self.training_metrics['episode_times'] else 0
                    eta = avg_episode_time * (self.max_episodes - episode)
                    
                    print(f"📈 Episodio {episode}/{self.max_episodes}")
                    print(f"⏱️ Tiempo transcurrido: {elapsed_time/60:.1f}min")
                    print(f"🔮 ETA: {eta/60:.1f}min")
                    print("-" * 40)
                
                # Ejecutar episodio
                episode_data = self.run_training_episode(episode)
                
                # Actualizar visualización
                self.visualizer.update_metrics(episode_data)
                
                # Guardar modelos periódicamente
                if episode % self.save_interval == 0 and episode > 0:
                    self.save_models(episode)
                
                # Detectar convergencia
                if episode > 100:
                    recent_rewards = []
                    for agent_rewards in self.visualizer.metrics['episode_rewards'].values():
                        if len(agent_rewards) >= 50:
                            recent_rewards.extend(agent_rewards[-50:])
                    
                    if recent_rewards and np.std(recent_rewards) < 1.0:
                        if not self.training_metrics['convergence_episode']:
                            self.training_metrics['convergence_episode'] = episode
                            print(f"🎯 Convergencia detectada en episodio {episode}")
            
            print("✅ Entrenamiento completado!")
            
        except KeyboardInterrupt:
            print("\n⏹️ Entrenamiento interrumpido por el usuario")
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
        finally:
            # Detener visualización y guardar resultados
            self.visualizer.stop_training_visualization()
            self.save_final_results()
    
    def save_models(self, episode):
        """Guarda los modelos de los agentes"""
        try:
            models_dir = "trained_models"
            os.makedirs(models_dir, exist_ok=True)
            
            for agent_id, agent in self.agents.items():
                model_path = os.path.join(models_dir, f"model_{agent_id}_ep_{episode}.pth")
                agent.save(model_path)
            
            print(f"💾 Modelos guardados (episodio {episode})")
            
        except Exception as e:
            print(f"Error guardando modelos: {e}")
    
    def save_final_results(self):
        """Guarda los resultados finales"""
        try:
            # Guardar gráficos
            self.visualizer.save_training_plots()
            
            # Generar informe
            report = self.visualizer.generate_training_report()
            
            # Guardar modelos finales
            self.save_models("final")
            
            # Estadísticas de entrenamiento
            total_time = time.time() - self.training_metrics['start_time']
            print(f"\n📊 RESUMEN DEL ENTRENAMIENTO")
            print(f"⏱️ Tiempo total: {total_time/60:.1f} minutos")
            print(f"🎯 Episodios completados: {self.visualizer.current_episode}")
            
            if self.training_metrics['convergence_episode']:
                print(f"🎯 Convergencia en episodio: {self.training_metrics['convergence_episode']}")
            
            # Mostrar rendimiento final por agente
            for agent_id, rewards in self.visualizer.metrics['episode_rewards'].items():
                if rewards:
                    final_avg = np.mean(rewards[-50:])  # Promedio últimos 50
                    print(f"🤖 {agent_id}: Recompensa promedio final = {final_avg:.2f}")
            
            return report
            
        except Exception as e:
            print(f"Error guardando resultados finales: {e}")
            return None


def main():
    """Función principal para ejecutar el entrenamiento mejorado"""
    print("🎮 SISTEMA DE ENTRENAMIENTO MEJORADO - MUS MARL")
    print("=" * 50)
    
    # Configurar entrenamiento
    training_system = EnhancedTrainingSystem(
        max_episodes=300,  # Número de episodios
        save_interval=50   # Guardar cada 50 episodios
    )
    
    # Ejecutar entrenamiento
    training_system.train_with_visualization()
    
    print("\n🎉 ¡Entrenamiento completado!")
    print("📊 Revisa los gráficos y resultados generados")


if __name__ == "__main__":
    main()
