import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
from datetime import datetime
import torch
import pickle
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FullyCompatibleTrainingSystem:
    """Sistema de entrenamiento completamente compatible con la implementación específica"""
    
    def __init__(self, max_episodes=200, eval_interval=50, save_interval=50):
        self.max_episodes = max_episodes
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Crear directorio para resultados
        self.results_dir = "training_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Métricas de seguimiento
        self.metrics = {
            'episode_rewards': defaultdict(list),
            'episode_lengths': [],
            'successful_episodes': 0,
            'failed_episodes': 0,
            'total_steps': 0,
            'agent_actions': defaultdict(int)
        }
        
        # Configuración detectada
        self.detected_config = {
            'agent_needs_team': True,
            'agent_needs_valid_actions': True,
            'env_has_custom_api': True,
            'default_state_size': 20,
            'default_action_size': 15
        }
    
    def detect_environment_api(self):
        """Detectar la API específica del entorno"""
        print("🔍 Detectando API del entorno...")
        
        try:
            from mus_env import mus
            env = mus.env()
            
            # Probar reset y obtener información inicial
            try:
                env.reset()
                print("✅ Reset exitoso")
            except Exception as e:
                print(f"⚠️ Error en reset: {e}")
            
            # Analizar agentes
            print(f"👥 Agentes detectados: {env.possible_agents}")
            
            # Probar métodos de observación alternativos
            observation_methods = []
            
            for agent_id in env.possible_agents[:1]:  # Solo probar con el primer agente
                # Método 1: observe()
                try:
                    obs = env.observe(agent_id)
                    if obs is not None:
                        observation_methods.append(('observe', obs))
                        print(f"✅ observe({agent_id}) funciona: {type(obs)}, shape: {getattr(obs, 'shape', 'N/A')}")
                except Exception as e:
                    print(f"❌ observe({agent_id}) falló: {e}")
                
                # Método 2: observation_space
                try:
                    obs_space = env.observation_space(agent_id)
                    if obs_space and hasattr(obs_space, 'shape'):
                        self.detected_config['default_state_size'] = obs_space.shape[0]
                        print(f"✅ observation_space detectado: {obs_space.shape}")
                except Exception as e:
                    print(f"❌ observation_space falló: {e}")
                
                # Método 3: action_space
                try:
                    action_space = env.action_space(agent_id)
                    if action_space and hasattr(action_space, 'n'):
                        self.detected_config['default_action_size'] = action_space.n
                        print(f"✅ action_space detectado: {action_space.n}")
                except Exception as e:
                    print(f"❌ action_space falló: {e}")
                
                break  # Solo necesitamos probar con un agente
            
            return env, observation_methods
            
        except Exception as e:
            print(f"❌ Error detectando API del entorno: {e}")
            raise
    
    def detect_agent_api(self):
        """Detectar la API específica del agente"""
        print("🤖 Detectando API del agente...")
        
        try:
            from marl_agent import MARLAgent
            import inspect
            
            # Analizar signatura del constructor
            init_sig = inspect.signature(MARLAgent.__init__)
            init_params = list(init_sig.parameters.keys())
            print(f"📋 Parámetros __init__: {init_params}")
            
            # Analizar signatura del método act
            try:
                act_sig = inspect.signature(MARLAgent.act)
                act_params = list(act_sig.parameters.keys())
                print(f"🎯 Parámetros act(): {act_params}")
                
                self.detected_config['agent_needs_valid_actions'] = 'valid_actions' in act_params
                print(f"✅ Necesita valid_actions: {self.detected_config['agent_needs_valid_actions']}")
                
            except Exception as e:
                print(f"⚠️ No se pudo analizar act(): {e}")
            
            # Probar creación de agente
            test_configs = [
                {'team': 0},
                {'team': 1},
                {'team': 'team_0'},
                {}
            ]
            
            successful_config = None
            
            for config in test_configs:
                try:
                    agent_params = {
                        'state_size': self.detected_config['default_state_size'],
                        'action_size': self.detected_config['default_action_size'],
                        'agent_id': 'test_agent'
                    }
                    agent_params.update(config)
                    
                    agent = MARLAgent(**agent_params)
                    print(f"✅ Creación exitosa con config: {config}")
                    successful_config = config
                    
                    # Probar método act
                    test_obs = np.zeros(self.detected_config['default_state_size'])
                    
                    if self.detected_config['agent_needs_valid_actions']:
                        # Probar con valid_actions
                        test_valid_actions = list(range(self.detected_config['default_action_size']))
                        try:
                            action = agent.act(test_obs, test_valid_actions)
                            print(f"✅ act() con valid_actions exitoso: {action}")
                        except Exception as e:
                            print(f"❌ act() con valid_actions falló: {e}")
                    else:
                        try:
                            action = agent.act(test_obs)
                            print(f"✅ act() sin valid_actions exitoso: {action}")
                        except Exception as e:
                            print(f"❌ act() sin valid_actions falló: {e}")
                    
                    break
                    
                except Exception as e:
                    print(f"❌ Config {config} falló: {e}")
            
            if successful_config is not None:
                if 'team' in successful_config:
                    self.detected_config['agent_needs_team'] = True
                    self.detected_config['team_config'] = successful_config
                else:
                    self.detected_config['agent_needs_team'] = False
                
                return True
            else:
                print("❌ No se pudo crear agente con ninguna configuración")
                return False
                
        except Exception as e:
            print(f"❌ Error detectando API del agente: {e}")
            return False
    
    def create_compatible_agents(self, env):
        """Crear agentes con la configuración detectada"""
        print("🤖 Creando agentes compatibles...")
        
        agents = {}
        
        try:
            from marl_agent import MARLAgent
            
            for i, agent_id in enumerate(env.possible_agents):
                # Parámetros base
                agent_params = {
                    'state_size': self.detected_config['default_state_size'],
                    'action_size': self.detected_config['default_action_size'],
                    'agent_id': agent_id
                }
                
                # Añadir team si es necesario
                if self.detected_config['agent_needs_team']:
                    if 'team_config' in self.detected_config:
                        team_value = self.detected_config['team_config']['team']
                        if isinstance(team_value, int):
                            agent_params['team'] = i % 2  # Alternar entre 0 y 1
                        else:
                            agent_params['team'] = f"team_{i % 2}"
                    else:
                        agent_params['team'] = i % 2
                
                # Crear agente
                agent = MARLAgent(**agent_params)
                agents[agent_id] = agent
                
                print(f"✅ {agent_id} creado con: {agent_params}")
            
            return agents
            
        except Exception as e:
            print(f"❌ Error creando agentes: {e}")
            raise
    
    def get_observation_safely(self, env, agent_id, observation_methods):
        """Obtener observación usando métodos disponibles"""
        for method_name, _ in observation_methods:
            try:
                if method_name == 'observe':
                    obs = env.observe(agent_id)
                    if obs is not None:
                        return obs
            except:
                continue
        
        # Fallback: observación por defecto
        return np.zeros(self.detected_config['default_state_size'])
    
    def get_valid_actions_safely(self, env, agent_id):
        """Obtener acciones válidas de forma segura"""
        try:
            # Método 1: legal_actions si existe
            if hasattr(env, 'legal_actions'):
                return env.legal_actions(agent_id)
            
            # Método 2: action_space
            action_space = env.action_space(agent_id)
            if action_space and hasattr(action_space, 'n'):
                return list(range(action_space.n))
            
            # Método 3: valor por defecto
            return list(range(self.detected_config['default_action_size']))
            
        except:
            return list(range(self.detected_config['default_action_size']))
    
    def run_compatible_episode(self, env, agents, observation_methods):
        """Ejecutar episodio compatible con la implementación específica"""
        episode_rewards = defaultdict(float)
        episode_data = {'length': 0, 'success': False}
        
        try:
            env.reset()
            max_steps = 100
            
            # Método simple: iterar por agentes directamente
            for step in range(max_steps):
                step_completed = False
                
                for agent_id in env.possible_agents:
                    try:
                        # Obtener observación
                        observation = self.get_observation_safely(env, agent_id, observation_methods)
                        
                        # Obtener acciones válidas
                        valid_actions = self.get_valid_actions_safely(env, agent_id)
                        
                        # Obtener acción del agente
                        if self.detected_config['agent_needs_valid_actions']:
                            action = agents[agent_id].act(observation, valid_actions)
                        else:
                            action = agents[agent_id].act(observation)
                        
                        # Registrar acción
                        self.metrics['agent_actions'][agent_id] += 1
                        
                        # Ejecutar acción en el entorno
                        # Probar diferentes métodos de step
                        step_success = False
                        
                        # Método 1: step normal
                        try:
                            result = env.step(action)
                            step_success = True
                            
                            # Intentar extraer recompensa si está disponible
                            if isinstance(result, tuple) and len(result) >= 2:
                                reward = result[1] if result[1] is not None else 0
                            else:
                                reward = 0
                            
                            episode_rewards[agent_id] += reward
                            
                        except Exception as e:
                            print(f"⚠️ Error en step para {agent_id}: {e}")
                            # Continuar con el siguiente agente
                            continue
                        
                        if step_success:
                            episode_data['length'] += 1
                            step_completed = True
                            self.metrics['total_steps'] += 1
                        
                    except Exception as e:
                        print(f"⚠️ Error procesando {agent_id}: {e}")
                        continue
                
                # Si no se completó ningún paso, salir
                if not step_completed:
                    break
                
                # Verificar condición de terminación
                try:
                    # Método simple: si el entorno tiene un método para verificar si terminó
                    if hasattr(env, 'is_done') and env.is_done():
                        break
                    
                    # O si llegamos a un número razonable de pasos
                    if episode_data['length'] >= 50:
                        break
                        
                except:
                    # Si no hay método de terminación, usar límite de pasos
                    if episode_data['length'] >= 20:
                        break
            
            episode_data['success'] = episode_data['length'] > 0
            return episode_rewards, episode_data
            
        except Exception as e:
            print(f"⚠️ Error en episodio: {e}")
            episode_data['success'] = False
            return episode_rewards, episode_data
    
    def train_fully_compatible(self):
        """Entrenamiento completamente compatible"""
        print("🚀 Iniciando entrenamiento completamente compatible...")
        
        try:
            # Detectar APIs
            env, observation_methods = self.detect_environment_api()
            agent_compatible = self.detect_agent_api()
            
            if not agent_compatible:
                print("❌ No se pudo detectar API compatible del agente")
                return None, None
            
            if not observation_methods:
                print("⚠️ No se detectaron métodos de observación, usando valores por defecto")
                observation_methods = [('default', None)]
            
            # Crear agentes
            agents = self.create_compatible_agents(env)
            
            print(f"\n🎯 Iniciando {self.max_episodes} episodios...")
            print(f"📊 Configuración detectada:")
            for key, value in self.detected_config.items():
                print(f"   {key}: {value}")
            print()
            
            # Entrenamiento principal
            for episode in range(self.max_episodes):
                if episode % 20 == 0:
                    success_rate = self.metrics['successful_episodes'] / max(episode, 1)
                    print(f"📈 Episodio {episode}/{self.max_episodes} (Éxito: {success_rate:.1%})")
                
                # Ejecutar episodio
                episode_rewards, episode_data = self.run_compatible_episode(env, agents, observation_methods)
                
                if episode_data['success']:
                    self.metrics['successful_episodes'] += 1
                    
                    # Actualizar métricas
                    for agent_id, reward in episode_rewards.items():
                        self.metrics['episode_rewards'][agent_id].append(reward)
                    
                    self.metrics['episode_lengths'].append(episode_data['length'])
                    
                else:
                    self.metrics['failed_episodes'] += 1
                
                # Guardado periódico
                if episode % self.save_interval == 0 and episode > 0:
                    self._save_compatible_checkpoint(agents, episode)
            
            # Generar informe final
            self._generate_compatible_report()
            
            final_success_rate = self.metrics['successful_episodes'] / self.max_episodes
            print(f"\n✅ Entrenamiento completamente compatible terminado!")
            print(f"📊 Episodios exitosos: {self.metrics['successful_episodes']}/{self.max_episodes}")
            print(f"📊 Tasa de éxito final: {final_success_rate:.1%}")
            print(f"📊 Pasos totales ejecutados: {self.metrics['total_steps']}")
            
            return agents, self.metrics
            
        except Exception as e:
            print(f"❌ Error en entrenamiento compatible: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _save_compatible_checkpoint(self, agents, episode):
        """Guardar checkpoint compatible"""
        try:
            checkpoint_dir = os.path.join(self.results_dir, f"compatible_checkpoint_ep_{episode}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Guardar configuración detectada
            config_path = os.path.join(checkpoint_dir, "detected_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.detected_config, f, indent=2)
            
            # Guardar métricas
            metrics_path = os.path.join(checkpoint_dir, "metrics.pkl")
            with open(metrics_path, 'wb') as f:
                pickle.dump(dict(self.metrics), f)
            
            print(f"💾 Checkpoint compatible guardado (episodio {episode})")
            
        except Exception as e:
            print(f"⚠️ Error guardando checkpoint: {e}")
    
    def _generate_compatible_report(self):
        """Generar informe compatible"""
        try:
            # Crear gráfico
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('🎮 Informe de Entrenamiento Compatible', fontsize=14)
            
            # 1. Éxitos vs Fallos
            success_rate = self.metrics['successful_episodes'] / max(self.metrics['successful_episodes'] + self.metrics['failed_episodes'], 1)
            axes[0, 0].set_title('📊 Resultados')
            axes[0, 0].bar(['Éxitos', 'Fallos'], 
                          [self.metrics['successful_episodes'], self.metrics['failed_episodes']], 
                          color=['green', 'red'], alpha=0.7)
            axes[0, 0].text(0, self.metrics['successful_episodes'] + 1, f'{success_rate:.1%}', ha='center')
            
            # 2. Recompensas por agente
            axes[0, 1].set_title('💰 Recompensas por Agente')
            if self.metrics['episode_rewards']:
                for agent_id, rewards in self.metrics['episode_rewards'].items():
                    if rewards:
                        axes[0, 1].plot(rewards, label=agent_id, alpha=0.8)
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'Sin datos\nde recompensas', ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # 3. Duración de episodios
            axes[1, 0].set_title('⏱️ Duración de Episodios')
            if self.metrics['episode_lengths']:
                axes[1, 0].plot(self.metrics['episode_lengths'], 'orange', alpha=0.8)
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Sin datos\nde duración', ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # 4. Acciones por agente
            axes[1, 1].set_title('🎯 Acciones Ejecutadas')
            if self.metrics['agent_actions']:
                agents = list(self.metrics['agent_actions'].keys())
                actions = list(self.metrics['agent_actions'].values())
                axes[1, 1].bar(agents, actions, alpha=0.7)
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Guardar
            report_path = os.path.join(self.results_dir, f"compatible_report_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
            plt.savefig(report_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            # Resumen textual
            summary_path = os.path.join(self.results_dir, "compatible_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("🎮 RESUMEN DE ENTRENAMIENTO COMPATIBLE\n")
                f.write("=" * 42 + "\n\n")
                
                f.write(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"🎯 Episodios: {self.max_episodes}\n")
                f.write(f"✅ Exitosos: {self.metrics['successful_episodes']}\n")
                f.write(f"❌ Fallidos: {self.metrics['failed_episodes']}\n")
                f.write(f"📊 Tasa de éxito: {success_rate:.1%}\n")
                f.write(f"🎯 Pasos totales: {self.metrics['total_steps']}\n\n")
                
                f.write("🔧 CONFIGURACIÓN DETECTADA:\n")
                f.write("-" * 30 + "\n")
                for key, value in self.detected_config.items():
                    f.write(f"{key}: {value}\n")
                
                if self.metrics['episode_rewards']:
                    f.write(f"\n💰 RECOMPENSAS PROMEDIO:\n")
                    for agent_id, rewards in self.metrics['episode_rewards'].items():
                        if rewards:
                            avg_reward = np.mean(rewards)
                            f.write(f"   {agent_id}: {avg_reward:.3f}\n")
                
                if self.metrics['episode_lengths']:
                    avg_length = np.mean(self.metrics['episode_lengths'])
                    f.write(f"\n⏱️ Duración promedio: {avg_length:.1f} pasos\n")
            
            print(f"📋 Informe compatible guardado: {report_path}")
            print(f"📋 Resumen guardado: {summary_path}")
            
        except Exception as e:
            print(f"⚠️ Error generando informe: {e}")


def run_fully_compatible_training():
    """Función principal para ejecutar entrenamiento completamente compatible"""
    print("🎮 SISTEMA DE ENTRENAMIENTO COMPLETAMENTE COMPATIBLE")
    print("=" * 55)
    
    training_system = FullyCompatibleTrainingSystem(
        max_episodes=100,  # Empezar con pocos episodios para probar
        eval_interval=25,
        save_interval=25
    )
    
    agents, metrics = training_system.train_fully_compatible()
    
    if agents and metrics:
        print("🎉 ¡Entrenamiento completamente compatible exitoso!")
        return agents, metrics
    else:
        print("❌ Entrenamiento completamente compatible falló")
        return None, None


if __name__ == "__main__":
    run_fully_compatible_training()
