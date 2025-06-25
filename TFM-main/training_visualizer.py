import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
import time
import threading
from datetime import datetime
import os
import json

class TrainingVisualizer:
    """Visualizador en tiempo real del progreso de entrenamiento"""
    
    def __init__(self, update_interval=10):
        self.update_interval = update_interval
        self.metrics = {
            'episode_rewards': defaultdict(list),
            'episode_lengths': [],
            'win_rates': defaultdict(list),
            'loss_values': defaultdict(list),
            'epsilon_values': defaultdict(list),
            'team_performance': defaultdict(list),
            'phase_success': defaultdict(list)
        }
        
        self.current_episode = 0
        self.is_training = False
        self.fig = None
        self.axes = None
        
        # Buffer para suavizar las curvas
        self.smoothing_window = 20
        
    def start_training_visualization(self):
        """Inicia la visualización en tiempo real"""
        self.is_training = True
        self.setup_plots()
        
        # Hilo para actualizar gráficos
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def setup_plots(self):
        """Configura los gráficos iniciales"""
        plt.ion()  # Modo interactivo
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('🎮 Entrenamiento de Agentes MARL - Mus', fontsize=16)
        
        # Configurar cada subplot
        titles = [
            '📈 Recompensas por Episodio',
            '🎯 Tasa de Victoria por Equipo', 
            '🧠 Valores de Epsilon (Exploración)',
            '⚡ Pérdida de la Red Neuronal',
            '⏱️ Duración de Episodios',
            '🏆 Rendimiento por Fase'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i])
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show(block=False)
        
    def update_metrics(self, episode_data):
        """Actualiza las métricas con datos del episodio"""
        try:
            episode = episode_data.get('episode', self.current_episode)
            
            # Recompensas por agente
            rewards_data = episode_data.get('rewards', {})
            if isinstance(rewards_data, dict):
                for agent_id, reward in rewards_data.items():
                    if reward is not None:
                        try:
                            reward_val = float(reward)
                            if not np.isnan(reward_val):
                                self.metrics['episode_rewards'][agent_id].append(reward_val)
                        except (TypeError, ValueError):
                            pass
            
            # Duración del episodio
            if 'length' in episode_data and episode_data['length'] is not None:
                try:
                    length = int(episode_data['length'])
                    if length > 0:
                        self.metrics['episode_lengths'].append(length)
                except (TypeError, ValueError):
                    pass
            
            # Tasa de victoria por equipo
            if 'winner' in episode_data and episode_data['winner'] is not None:
                winner = episode_data['winner']
                for team in ['equipo_1', 'equipo_2']:
                    win = 1.0 if winner == team else 0.0
                    self.metrics['win_rates'][team].append(win)
                    
            # Valores de epsilon
            epsilons_data = episode_data.get('epsilons', {})
            if isinstance(epsilons_data, dict):
                for agent_id, epsilon in epsilons_data.items():
                    if epsilon is not None:
                        try:
                            epsilon_val = float(epsilon)
                            if not np.isnan(epsilon_val):
                                self.metrics['epsilon_values'][agent_id].append(epsilon_val)
                        except (TypeError, ValueError):
                            pass
            
            # Pérdidas de entrenamiento
            losses_data = episode_data.get('losses', {})
            if isinstance(losses_data, dict):
                for agent_id, loss in losses_data.items():
                    if loss is not None:
                        try:
                            loss_val = float(loss)
                            if not np.isnan(loss_val) and loss_val > 0:
                                self.metrics['loss_values'][agent_id].append(loss_val)
                        except (TypeError, ValueError):
                            pass
            
            # Rendimiento por fase
            phase_performance = episode_data.get('phase_performance', {})
            if isinstance(phase_performance, dict):
                for phase, success in phase_performance.items():
                    if success is not None:
                        try:
                            success_value = 1.0 if success else 0.0
                            self.metrics['phase_success'][phase].append(success_value)
                        except (TypeError, ValueError):
                            pass
            
            self.current_episode = episode
        
        except Exception as e:
            print(f"Error actualizando métricas: {e}")
        
    def _smooth_curve(self, data, window=None):
        """Suaviza una curva usando media móvil"""
        if data is None or (hasattr(data, '__len__')) and len(data) == 0:
            return []
        
        # Convertir a lista si es un array de numpy
        if hasattr(data, 'tolist'):
            data = data.tolist()
        
        # Filtrar valores None y NaN
        clean_data = []
        for item in data:
            if item is None:
                clean_data.append(clean_data[-1] if clean_data else 0.0)
            elif isinstance(item, (np.ndarray, list)):
                # Handle arrays/lists by taking the first element (or another logic)
                clean_data.append(float(item[0]) if len(item) > 0 else 0.0)
            else:
                try:
                    clean_data.append(float(item))
                except (ValueError, TypeError):
                    clean_data.append(clean_data[-1] if clean_data else 0.0)
        
        if len(clean_data) == 0:
            return []
        
        window = window or self.smoothing_window
        if len(clean_data) < window:
            return clean_data
        
        smoothed = []
        for i in range(len(clean_data)):
            start_idx = max(0, i - window + 1)
            window_data = clean_data[start_idx:i+1]
            smoothed.append(np.mean(window_data))
        
        return smoothed
        
    def _update_loop(self):
        """Bucle principal de actualización de gráficos"""
        while self.is_training:
            try:
                self._update_plots()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error actualizando gráficos: {e}")
                
    
    def _update_plots(self):
        """Actualiza todos los gráficos"""
        if not self.fig or not self.axes:
            return

        try:
            # Limpiar axes
            for ax in self.axes.flat:
                ax.clear()
                
            # 1. Recompensas por episodio
            ax = self.axes[0, 0]
            ax.set_title('📈 Recompensas por Episodio')
            for agent_id, rewards in self.metrics['episode_rewards'].items():
                if len(rewards) > 0:
                    episodes = range(len(rewards))
                    smoothed = self._smooth_curve(rewards)
                    ax.plot(episodes, rewards, alpha=0.3, label=f'{agent_id} (raw)')
                    ax.plot(episodes, smoothed, linewidth=2, label=f'{agent_id} (smooth)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Episodio')
            ax.set_ylabel('Recompensa')
            
            # 2. Tasa de victoria por equipo
            ax = self.axes[0, 1]
            ax.set_title('🎯 Tasa de Victoria por Equipo')
            for team, wins in self.metrics['win_rates'].items():
                if len(wins) > 0:
                    # Calcular tasa de victoria móvil
                    win_rate = []
                    window = min(50, len(wins))  # Usar ventana más pequeña si hay pocos datos
                    for i in range(len(wins)):
                        start_idx = max(0, i - window + 1)
                        win_rate.append(np.mean(wins[start_idx:i+1]))
                    ax.plot(range(len(win_rate)), win_rate, linewidth=2, label=team)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Episodio')
            ax.set_ylabel('Tasa de Victoria')
            ax.set_ylim(0, 1)
            
            # 3. Valores de epsilon
            ax = self.axes[0, 2]
            ax.set_title('🧠 Valores de Epsilon (Exploración)')
            for agent_id, epsilons in self.metrics['epsilon_values'].items():
                if len(epsilons) > 0:
                    ax.plot(range(len(epsilons)), epsilons, linewidth=2, label=agent_id)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Episodio')
            ax.set_ylabel('Epsilon')
            
            # 4. Pérdida de la red neuronal
            ax = self.axes[1, 0]
            ax.set_title('⚡ Pérdida de la Red Neuronal')
            has_loss_data = False
            for agent_id, losses in self.metrics['loss_values'].items():
                if len(losses) > 0:
                    # Filtrar valores None y convertir a float
                    valid_losses = [float(loss) for loss in losses if loss is not None and not np.isnan(float(loss))]
                    if len(valid_losses) > 0:
                        smoothed_losses = self._smooth_curve(valid_losses, window=10)
                        ax.plot(range(len(smoothed_losses)), smoothed_losses, linewidth=2, label=agent_id)
                        has_loss_data = True
            
            if has_loss_data:
                ax.legend()
                ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, 'Sin datos de pérdida\ndisponibles', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Paso de Entrenamiento')
            ax.set_ylabel('Pérdida')
            
            # 5. Duración de episodios
            ax = self.axes[1, 1]
            ax.set_title('⏱️ Duración de Episodios')
            if len(self.metrics['episode_lengths']) > 0:
                lengths = self.metrics['episode_lengths']
                smoothed_lengths = self._smooth_curve(lengths)
                ax.plot(range(len(lengths)), lengths, alpha=0.3, label='Raw')
                ax.plot(range(len(smoothed_lengths)), smoothed_lengths, linewidth=2, label='Smooth')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'Sin datos de\nduración disponibles', 
                       ha='center', va='center', transform=ax.transAxes)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Episodio')
            ax.set_ylabel('Pasos')
            
            # 6. Rendimiento por fase
            ax = self.axes[1, 2]
            ax.set_title('🏆 Rendimiento por Fase')
            phases = ['GRANDE', 'CHICA', 'PARES', 'JUEGO']
            phase_scores = []
            
            for phase in phases:
                if phase in self.metrics['phase_success'] and len(self.metrics['phase_success'][phase]) > 0:
                    recent_performance = self.metrics['phase_success'][phase][-50:]  # Últimos 50
                    if len(recent_performance) > 0:
                        # Asegurar que todos los valores son numéricos
                        numeric_performance = [float(x) for x in recent_performance if x is not None]
                        if len(numeric_performance) > 0:
                            phase_scores.append(np.mean(numeric_performance))
                        else:
                            phase_scores.append(0.0)
                    else:
                        phase_scores.append(0.0)
                else:
                    phase_scores.append(0.0)
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = ax.bar(phases, phase_scores, color=colors, alpha=0.7)
            ax.set_ylabel('Tasa de Éxito')
            ax.set_ylim(0, 1)
            
            # Añadir valores en las barras
            for bar, score in zip(bars, phase_scores):
                height = bar.get_height()
                if height > 0:  # Solo mostrar si hay datos
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.2f}', ha='center', va='bottom')
            
            # Actualizar título con información actual
            if self.current_episode > 0:
                self.fig.suptitle(f'🎮 Entrenamiento de Agentes MARL - Mus (Episodio {self.current_episode})', 
                                fontsize=16)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"Error específico en _update_plots: {e}")
            import traceback
            traceback.print_exc()
        
    def stop_training_visualization(self):
        """Detiene la visualización"""
        self.is_training = False
        if hasattr(self, 'update_thread'):
            self.update_thread.join(timeout=1)
            
    def save_training_plots(self, save_dir="training_results"):
        """Guarda los gráficos finales"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"training_progress_{timestamp}.png")
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Gráficos de entrenamiento guardados: {filename}")
            
    def generate_training_report(self, save_dir="training_results"):
        """Genera un informe completo del entrenamiento"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(save_dir, f"training_report_{timestamp}.json")
        
        # Calcular estadísticas finales
        report = {
            "timestamp": timestamp,
            "total_episodes": self.current_episode,
            "final_statistics": {}
        }
        
        # Estadísticas de recompensas
        for agent_id, rewards in self.metrics['episode_rewards'].items():
            if rewards:
                report["final_statistics"][agent_id] = {
                    "avg_reward": float(np.mean(rewards)),
                    "max_reward": float(np.max(rewards)),
                    "min_reward": float(np.min(rewards)),
                    "final_reward": float(rewards[-1]),
                    "improvement": float(rewards[-1] - rewards[0]) if len(rewards) > 1 else 0
                }
        
        # Estadísticas de equipos
        for team, wins in self.metrics['win_rates'].items():
            if wins:
                recent_wins = wins[-100:]  # Últimos 100 episodios
                report["final_statistics"][f"{team}_win_rate"] = float(np.mean(recent_wins))
        
        # Guardar informe
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"📋 Informe de entrenamiento guardado: {report_file}")
        return report
