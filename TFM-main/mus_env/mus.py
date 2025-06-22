from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
import numpy as np
import random
import time

class MusEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "mus_v0"}

    def __init__(self):
        super().__init__()
        global fin, ronda_completa
        fin = False
        ronda_completa = False
        self.mano = 0

        self.agents = [f"jugador_{i}" for i in range(4)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
        
        # Definir equipos (0,2) y (1,3)
        self.equipos = {
            "equipo_1": ["jugador_0", "jugador_2"],
            "equipo_2": ["jugador_1", "jugador_3"]
        }
        # Mapeo inverso para saber a qué equipo pertenece cada jugador
        self.equipo_de_jugador = {
            "jugador_0": "equipo_1",
            "jugador_1": "equipo_2", 
            "jugador_2": "equipo_1",
            "jugador_3": "equipo_2"
        }

        # Fases del juego
        self.fases = ["MUS", "DESCARTE", "GRANDE", "CHICA", "PARES", "JUEGO", "RECUENTO"]
        self.fase_actual = self.fases[0]

        self.manos = {}
        self.cartas_a_descartar = {}
        self.votos_mus = []
        self.historial_apuestas = []
        
        # Diccionarios para almacenar declaraciones
        self.declaraciones_pares = {}
        self.declaraciones_juego = {}
        self.valores_juego = {}
        
        # Registro de decisiones de los jugadores
        self.ultima_decision = {agent: "Esperando..." for agent in self.agents}
        
        # Control de apuestas
        self.apuesta_actual = 0
        apuesta_actual_ordago = 0
        self.equipo_apostador = None
        self.jugador_apostador = None
        self.ronda_completa = False
        self.jugadores_pasado = set()
        self.jugadores_hablaron = set()
        self.jugadores_que_pueden_hablar = set()
        self.hay_ordago = False
        self.jugadores_confirmaron_descarte = set()
        
        self.hand_size = 4
        # Crear mazo sin 8s y 9s correctamente
        self.deck = [(v, s) for v in range(1, 13) for s in range(4) if v not in [8, 9]]
        self.mazo = self.deck.copy()

        # Acciones: 0=pasar, 1=envido, 2=mus, 3=no mus, 4=confirmar, 5=no quiero, 6=ordago, 
        # 7=quiero (capear), 11-14=descartar carta 0-3
        self.action_spaces = {agent: spaces.Discrete(15) for agent in self.agents}
        
        self.observation_spaces = {
            agent: spaces.Dict({
                "cartas": spaces.Box(low=1, high=12, shape=(self.hand_size, 2), dtype=np.int8),
                "fase": spaces.Discrete(len(self.fases)),
                "turno": spaces.Discrete(4)
            }) for agent in self.agents
        }

        # CORRECCIÓN: Estructura de apuestas que coincida con la GUI
        self.apuestas = {
            "equipo_1": {
                "GRANDE": 0,
                "CHICA": 0,
                "PARES": 0,
                "JUEGO": 0
            },
            "equipo_2": {
                "GRANDE": 0,
                "CHICA": 0,
                "PARES": 0,
                "JUEGO": 0
            }
        }
        
        self.acciones_validas = {
            "GRANDE": [0, 1, 5, 6, 7],
            "CHICA": [0, 1, 5, 6, 7],
            "PARES": [0, 1, 5, 6, 7],
            "JUEGO": [0, 1, 5, 6, 7]
        }
        
        # Inicializar dones y rewards
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Puntos totales para los equipos
        self.puntos_equipos = {"equipo_1": 0, "equipo_2": 0}

        self.partidas_ganadas = {"equipo_1": 0, "equipo_2": 0}  # Nuevo: registro de partidas ganadas
        self.partida_terminada = False  # Nuevo: indica si la partida actual terminó
        
        # Ganadores de cada fase para el recuento final
        self.ganadores_fases = {
            "GRANDE": None,
            "CHICA": None,
            "PARES": None,
            "JUEGO": None
        }

        # Control de tiempo para las acciones
        self.action_delay = 1.0  # 1 segundo de delay
        self.last_action_time = 0

        self.primera_ronda = True

    def generar_mazo(self):
        """Generar mazo correctamente sin 8s y 9s"""
        self.mazo = [(v, p) for p in range(4) for v in range(1, 13) if v not in [8, 9]]
        random.shuffle(self.mazo)

    def reset(self, seed=None):
        global fin, ronda_completa  # <-- Añade esta línea
        if seed is not None:
            random.seed(seed)
            
        self.generar_mazo()
        self.cartas_a_descartar = {agent: [] for agent in self.agents}
        self.votos_mus = []
        self.historial_apuestas = []
        self.declaraciones_pares = {}
        self.declaraciones_juego = {}
        self.valores_juego = {}
        self.ultima_decision = {agent: "Esperando..." for agent in self.agents}
        if fin:
            self.partidas_ganadas = {"equipo_1": 0, "equipo_2": 0}
            fin = False
            self.primera_ronda = True

        if ronda_completa:
            self.puntos_equipos = {"equipo_1": 0, "equipo_2": 0}
            ronda_completa = False

        self.partida_terminada = False
        
        # Reiniciar control de apuestas
        self.apuesta_actual = 0
        self.equipo_apostador = None
        self.jugador_apostador = None
        self.ronda_completa = False
        self.jugadores_pasado = set()
        self.jugadores_hablaron = set()
        self.jugadores_que_pueden_hablar = set()
        self.hay_ordago = False

        self.equipo_de_jugador = {
            "jugador_0": "equipo_1",
            "jugador_1": "equipo_2", 
            "jugador_2": "equipo_1",
            "jugador_3": "equipo_2"
        }
        
        self.agents = self.possible_agents[:]
        self.agents = self.agents[self.mano:] + self.agents[:self.mano]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        print(f"Agente seleccionado: {self.agent_selection}")
        
        self.repartir_cartas()
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.fase_actual = "MUS"
        
        # Reiniciar apuestas
        for equipo in self.apuestas:
            for fase in self.apuestas[equipo]:
                self.apuestas[equipo][fase] = 0
            
        # Reiniciar ganadores de fases
        for fase in self.ganadores_fases:
            self.ganadores_fases[fase] = None
            
        # Reiniciar tiempo
        self.last_action_time = time.time()
            
        return self.observe(self.agent_selection)
    
    def cambiar_mano(self, nuevo_mano_jugador):
        """Cambia la mano a un jugador específico y reorganiza el orden de juego"""
        # Obtener el índice del nuevo jugador mano
        nuevo_mano_index = int(nuevo_mano_jugador.split('_')[1])
        self.mano = nuevo_mano_index
        
        # Reorganizar el orden de los agentes con el nuevo mano al frente
        self.agents = self.possible_agents[:]
        self.agents = self.agents[self.mano:] + self.agents[:self.mano]
        
        # IMPORTANTE: Los equipos NO cambian, se mantienen fijos
        # No tocar self.equipo_de_jugador aquí
        
        # Crear nuevo selector de agentes
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        
        print(f"Nueva mano: jugador_{self.mano}, Nuevo orden: {self.agents}")
        print(f"Primer jugador en hablar: {self.agent_selection}")
        print(f"Equipos (NO cambian): {self.equipo_de_jugador}")

    def repartir_cartas(self):
        """Verificar que hay suficientes cartas antes de repartir"""
        if len(self.mazo) < len(self.agents) * self.hand_size:
            self.generar_mazo()
            
        self.manos = {}
        for agent in self.agents:
            self.manos[agent] = []
            for _ in range(self.hand_size):
                if self.mazo:
                    self.manos[agent].append(self.mazo.pop())
                else:
                    self.generar_mazo()
                    self.manos[agent].append(self.mazo.pop())
        
        self.actualizar_declaraciones()

    def actualizar_declaraciones(self):
        """Actualiza automáticamente las declaraciones de pares y juego para todos los jugadores"""
        self.declaraciones_pares = {}
        self.declaraciones_juego = {}
        self.valores_juego = {}
        
        for agent in self.agents:
            if agent in self.manos:
                # Calcular si tiene pares
                self.declaraciones_pares[agent] = self.tiene_pares(self.manos[agent])
                
                # Calcular si tiene juego y su valor
                valor_juego = self.calcular_valor_juego(self.manos[agent])
                self.valores_juego[agent] = valor_juego
                self.declaraciones_juego[agent] = valor_juego >= 31

    def actualizar_jugadores_que_pueden_hablar(self):
        """Mejorar la lógica de quién puede hablar"""
        self.jugadores_que_pueden_hablar = set()
        self.equipos_que_pueden_hablar = set()
        
        if self.fase_actual == "PARES":
            for agent in self.agents:
                if self.declaraciones_pares.get(agent, False):
                    self.jugadores_que_pueden_hablar.add(agent)
                    self.equipos_que_pueden_hablar.add(self.equipo_de_jugador[agent])
        elif self.fase_actual == "JUEGO":
            for agent in self.agents:
                valor_juego = self.calcular_valor_juego(self.manos[agent])
                # Permitir hablar si tiene juego (>=31) o si nadie tiene juego (todos pueden hablar)
                if valor_juego >= 31 or not any(self.declaraciones_juego.values()):
                    self.jugadores_que_pueden_hablar.add(agent)
                    self.equipos_que_pueden_hablar.add(self.equipo_de_jugador[agent])
        else:
            # En otras fases, todos pueden hablar
            self.jugadores_que_pueden_hablar = set(self.agents)
            self.equipos_que_pueden_hablar = {"equipo_1", "equipo_2"}
            
        # Si solo un equipo puede hablar, ese equipo gana automáticamente
        if self.fase_actual in ["PARES", "JUEGO"] and len(self.equipos_que_pueden_hablar) == 1:
            equipo_ganador = list(self.equipos_que_pueden_hablar)[0]
            self.ganadores_fases[self.fase_actual] = equipo_ganador
            
            # Asignar puntos según la fase
            if self.fase_actual == "PARES":
                puntos = self.calcular_puntos_pares(equipo_ganador)
            elif self.fase_actual == "JUEGO":
                puntos = self.calcular_puntos_juego(equipo_ganador)
            else:
                puntos = 1
                
            self.puntos_equipos[equipo_ganador] += puntos
            self.apuestas[equipo_ganador][self.fase_actual] = puntos
            print(f"Equipo {equipo_ganador} gana {puntos} puntos en {self.fase_actual} automáticamente")
            
            self.avanzar_fase()
        elif self.fase_actual in ["PARES", "JUEGO"] and len(self.equipos_que_pueden_hablar) == 0:
            print(f"Nadie puede hablar en {self.fase_actual}, avanzando...")
            self.avanzar_fase()

    def observe(self, agent):
        """Mejorar observación con información del juego"""
        obs = {
            "cartas": np.zeros((self.hand_size, 2), dtype=np.int8),
            "fase": self.fases.index(self.fase_actual),
            "turno": self.agent_name_mapping.get(self.agent_selection, 0)
        }
        
        if agent in self.manos:
            obs["cartas"] = np.array(self.manos[agent], dtype=np.int8)
        
        # Añadir información adicional
        obs["apuesta_actual"] = self.apuesta_actual
        obs["equipo_apostador"] = 0
        if self.equipo_apostador:
            obs["equipo_apostador"] = 1 if self.equipo_apostador == "equipo_1" else 2
        
        return obs
    
    def _was_done_step(self, action):
        """Mejorar manejo de agentes terminados"""
        if self.agents:
            attempts = 0
            while attempts < len(self.agents):
                self.agent_selection = self.agent_selector.next()
                if not self.dones.get(self.agent_selection, False):
                    break
                attempts += 1
        
        if action is not None and self.agent_selection in self.action_spaces:
            assert self.action_spaces[self.agent_selection].contains(action), \
                f"Action {action} is invalid for agent {self.agent_selection}"
        
        if self.agent_selection in self.dones and self.dones[self.agent_selection]:
            self.rewards[self.agent_selection] = 0

    def wait_for_action_delay(self):
        """Esperar el tiempo necesario entre acciones"""
        if self.fase_actual == "RECUENTO":
            return 
        
        current_time = time.time()
        time_since_last_action = current_time - self.last_action_time
        
        if time_since_last_action < self.action_delay:
            time.sleep(self.action_delay - time_since_last_action)
        
        self.last_action_time = time.time()
    
    def calcular_valor_mano_grande(self, mano):
        """Calcula el valor de una mano para GRANDE"""
        if not mano:
            return []
        
        # Jerarquía para GRANDE: Rey(12), Caballo(11), Sota(10), 7, 6, 5, 4, 3, 2, As(1)
        jerarquia_grande = [12, 11, 10, 7, 6, 5, 4, 3, 2, 1]
        
        valores = [carta[0] for carta in mano]
        conteo = {}
        for valor in valores:
            conteo[valor] = conteo.get(valor, 0) + 1
        
        # Crear lista de valores ordenados por jerarquía
        resultado = []
        for valor in jerarquia_grande:
            if valor in conteo:
                resultado.extend([valor] * conteo[valor])
        
        return resultado
    
    def calcular_valor_mano_chica(self, mano):
        """Calcula el valor de una mano para CHICA"""
        if not mano:
            return []
        
        # Jerarquía para CHICA: As(1), 2, 3, 4, 5, 6, 7, Sota(10), Caballo(11), Rey(12)
        jerarquia_chica = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
        
        valores = [carta[0] for carta in mano]
        conteo = {}
        for valor in valores:
            conteo[valor] = conteo.get(valor, 0) + 1
        
        # Crear lista de valores ordenados por jerarquía
        resultado = []
        for valor in jerarquia_chica:
            if valor in conteo:
                resultado.extend([valor] * conteo[valor])
        
        return resultado
    
    def comparar_manos(self, mano1, mano2, fase):
        """Compara dos manos según la fase"""
        if fase == "GRANDE":
            valores1 = self.calcular_valor_mano_grande(mano1)
            valores2 = self.calcular_valor_mano_grande(mano2)
            for i in range(min(len(valores1), len(valores2))):
                if valores1[i] > valores2[i]:
                    return 1  # mano1 gana
                elif valores1[i] < valores2[i]:
                    return -1  # mano2 gana
        elif fase == "CHICA":
            valores1 = self.calcular_valor_mano_chica(mano1)
            valores2 = self.calcular_valor_mano_chica(mano2)
            for i in range(min(len(valores1), len(valores2))):
                if valores1[i] < valores2[i]:
                    return 1  # mano1 gana
                elif valores1[i] > valores2[i]:
                    return -1  # mano2 gana
        else:
            return 0
        
        print(valores1, valores2)
        # Comparar carta por carta
        
        
        # Si todas las cartas son iguales hasta aquí, comparar por longitud
        if len(valores1) > len(valores2):
            return 1
        elif len(valores1) < len(valores2):
            return -1
        
        return 0  # Empate

    def calcular_puntos(self, mano, fase):
        """Versión corregida de calcular_puntos que siempre devuelve un entero"""
        if not mano:
            return 0
            
        if fase == "PARES":
            return self.calcular_puntos_pares_jugador(mano)
                
        elif fase == "JUEGO":
            return self.calcular_valor_juego(mano)
            
        elif fase in ["GRANDE", "CHICA"]:
            # Para GRANDE y CHICA, devolvemos un valor numérico simple para comparaciones
            # Este método solo se usa para comparaciones básicas, no para determinar ganadores
            # La determinación real de ganadores se hace en determinar_ganador_fase
            if fase == "GRANDE":
                valores = self.calcular_valor_mano_grande(mano)
            else:  # CHICA
                valores = self.calcular_valor_mano_chica(mano)
            
            # Convertir la lista de valores en un número para comparaciones simples
            # Usamos un sistema de pesos para crear un valor único
            valor_total = 0
            for i, valor in enumerate(valores[:4]):  # Solo las primeras 4 cartas
                valor_total += valor * (100 ** (3 - i))
            
            return valor_total
        
        return 0

    def calcular_valor_juego(self, mano):
        """Calcula el valor de la mano para juego"""
        if not mano:
            return 0
            
        total = 0
        for valor, _ in mano:
            if valor >= 10:
                total += 10
            else:
                total += valor
        return total

    def tiene_pares(self, mano):
        """Determina si un jugador tiene pares"""
        if not mano:
            return False
            
        valores = [carta[0] for carta in mano]
        counts = {}
        for v in valores:
            counts[v] = counts.get(v, 0) + 1
        return any(c >= 2 for c in counts.values())
    
    def puede_hablar(self, agent):
        """Determina si un jugador puede hablar en la fase actual"""
        if self.fase_actual == "PARES":
            return self.declaraciones_pares.get(agent, False)
        elif self.fase_actual == "JUEGO":
            return self.declaraciones_juego.get(agent, False)
        return True
    
    def siguiente_jugador_que_puede_hablar(self):
        """Mejorar búsqueda del siguiente jugador"""
        self.actualizar_jugadores_que_pueden_hablar()
        if not self.jugadores_que_pueden_hablar:
            print(f"Nadie puede hablar en la fase {self.fase_actual}, avanzando...")
            self.avanzar_fase()
            return
        
        intentos = 0
        while intentos < len(self.agents) * 2:
            self.agent_selection = self.agent_selector.next()
            if (self.agent_selection in self.jugadores_que_pueden_hablar and 
                not self.dones.get(self.agent_selection, False)):
                return
            intentos += 1
        
        print(f"No se encontró jugador válido en fase {self.fase_actual}, avanzando...")
        self.avanzar_fase()
    
    def es_del_mismo_equipo(self, jugador1, jugador2):
        """Determina si dos jugadores son del mismo equipo"""
        return self.equipo_de_jugador.get(jugador1) == self.equipo_de_jugador.get(jugador2)
    
    def step(self, action):
        """Mejorar manejo de pasos y validaciones con delay"""
        # Aplicar delay antes de procesar la acción
        self.wait_for_action_delay()
        agent = self.agent_selection
        print(f"step - Agent: {agent}, Action: {action}, Fase: {self.fase_actual}")
        
        if self.dones.get(agent, False):
            self._was_done_step(action)
            return
            
        if not self.action_spaces[agent].contains(action):
            print(f"Acción inválida {action} para el agente {agent}")
            return
            
        self.registrar_decision(agent, action)
        
        if self.fase_actual == "MUS":
            if action in [2, 3]:  # Mus (2) o No Mus (3)
                if action == 3:  # No Mus
                    if self.primera_ronda:
                        self.cambiar_mano(agent)
                        self.primera_ronda = False
                    self.fase_actual = "GRANDE"
                    self.reiniciar_para_nueva_fase()
                    return
                else:  # Mus (2)
                    # Verificar si ya votó
                    if agent not in [a for a, v in self.votos_mus]:
                        self.votos_mus.append((agent, action))
                        
                        if len(self.votos_mus) == len(self.agents):
                            # Todos dijeron Mus
                            self.fase_actual = "DESCARTE"
                            self.cartas_a_descartar = {agent: [] for agent in self.agents}
                            self.votos_mus = []
                            # CORRECCIÓN: Reiniciar selector de agentes para la fase de descarte
                            self.agent_selector = agent_selector(self.agents)
                            self.agent_selection = self.agent_selector.next()
                            # Agregar control para saber quién ha confirmado su descarte
                            self.jugadores_confirmaron_descarte = set()
                            print(f"Cambiando a DESCARTE - Nuevo agente: {self.agent_selection}")
                            return
                        else:
                            # Pasar al siguiente jugador
                            old_agent = self.agent_selection
                            self.agent_selection = self.agent_selector.next()
                            print(f"Siguiente jugador en MUS: {old_agent} -> {self.agent_selection}")
                            return
                    return

        elif self.fase_actual == "DESCARTE":
            if 11 <= action <= 14:  # Selección de cartas
                carta_idx = action - 11
                if agent not in self.cartas_a_descartar:
                    self.cartas_a_descartar[agent] = []
                    
                if carta_idx in self.cartas_a_descartar[agent]:
                    self.cartas_a_descartar[agent].remove(carta_idx)
                else:
                    self.cartas_a_descartar[agent].append(carta_idx)
                # No cambiar de agente aquí, el jugador sigue seleccionando
                return

            elif action == 4:  # Confirmar descarte
                self.realizar_descarte(agent)
                self.jugadores_confirmaron_descarte.add(agent)
                
                # Verificar si todos han confirmado su descarte
                if len(self.jugadores_confirmaron_descarte) == len(self.agents):
                    self.fase_actual = "MUS"  # ¿O debería ser "GRANDE"?
                    self.reiniciar_para_nueva_fase()
                    # Limpiar el conjunto de confirmaciones
                    self.jugadores_confirmaron_descarte = set()
                    print(f"Todos confirmaron descarte - Nueva fase: {self.fase_actual}")
                else:
                    # Pasar al siguiente jugador
                    old_agent = self.agent_selection
                    self.agent_selection = self.agent_selector.next()
                    print(f"Siguiente jugador en DESCARTE: {old_agent} -> {self.agent_selection}")
                return
            
        elif self.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
            self.actualizar_jugadores_que_pueden_hablar()
            
            if agent not in self.jugadores_que_pueden_hablar:
                self.siguiente_jugador_que_puede_hablar()
                return
                
            self.procesar_apuesta_corregida(self.fase_actual, agent, action)
            # Después de procesar apuesta, verificar si necesitamos cambiar de agente
            # Esto debería manejarse dentro de procesar_apuesta_corregida
                    
        return self.observe(self.agent_selection)

    def reiniciar_para_nueva_fase(self):
        """Función auxiliar para reiniciar estado entre fases"""
        self.agents = self.possible_agents[:]
        self.agents = self.agents[self.mano:] + self.agents[:self.mano]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        self.apuesta_actual = 0
        self.equipo_apostador = None
        self.jugador_apostador = None
        self.ronda_completa = False
        self.jugadores_pasado = set()
        self.jugadores_hablaron = set()
        self.hay_ordago = False
        self.actualizar_jugadores_que_pueden_hablar()
        # Limpiar confirmaciones de descarte
        if hasattr(self, 'jugadores_confirmaron_descarte'):
            self.jugadores_confirmaron_descarte = set()

        self.equipo_de_jugador = {
            "jugador_0": "equipo_1",
            "jugador_1": "equipo_2", 
            "jugador_2": "equipo_1",
            "jugador_3": "equipo_2"
        }

    def realizar_descarte(self, agent):
        """Mejorar lógica de descarte"""
        if agent not in self.manos:
            return
            
        nuevas_cartas = []
        for i in range(4):
            if i in self.cartas_a_descartar.get(agent, []):
                if self.mazo:
                    nuevas_cartas.append(self.mazo.pop())
                else:
                    self.generar_mazo()
                    if self.mazo:
                        nuevas_cartas.append(self.mazo.pop())
                    else:
                        nuevas_cartas.append(self.manos[agent][i])
            else:
                nuevas_cartas.append(self.manos[agent][i])
        
        self.manos[agent] = nuevas_cartas
        self.cartas_a_descartar[agent] = []
        self.actualizar_declaraciones()
        
    def registrar_decision(self, agent, action):
        """Registra la decisión tomada por un jugador"""
        decisiones = {
            0: "Paso", 1: "Envido", 2: "Mus", 3: "No Mus", 4: "Confirmar",
            5: "No quiero", 6: "Órdago", 7: "Quiero"
        }
        
        if action in decisiones:
            self.ultima_decision[agent] = decisiones[action]
        elif 11 <= action <= 14:
            carta_idx = action - 11
            if carta_idx in self.cartas_a_descartar.get(agent, []):
                self.ultima_decision[agent] = f"Deseleccionar carta {carta_idx+1}"
            else:
                self.ultima_decision[agent] = f"Seleccionar carta {carta_idx+1}"

    def procesar_apuesta_corregida(self, fase, agent, action):
        """Implementa la lógica correcta de apuestas para el juego de Mus"""
        if action not in self.acciones_validas[fase]:
            print(f"Acción {action} no válida para la fase {fase}")
            return
        
        self.jugadores_hablaron.add(agent)
        equipo_actual = self.equipo_de_jugador[agent]
        
        if action == 0:  # Pasar
            self.jugadores_pasado.add(agent)
            
            if self.jugadores_hablaron >= self.jugadores_que_pueden_hablar:
                if self.apuesta_actual == 0:
                    self.determinar_ganador_fase(fase)
                    self.avanzar_fase()
                    return
                elif self.equipo_apostador:
                    equipo_contrario = "equipo_2" if self.equipo_apostador == "equipo_1" else "equipo_1"
                    jugadores_equipo_contrario = set(self.equipos[equipo_contrario]) & self.jugadores_que_pueden_hablar
                    
                    if jugadores_equipo_contrario.issubset(self.jugadores_pasado):
                        self.puntos_equipos[self.equipo_apostador] += self.apuesta_actual
                        self.apuestas[self.equipo_apostador][fase] = self.apuesta_actual
                        self.ganadores_fases[fase] = self.equipo_apostador
                        self.avanzar_fase()
                        return
            
            self.siguiente_jugador_que_puede_hablar()
        
        elif action == 1:  # Envido

            if self.equipo_apostador == equipo_actual:
                print(f"El jugador {agent} no puede envidar porque su compañero ya ha envidado")
                self.siguiente_jugador_que_puede_hablar()
                return
            
            self.apuesta_actual += 2
            self.jugador_apostador = agent
            self.equipo_apostador = self.equipo_de_jugador[agent]
            self.jugadores_pasado = set()
            self.jugadores_hablaron = set()
            self.jugadores_hablaron.add(agent)
            self.siguiente_jugador_que_puede_hablar()
        
        elif action == 5:  # No quiero
            if self.hay_ordago:
                puntos_ganados = max(1, self.apuesta_actual)
                self.puntos_equipos[self.equipo_apostador] += puntos_ganados
                self.apuestas[self.equipo_apostador][fase] = puntos_ganados
                self.ganadores_fases[fase] = self.equipo_apostador
                self.avanzar_fase()
                return
                
            elif self.apuesta_actual > 0 and self.equipo_apostador is not None:
                if equipo_actual != self.equipo_apostador:
                    puntos_ganados = max(1, self.apuesta_actual)
                    self.puntos_equipos[self.equipo_apostador] += puntos_ganados
                    self.apuestas[self.equipo_apostador][fase] = puntos_ganados
                    self.ganadores_fases[fase] = self.equipo_apostador
                    self.avanzar_fase()
                    return
            
            self.siguiente_jugador_que_puede_hablar()
        
        elif action == 6:  # Ordago

            if self.equipo_apostador == equipo_actual:
                print(f"El jugador {agent} no puede hacer órdago porque su compañero ya ha envidado")
                self.siguiente_jugador_que_puede_hablar()
                return
            
            self.apuesta_actual_ordago = 30
            self.hay_ordago = True
            self.jugador_apostador = agent
            self.equipo_apostador = self.equipo_de_jugador[agent]
            self.jugadores_pasado = set()
            self.jugadores_hablaron = set()
            self.jugadores_hablaron.add(agent)
            self.siguiente_jugador_que_puede_hablar()
        
        elif action == 7:  # Quiero
            if self.hay_ordago:
                self.determinar_ganador_fase(fase)
                equipo_ganador = self.ganadores_fases[fase]
                if equipo_ganador:
                    self.apuestas[equipo_ganador][fase] = self.apuesta_actual_ordago
                    self.puntos_equipos[equipo_ganador] = self.apuesta_actual_ordago  # Ganar la partida (30 puntos)
                    self.partidas_ganadas[equipo_ganador] += 1  # Registrar partida ganada
                    print(self.partidas_ganadas[equipo_ganador])
                    self.partida_terminada = True
                    self.hay_ordago = True
                    self.determinar_ganador_global()
                    self.fase_actual = "RECUENTO"
                    for agent in self.agents:
                        self.dones[agent] = True
                return
            elif self.apuesta_actual > 0 and self.equipo_apostador is not None:
                if equipo_actual != self.equipo_apostador:
                    self.determinar_ganador_fase(fase)
                    self.avanzar_fase()
                    return
            
            self.siguiente_jugador_que_puede_hablar()
    
    
    def determinar_ganador_fase(self, fase):
        """Determina el ganador de una fase basado en las reglas del Mus"""
        jugadores_participantes = self.jugadores_que_pueden_hablar if self.jugadores_que_pueden_hablar else set(self.agents)
        
        if fase in ["GRANDE"]:
            # Para GRANDE y CHICA, necesitamos comparar las mejores manos de cada equipo
            mejor_mano_equipo1 = None
            mejor_jugador_equipo1 = None
            mejor_mano_equipo2 = None
            mejor_jugador_equipo2 = None
            
            # Encontrar la mejor mano de cada equipo
            for agent in jugadores_participantes:
                equipo = self.equipo_de_jugador[agent]
                mano = self.manos[agent]
                
                if equipo == "equipo_1":
                    if mejor_mano_equipo1 is None:
                        mejor_mano_equipo1 = mano
                        mejor_jugador_equipo1 = agent
                    elif self.comparar_manos(mano, mejor_mano_equipo1, fase) > 0:
                        mejor_mano_equipo1 = mano
                        mejor_jugador_equipo1 = agent
                else:  # equipo_2
                    if mejor_mano_equipo2 is None:
                        mejor_mano_equipo2 = mano
                        mejor_jugador_equipo2 = agent
                    elif self.comparar_manos(mano, mejor_mano_equipo2, fase) > 0:
                        mejor_mano_equipo2 = mano
                        mejor_jugador_equipo2 = agent
            
            # Comparar las mejores manos entre equipos
            if mejor_mano_equipo1 is not None and mejor_mano_equipo2 is not None:
                resultado = self.comparar_manos(mejor_mano_equipo1, mejor_mano_equipo2, fase)
                if resultado > 0:
                    equipo_ganador = "equipo_1"
                elif resultado < 0:
                    equipo_ganador = "equipo_2"
                else:
                    equipo_ganador = None  # Empate
            elif mejor_mano_equipo1 is not None:
                equipo_ganador = "equipo_1"
            elif mejor_mano_equipo2 is not None:
                equipo_ganador = "equipo_2"
            else:
                equipo_ganador = None

        elif fase in ["CHICA"]:
            # Para GRANDE y CHICA, necesitamos comparar las mejores manos de cada equipo
            mejor_mano_equipo1 = None
            mejor_jugador_equipo1 = None
            mejor_mano_equipo2 = None
            mejor_jugador_equipo2 = None
            
            # Encontrar la mejor mano de cada equipo
            for agent in jugadores_participantes:
                equipo = self.equipo_de_jugador[agent]
                mano = self.manos[agent]
                
                if equipo == "equipo_1":
                    if mejor_mano_equipo1 is None:
                        mejor_mano_equipo1 = mano
                        mejor_jugador_equipo1 = agent
                    elif self.comparar_manos(mano, mejor_mano_equipo1, fase) > 0:
                        mejor_mano_equipo1 = mano
                        mejor_jugador_equipo1 = agent
                else:  # equipo_2
                    if mejor_mano_equipo2 is None:
                        mejor_mano_equipo2 = mano
                        mejor_jugador_equipo2 = agent
                    elif self.comparar_manos(mano, mejor_mano_equipo2, fase) > 0:
                        mejor_mano_equipo2 = mano
                        mejor_jugador_equipo2 = agent
            
            # Comparar las mejores manos entre equipos
            if mejor_mano_equipo1 is not None and mejor_mano_equipo2 is not None:
                resultado = self.comparar_manos(mejor_mano_equipo1, mejor_mano_equipo2, fase)
                if resultado > 0:
                    equipo_ganador = "equipo_1"
                elif resultado < 0:
                    equipo_ganador = "equipo_2"
                else:
                    equipo_ganador = None  # Empate
            elif mejor_mano_equipo1 is not None:
                equipo_ganador = "equipo_1"
            elif mejor_mano_equipo2 is not None:
                equipo_ganador = "equipo_2"
            else:
                equipo_ganador = None
                
        elif fase == "PARES":
            # Para PARES, suma los puntos de pares de cada equipo
            puntos_equipo1 = 0
            puntos_equipo2 = 0
            
            for agent in jugadores_participantes:
                if not self.declaraciones_pares.get(agent, False):
                    continue
                    
                equipo = self.equipo_de_jugador[agent]
                mano = self.manos[agent]
                puntos_pares = self.calcular_puntos_pares_jugador(mano)
                
                if equipo == "equipo_1":
                    puntos_equipo1 += puntos_pares
                else:
                    puntos_equipo2 += puntos_pares
            
            if puntos_equipo1 > puntos_equipo2:
                equipo_ganador = "equipo_1"
            elif puntos_equipo2 > puntos_equipo1:
                equipo_ganador = "equipo_2"
            else:
                equipo_ganador = None
                
        elif fase == "JUEGO":
            # Para JUEGO, encontrar el mejor valor de cada equipo
            mejor_valor_equipo1 = 0
            mejor_valor_equipo2 = 0
            hay_juego_equipo1 = False
            hay_juego_equipo2 = False
            
            for agent in jugadores_participantes:
                equipo = self.equipo_de_jugador[agent]
                valor_juego = self.calcular_valor_juego(self.manos[agent])
                tiene_juego = valor_juego >= 31
                
                if equipo == "equipo_1":
                    if tiene_juego and (not hay_juego_equipo1 or valor_juego > mejor_valor_equipo1):
                        mejor_valor_equipo1 = valor_juego
                        hay_juego_equipo1 = True
                    elif not hay_juego_equipo1 and valor_juego > mejor_valor_equipo1:
                        mejor_valor_equipo1 = valor_juego
                else:  # equipo_2
                    if tiene_juego and (not hay_juego_equipo2 or valor_juego > mejor_valor_equipo2):
                        mejor_valor_equipo2 = valor_juego
                        hay_juego_equipo2 = True
                    elif not hay_juego_equipo2 and valor_juego > mejor_valor_equipo2:
                        mejor_valor_equipo2 = valor_juego
            
            # Si algún equipo tiene juego (>=31) y el otro no, gana el que tiene juego
            if hay_juego_equipo1 and not hay_juego_equipo2:
                equipo_ganador = "equipo_1"
            elif hay_juego_equipo2 and not hay_juego_equipo1:
                equipo_ganador = "equipo_2"
            elif hay_juego_equipo1 and hay_juego_equipo2:
                # Ambos tienen juego, gana el mayor
                if mejor_valor_equipo1 > mejor_valor_equipo2:
                    equipo_ganador = "equipo_1"
                elif mejor_valor_equipo2 > mejor_valor_equipo1:
                    equipo_ganador = "equipo_2"
                else:
                    equipo_ganador = None
            else:
                # Nadie tiene juego, se juega "al punto" (quien se acerca más a 30)
                diff_equipo1 = abs(30 - mejor_valor_equipo1)
                diff_equipo2 = abs(30 - mejor_valor_equipo2)
                
                if diff_equipo1 < diff_equipo2:
                    equipo_ganador = "equipo_1"
                elif diff_equipo2 < diff_equipo1:
                    equipo_ganador = "equipo_2"
                else:
                    equipo_ganador = None
        else:
            equipo_ganador = None
        
        if equipo_ganador is None:
            print(f"Empate en {fase} - El equipo de la mano (jugador_{self.mano}) gana la apuesta")
            
            # Determinar el equipo de la mano
            jugador_mano = f"jugador_{self.mano}"
            equipo_ganador = self.equipo_de_jugador[jugador_mano]
            print(f"Equipo ganador por mano: {equipo_ganador}")

        jugadores_participantes = self.jugadores_que_pueden_hablar if self.jugadores_que_pueden_hablar else set(self.agents)
        
        # Asignar puntos al equipo ganador
        if equipo_ganador:
            self.ganadores_fases[fase] = equipo_ganador
            
            # Calcular puntos ganados
            if self.apuesta_actual > 0:
                puntos_ganados = self.apuesta_actual
            else:
                puntos_ganados = 1  # Puntos base
                
            # Agregar puntos adicionales según la fase
            if fase == "PARES":
                puntos_adicionales = self.calcular_puntos_pares(equipo_ganador)
                puntos_ganados += puntos_adicionales
            elif fase == "JUEGO":
                puntos_adicionales = self.calcular_puntos_juego(equipo_ganador)
                puntos_ganados += puntos_adicionales
                
            self.puntos_equipos[equipo_ganador] += puntos_ganados
            self.apuestas[equipo_ganador][fase] = puntos_ganados
            
            print(f"Equipo {equipo_ganador} gana {puntos_ganados} puntos en {fase}")
        else:
            self.ganadores_fases[fase] = None
            print(f"Empate en la fase {fase}")
        
        # Reiniciar variables de apuesta
        self.apuesta_actual = 0
        self.equipo_apostador = None
        self.jugador_apostador = None
        self.hay_ordago = False

    def determinar_jugador_mas_proximo(self, jugadores):
        """Determina qué jugador está más cerca del mano en el orden de juego"""
        mano = self.mano
        orden_juego = self.agents  # El orden actual de juego (mano primero)
        
        # Encontrar posiciones en el orden de juego
        posiciones = {}
        for jugador in jugadores:
            try:
                idx = orden_juego.index(jugador)
                distancia = (idx - 0) % len(orden_juego)  # Distancia desde el mano
                posiciones[jugador] = distancia
            except ValueError:
                posiciones[jugador] = float('inf')
        
        # Encontrar jugador con menor distancia
        jugador_mas_cercano = min(posiciones, key=posiciones.get)
        print(f"Jugador más cercano al mano: {jugador_mas_cercano} (distancia: {posiciones[jugador_mas_cercano]})")
        return jugador_mas_cercano

    def calcular_puntos_pares_jugador(self, mano):
        """Calcula los puntos de pares para un jugador individual"""
        if not mano:
            return 0
            
        valores = [carta[0] for carta in mano]
        conteo = {}
        for valor in valores:
            conteo[valor] = conteo.get(valor, 0) + 1
        
        print(f"Conteo de valores: {conteo}")
        # Contar diferentes tipos de pares
        if any(c == 4 for c in conteo.values()):
            return 3  # Cuatro iguales
        elif any(c == 3 for c in conteo.values()):
            return 2  # Tres iguales
        elif list(conteo.values()).count(2) >= 2:
            return 3  # Dos pares
        elif any(c == 2 for c in conteo.values()):
            return 1  # Un par
        
        return 0

    def avanzar_fase(self):
        """Avanza a la siguiente fase del juego"""
        current_idx = self.fases.index(self.fase_actual)
        if current_idx < len(self.fases) - 1:
            next_idx = current_idx + 1
            next_fase = self.fases[next_idx]
            
            if next_fase == "PARES" and not any(self.declaraciones_pares.values()):
                if "JUEGO" in self.fases:
                    next_fase = "JUEGO"
                else:
                    next_fase = "RECUENTO"
            
            if next_fase == "JUEGO" and not any(self.declaraciones_juego.values()):
                # Solo saltar si no hay cartas (caso extremo)
                if all(not mano for mano in self.manos.values()):
                    next_fase = "RECUENTO"
                else:
                    # Permitir jugar al punto aunque nadie tenga juego
                    next_fase = "JUEGO"
                
            self.fase_actual = next_fase
            self.reiniciar_para_nueva_fase()
            
            if self.agent_selection not in self.jugadores_que_pueden_hablar:
                self.siguiente_jugador_que_puede_hablar()
                
            if self.fase_actual == "RECUENTO":
                self.mano = (self.mano + 1) % 4
                print("Avanzando a la fase de RECUENTO 1")
                self.determinar_ganador_global()
        else:
            self.fase_actual = "RECUENTO"
            print("Avanzando a la fase de RECUENTO 2")
            self.ronda_completa = True
            for agent in self.agents:
                self.dones[agent] = True
            self.determinar_ganador_global()

    def determinar_ganador_global(self):
        """Determina el ganador de la partida actual y verifica si hay ganador del juego (mejor de 3)"""
        equipo_ganador_partida = max(self.puntos_equipos.items(), key=lambda x: x[1])[0]
        global fin, ronda_completa
        if self.puntos_equipos[equipo_ganador_partida] >= 30:
            ronda_completa = True
            if not self.hay_ordago:
                self.partidas_ganadas[equipo_ganador_partida] += 1
            self.hay_ordago = False
            if self.partidas_ganadas[equipo_ganador_partida] >= 2:
                fin = True

        print(ronda_completa, fin, self.puntos_equipos[equipo_ganador_partida])
        return equipo_ganador_partida

    def render(self):
        print(f"Fase: {self.fase_actual}")
        print(f"Jugadores que pueden hablar: {self.jugadores_que_pueden_hablar}")
        for ag in self.agents:
            print(f"{ag}: {self.manos[ag]} descarta {self.cartas_a_descartar.get(ag, [])}")
        if self.fase_actual == "MUS":
            print(f"Votos MUS: {self.votos_mus}")
        elif self.fase_actual == "PARES":
            print(f"Declaraciones PARES: {self.declaraciones_pares}")
        elif self.fase_actual == "JUEGO":
            print(f"Declaraciones JUEGO: {self.declaraciones_juego}")
            print(f"Valores JUEGO: {self.valores_juego}")
        elif self.fase_actual == "RECUENTO":
            print(f"Puntos equipos: {self.puntos_equipos}")
            print(f"Ganadores fases: {self.ganadores_fases}")

    def close(self):
        pass

    def calcular_puntos_pares(self, equipo):
        """Mejorar cálculo de puntos por pares"""
        puntos_totales = 0
        
        for jugador in self.equipos.get(equipo, []):
            if not self.declaraciones_pares.get(jugador, False):
                continue
                
            if jugador not in self.manos:
                continue
                
            mano = self.manos[jugador]
            valores = [carta[0] for carta in mano]
            conteo = {}
            for valor in valores:
                conteo[valor] = conteo.get(valor, 0) + 1
            
            if any(c == 4 for c in conteo.values()):
                puntos_totales += 3
            elif list(conteo.values()).count(2) >= 2:
                puntos_totales += 3
            elif any(c == 3 for c in conteo.values()):
                puntos_totales += 2
            elif any(c == 2 for c in conteo.values()):
                puntos_totales += 1
        
        return max(1, puntos_totales)

    def calcular_puntos_juego(self, equipo):
        """Mejorar cálculo de puntos por juego"""
        puntos_totales = 0
        
        for jugador in self.equipos.get(equipo, []):
            if not self.declaraciones_juego.get(jugador, False):
                continue
                
            if jugador not in self.valores_juego:
                continue
                
            valor_juego = self.valores_juego[jugador]
            
            if valor_juego == 31:
                puntos_totales += 3
            elif valor_juego == 32:
                puntos_totales += 2
            else:
                puntos_totales += 1
        
        return max(1, puntos_totales)

def env():
    return MusEnv()
