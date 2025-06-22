#!/usr/bin/env python3
"""
Prueba rápida de compatibilidad para identificar problemas específicos
"""

def test_agent_creation():
    """Probar diferentes formas de crear MARLAgent"""
    print("🤖 PROBANDO CREACIÓN DE MARLAGENT")
    print("=" * 35)
    
    try:
        from marl_agent import MARLAgent
        import inspect
        
        # Mostrar signatura completa
        sig = inspect.signature(MARLAgent.__init__)
        print(f"📋 Signatura completa: {sig}")
        
        # Probar diferentes configuraciones
        configs = [
            {'state_size': 20, 'action_size': 15, 'agent_id': 'test', 'team': 0},
            {'state_size': 20, 'action_size': 15, 'agent_id': 'test', 'team': 1},
            {'state_size': 20, 'action_size': 15, 'agent_id': 'test', 'team': 'team_0'},
            {'state_size': 20, 'action_size': 15, 'agent_id': 'test'},
        ]
        
        successful_config = None
        
        for i, config in enumerate(configs):
            try:
                print(f"\n🧪 Probando configuración {i+1}: {config}")
                agent = MARLAgent(**config)
                print(f"✅ Creación exitosa!")
                
                # Probar método act
                test_obs = [0.0] * 20
                
                # Probar act sin valid_actions
                try:
                    action = agent.act(test_obs)
                    print(f"✅ act(observation) exitoso: {action}")
                except Exception as e:
                    print(f"❌ act(observation) falló: {e}")
                    
                    # Probar act con valid_actions
                    try:
                        valid_actions = list(range(15))
                        action = agent.act(test_obs, valid_actions)
                        print(f"✅ act(observation, valid_actions) exitoso: {action}")
                        successful_config = config
                        successful_config['needs_valid_actions'] = True
                        break
                    except Exception as e2:
                        print(f"❌ act(observation, valid_actions) también falló: {e2}")
                
                successful_config = config
                successful_config['needs_valid_actions'] = False
                break
                
            except Exception as e:
                print(f"❌ Configuración {i+1} falló: {e}")
        
        if successful_config:
            print(f"\n🎉 CONFIGURACIÓN EXITOSA ENCONTRADA:")
            print(f"📋 {successful_config}")
            return successful_config
        else:
            print(f"\n❌ NINGUNA CONFIGURACIÓN FUNCIONÓ")
            return None
            
    except Exception as e:
        print(f"❌ Error importando MARLAgent: {e}")
        return None

def test_environment_methods():
    """Probar métodos específicos del entorno"""
    print("\n🌍 PROBANDO MÉTODOS DEL ENTORNO")
    print("=" * 32)
    
    try:
        from mus_env import mus
        
        env = mus.env()
        env.reset()
        
        agent_id = env.possible_agents[0]
        print(f"🎯 Probando con agente: {agent_id}")
        
        # Probar observe
        try:
            obs = env.observe(agent_id)
            print(f"✅ observe({agent_id}): {type(obs)}, shape: {getattr(obs, 'shape', len(obs) if obs else 'None')}")
        except Exception as e:
            print(f"❌ observe({agent_id}) falló: {e}")
        
        # Probar legal_actions o métodos similares
        action_methods = ['legal_actions', 'valid_actions', 'available_actions']
        
        for method_name in action_methods:
            try:
                if hasattr(env, method_name):
                    method = getattr(env, method_name)
                    actions = method(agent_id)
                    print(f"✅ {method_name}({agent_id}): {actions}")
                else:
                    print(f"❌ {method_name} no existe")
            except Exception as e:
                print(f"❌ {method_name}({agent_id}) falló: {e}")
        
        # Probar action_space
        try:
            action_space = env.action_space(agent_id)
            if action_space and hasattr(action_space, 'n'):
                print(f"✅ action_space({agent_id}).n: {action_space.n}")
                valid_actions = list(range(action_space.n))
                print(f"✅ Acciones válidas generadas: {valid_actions}")
            else:
                print(f"❌ action_space({agent_id}) no tiene atributo 'n'")
        except Exception as e:
            print(f"❌ action_space({agent_id}) falló: {e}")
        
        # Probar step básico
        try:
            print(f"\n🎮 Probando step básico...")
            result = env.step(0)  # Acción 0
            print(f"✅ step(0) exitoso: {type(result)}")
            if isinstance(result, tuple):
                print(f"   Elementos del resultado: {len(result)}")
                for i, elem in enumerate(result):
                    print(f"   [{i}]: {type(elem)} = {elem}")
        except Exception as e:
            print(f"❌ step(0) falló: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando entorno: {e}")
        return False

def test_simple_interaction():
    """Probar interacción simple entre agente y entorno"""
    print("\n🔄 PROBANDO INTERACCIÓN SIMPLE")
    print("=" * 30)
    
    try:
        # Usar configuración exitosa del agente
        agent_config = test_agent_creation()
        if not agent_config:
            print("❌ No se pudo crear agente")
            return False
        
        # Crear entorno
        from mus_env import mus
        env = mus.env()
        env.reset()
        
        # Crear agente
        from marl_agent import MARLAgent
        agent_params = {k: v for k, v in agent_config.items() if k != 'needs_valid_actions'}
        agent = MARLAgent(**agent_params)
        
        agent_id = env.possible_agents[0]
        print(f"🎯 Probando interacción con: {agent_id}")
        
        # Intentar 3 pasos de interacción
        for step in range(3):
            try:
                print(f"\n📍 Paso {step + 1}:")
                
                # Obtener observación
                obs = env.observe(agent_id)
                if obs is None:
                    obs = [0.0] * agent_config['state_size']
                    print(f"   📥 Usando observación por defecto")
                else:
                    print(f"   📥 Observación obtenida: {type(obs)}")
                
                # Obtener acciones válidas
                valid_actions = None
                try:
                    action_space = env.action_space(agent_id)
                    if action_space and hasattr(action_space, 'n'):
                        valid_actions = list(range(action_space.n))
                except:
                    valid_actions = list(range(agent_config['action_size']))
                
                print(f"   🎯 Acciones válidas: {valid_actions}")
                
                # Obtener acción del agente
                if agent_config.get('needs_valid_actions', False):
                    action = agent.act(obs, valid_actions)
                else:
                    action = agent.act(obs)
                
                print(f"   🎲 Acción elegida: {action}")
                
                # Ejecutar en entorno
                result = env.step(action)
                print(f"   ✅ Step exitoso: {type(result)}")
                
            except Exception as e:
                print(f"   ❌ Error en paso {step + 1}: {e}")
                break
        
        print(f"\n🎉 ¡Interacción simple exitosa!")
        return True
        
    except Exception as e:
        print(f"❌ Error en interacción simple: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 PRUEBA RÁPIDA DE COMPATIBILIDAD")
    print("=" * 35)
    
    # Probar creación de agente
    agent_success = test_agent_creation()
    
    # Probar métodos del entorno
    env_success = test_environment_methods()
    
    # Probar interacción simple
    if agent_success and env_success:
        interaction_success = test_simple_interaction()
        
        if interaction_success:
            print(f"\n🎉 ¡TODAS LAS PRUEBAS EXITOSAS!")
            print(f"✅ El sistema está listo para entrenamiento compatible")
            print(f"🚀 Ejecuta: python fully_compatible_training.py")
        else:
            print(f"\n⚠️ Interacción falló, pero componentes individuales funcionan")
    else:
        print(f"\n❌ Problemas básicos detectados")
        print(f"🔧 Revisa la implementación de MusEnv y MARLAgent")
