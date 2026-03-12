import sys
sys.path.append('.')
from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from agents.greedy_attacker import GreedyAttacker
from agents.greedy_defender import GreedyDefender
from core.actions import ActionType

rng = CentralizedRNG(seed=42)
env = EnvironmentEngine(rng)
env.reset()
attacker = GreedyAttacker('atk_1')
defender = GreedyDefender('def_1')

for step in range(8):
    obs_a = env.get_observation_by_id('atk_1')
    act_d = ActionType.DEFENDER_NO_OP
    import core.actions
    act_d = core.actions.DefenderAction('def_1', ActionType.DEFENDER_NO_OP)
    
    if step == 0: act_a_type = core.actions.ActionType.MOVE_LATERAL; target = 'dmz'
    elif step == 1: act_a_type = core.actions.ActionType.EXPLOIT; target = 'dmz'
    elif step == 2: act_a_type = core.actions.ActionType.MOVE_LATERAL; target = 'internal'
    elif step == 3: act_a_type = core.actions.ActionType.EXPLOIT; target = 'internal'
    elif step == 4: act_a_type = core.actions.ActionType.MOVE_LATERAL; target = 'data'
    elif step == 5: act_a_type = core.actions.ActionType.EXPLOIT; target = 'data'
    elif step == 6: act_a_type = core.actions.ActionType.EXPLOIT; target = 'data'
    else: act_a_type = core.actions.ActionType.ATTACKER_NO_OP; target = None
    
    act_a = core.actions.AttackerAction('atk_1', act_a_type, target_node=target, metadata={"vuln_id": "UNKNOWN", "probability": 1.0})
    
    print(f'-- Step {step}: Atk={act_a.action_type.name} {target} --')
    env.step(act_a, act_d)
    
    for nn in ['dmz', 'internal', 'data']:
        try:
            print(f'{nn} priv: {env._state.vulnerability_registry.get_privilege(nn).name}')
        except: pass
