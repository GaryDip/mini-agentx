import json
from pathlib import Path
import tempfile
import unittest
from mini_agentx.simulation.interest import score_pairs, validate_ratings


class Client:
    model='mock';config={'max_attempts':2}
    def __init__(self): self.calls=0;self.inputs=[]
    def complete(self,messages):
        self.calls+=1
        content=json.loads(messages[1]['content']);self.inputs.append(content)
        return {'ratings':[{'user_id':case['user_id'],'item_id':item['item_id'],'interest_probability':.8,'reason':'类型符合画像'} for case in content['cases'] for item in case['items']]},{}


class InterestTest(unittest.TestCase):
    def test_blind_input_cache_and_profile_version(self):
        with tempfile.TemporaryDirectory() as folder:
            profiles={0:{'user_id':0,'summary':'喜剧偏好','preferences':[],'uncertainties':['模拟']}}
            catalog={'1':{'item_id':1,'title':'movie','genres':['Comedy']}}
            client=Client()
            first,_=score_pairs(profiles,catalog,[(0,1)],Path(folder),client)
            second,audit=score_pairs(profiles,catalog,[(0,1)],Path(folder),client)
            self.assertEqual(first,second);self.assertEqual(client.calls,1)
            self.assertEqual(audit['cached_pairs'],1)
            case=client.inputs[0]['cases'][0]
            self.assertEqual(set(case),{'user_id','profile','items'})
            profiles[0]['summary']='改了画像'
            score_pairs(profiles,catalog,[(0,1)],Path(folder),client)
            self.assertEqual(client.calls,2)

    def test_invalid_probability_duplicate_and_missing_pairs(self):
        rating={'user_id':0,'item_id':1,'interest_probability':True,'reason':'test'}
        with self.assertRaises(ValueError):validate_ratings({'ratings':[rating]},[(0,1)])
        rating['interest_probability']=float('nan')
        with self.assertRaises(ValueError):validate_ratings({'ratings':[rating]},[(0,1)])
        rating['interest_probability']=.5
        with self.assertRaises(ValueError):validate_ratings({'ratings':[rating,rating]},[(0,1)])
        with self.assertRaises(ValueError):validate_ratings({'ratings':[rating]},[(0,1),(0,2)])

    def test_simulator_preserves_fixed_population_ids(self):
        from unittest.mock import patch
        from mini_agentx.abtest.simulator import simulate, load_ab_config, assign_arm
        config=load_ab_config('configs/abtest.toml')
        config['feedback']={'backend':'llm_profiles'}
        users=[next(user for user in range(100,200) if assign_arm(user,42)==arm) for arm in ('A','B')]
        oracle=lambda user,arm,session:(7,.4)
        with tempfile.TemporaryDirectory() as folder, patch('mini_agentx.simulation.interest.profile_feedback',return_value=(oracle,users,{'feedback_oracle':'fixed profiles'})):
            report=simulate(config,'null',folder,Path(folder)/'a',Path(folder)/'b')
            import csv
            with (Path(folder)/report['experiment_id']/'events.csv').open() as file:
                events=list(csv.DictReader(file))
            self.assertEqual({int(row['user_id']) for row in events},set(users))
            self.assertEqual(report['observation_window']['planned_users'],2)
            self.assertEqual(len(events),40)
