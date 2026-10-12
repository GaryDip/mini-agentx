import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mini_agentx.recommender.interface import check_contract, model_class
from mini_agentx.recommender.train import train


def load_example(name):
    path=Path(__file__).resolve().parents[1]/'examples/models'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('example_'+name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path


class InterfaceTest(unittest.TestCase):
    def settings(self):
        return {'dimensions':4,'learning_rate':.001,'l2':.0001,'batch_size':4,'threads':1,'epochs':1,'loss':'bpr','negative_sampling':'uniform'}
    def test_embedding_free_and_mlp_training_reload(self):
        for name in ('popularity','mlp'):
            module,_=load_example(name);cls=model_class(module)
            self.assertEqual(check_contract(cls,self.settings(),42)['model_protocol'],'model_owned_training_v1')
            instance=cls(3,5,self.settings())
            self.assertFalse(hasattr(instance,'users'));self.assertFalse(hasattr(instance,'items'))

    def test_framework_only_passes_training_and_saves_model_owned_state(self):
        module,path=load_example('popularity')
        original=module.Recommender.fit_epoch
        seen=[]
        def fit(model,pairs,settings,rng):
            seen.extend(pairs);return original(model,pairs,settings,rng)
        with tempfile.TemporaryDirectory() as folder, patch.object(module.Recommender,'fit_epoch',fit), patch('mini_agentx.recommender.train.load_data',return_value=({'users':3,'items':5,'dataset_version':'v1'},{'train':[(0,1),(1,2),(2,3)],'validation':[(0,2)],'test':[(0,4)]})), patch('mini_agentx.recommender.train.ranking_metrics',return_value={'ndcg@10':.1,'recall@10':.2}):
            config={'seed':42,'training':self.settings(),'paths':{'processed_data':folder,'runs':folder}}
            result=train(config,module.Recommender,path)
            report=json.loads((Path(folder)/result['run_id']/'report.json').read_text())
            self.assertEqual(seen,[(0,1),(1,2),(2,3)])
            self.assertEqual(report['model_protocol'],'model_owned_training_v1')
