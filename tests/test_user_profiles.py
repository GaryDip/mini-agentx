import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from mini_agentx.simulation.profiles import prepare_profiles, generate_profiles, validate_profiles


class ProfileTest(unittest.TestCase):
    def fixture(self, root):
        data=root/'processed/movielens-100k-v1';data.mkdir(parents=True)
        (data/'train.csv').write_text('user_id,item_id,timestamp\n0,0,1\n0,1,2\n0,2,3\n1,0,1\n1,1,2\n1,2,3\n')
        (data/'id_maps.json').write_text(json.dumps({'item_original_ids':[1,2,3],'user_original_ids':[10,20]}))
        (data/'manifest.json').write_text(json.dumps({'dataset_version':'test-v1','file_sha256':{name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in ('train.csv','id_maps.json')}}))
        # 留出文件不可解析，画像准备仍须成功；不访问原始评分文件。
        (data/'validation.csv').write_text('INVALID HOLDOUT')
        (data/'test.csv').write_text('INVALID HOLDOUT')
        raw=root/'raw/ml-100k';raw.mkdir(parents=True)
        (raw/'u.genre').write_text('\n'.join(f'Genre{i}|{i}' for i in range(19)))
        (raw/'u.item').write_text('\n'.join('|'.join([str(i),f'Movie{i}','date','','url']+['1']+['0']*18) for i in range(1,4)))
        return {'paths':{'processed_data':str(root/'processed'),'raw_data':str(root/'raw'),'runs':str(root/'runs')}}

    def test_recent_history_mapping_cache_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=self.fixture(root)
            state=prepare_profiles(config,count=2,min_history=2,history_limit=2)
            directory=Path(state['directory']);users=json.loads((directory/'users.json').read_text())
            self.assertEqual([i['item_id'] for i in users[0]['history']],[1,2])
            self.assertEqual(users[0]['history'][0]['title'],'Movie2')
            class Client:
                config={'max_attempts':2,'model':'mock'}
                calls=0
                def complete(self,messages):
                    self.calls+=1
                    return {'profiles':[{'user_id':u['user_id'],'summary':'偏好测试类型','preferences':[{'genre':'Genre0','evidence_item_ids':[u['history'][0]['item_id']]}],'uncertainties':['仅正反馈']} for u in json.loads(messages[1]['content'])['users']]},{}
            client=Client();result=generate_profiles(directory,client)
            self.assertEqual(result['completed_users'],2)
            generate_profiles(directory,client)
            self.assertEqual(client.calls,1)
            self.assertEqual(len(list((directory/'profiles').glob('user-*.json'))),2)
            (directory/'users.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'修改'):
                generate_profiles(directory,client)

    def test_wrong_user_movie_or_genre_rejected(self):
        users=[{'user_id':0,'history':[{'item_id':1,'genres':['Comedy']}]}]
        profile={'user_id':0,'summary':'喜欢喜剧','preferences':[{'genre':'Comedy','evidence_item_ids':[999]}],'uncertainties':['仅正反馈']}
        with self.assertRaisesRegex(ValueError,'历史'):
            validate_profiles({'profiles':[profile]},users)
        profile['preferences'][0]={'genre':'Horror','evidence_item_ids':[1]}
        with self.assertRaisesRegex(ValueError,'类型'):
            validate_profiles({'profiles':[profile]},users)

    def test_insufficient_users_does_not_duplicate_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError,'不足'):
                prepare_profiles(config,count=3,min_history=2)
