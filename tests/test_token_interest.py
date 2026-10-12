import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from mini_agentx.simulation.token_interest import parse_binary, score_token_pairs


def output(yes=.8,no=.2):
    answer='yes' if yes>=no else 'no'
    return {'answer':answer,'logprobs':{'content':[{'token':answer,'top_logprobs':[{'token':'yes','logprob':math.log(yes)},{'token':'no','logprob':math.log(no)}]}]}}


class TokenTest(unittest.TestCase):
    def test_signed_score_and_missing_not_zero(self):
        self.assertAlmostEqual(parse_binary(output())['signed_score'],.8)
        self.assertAlmostEqual(parse_binary(output(.1,.9))['signed_score'],-.9)
        missing=output();missing['logprobs']['content'][0]['top_logprobs'].pop()
        with self.assertRaisesRegex(ValueError,'缺少'):parse_binary(missing)
        invalid=output();invalid['answer']='yes because';invalid['logprobs']['content'].append({})
        with self.assertRaises(ValueError):parse_binary(invalid)

    def test_bounded_concurrency_and_cache_reuse(self):
        class Client:
            model='mock'
            def __init__(self):self.lock=threading.Lock();self.active=0;self.peak=0;self.calls=0
            def binary_logprobs(self,messages):
                content=json.loads(messages[1]['content'])
                assert set(content)=={'user_id','profile','item'}
                with self.lock:self.active+=1;self.calls+=1;self.peak=max(self.peak,self.active)
                time.sleep(.03)
                with self.lock:self.active-=1
                return output(),{}
        with tempfile.TemporaryDirectory() as folder:
            profiles={i:{'user_id':i,'summary':'test'} for i in range(6)};catalog={'1':{'item_id':1,'genres':['Comedy']}}
            client=Client();pairs=[(i,1) for i in profiles]
            result,state=score_token_pairs(profiles,catalog,pairs,folder,client,concurrency=3,max_calls=6)
            self.assertEqual(state['model_calls'],6);self.assertEqual(client.peak,3)
            again,second=score_token_pairs(profiles,catalog,pairs*2,folder,client,concurrency=3,max_calls=1)
            self.assertEqual(result,again);self.assertEqual(second['model_calls'],0);self.assertEqual(client.calls,6)

    def test_budget_preserves_successful_cache(self):
        class Client:
            model='mock'
            def binary_logprobs(self,messages):return output(),{}
        with tempfile.TemporaryDirectory() as folder:
            profiles={0:{'summary':'test'}};catalog={str(i):{'item_id':i} for i in range(3)}
            with self.assertRaisesRegex(ValueError,'预算'):
                score_token_pairs(profiles,catalog,[(0,i) for i in range(3)],folder,Client(),max_calls=1,concurrency=1)
            self.assertEqual(len(list((Path(folder)/'interest-token-cache').glob('*.json'))),1)
