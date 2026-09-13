import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import io
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from model_zoo.core import tensor_hash
from model_zoo.loaders.unified import Classifier, SharpnessPreActResNet18, pair_relation
import model_zoo.core as core

class Contracts(unittest.TestCase):
    def test_native_preprocessing_matches_manual(self):
        meta={'normalization_mean':[.1,.2,.3],'normalization_std':[.2,.4,.5]}
        x=torch.rand(2,3,32,32)
        model=Classifier(nn.Identity(),meta)
        expected=(x-torch.tensor([.1,.2,.3]).view(1,3,1,1))/torch.tensor([.2,.4,.5]).view(1,3,1,1)
        torch.testing.assert_close(model(x),expected)
        none=Classifier(nn.Identity(),meta,'none')
        torch.testing.assert_close(none(x),x)
        common=Classifier(nn.Identity(),meta,'common',[0,0,0],[1,1,1])
        torch.testing.assert_close(common(x),x)
        with self.assertRaises(ValueError):Classifier(nn.Identity(),meta,'common')
        with self.assertRaises(ValueError):model(torch.zeros(2,3,32,32,dtype=torch.uint8))

    def test_lineage_topology_and_family_are_separate(self):
        a=dict(lineage_id='one',topology_id='arch1',architecture_family='ResNet')
        b=dict(lineage_id='two',topology_id='arch1',architecture_family='ResNet')
        self.assertTrue(pair_relation(a,b)['hard_negative'])
        self.assertFalse(pair_relation(a,b)['same_lineage'])
        self.assertTrue(pair_relation(a,a)['same_lineage'])
        c=dict(lineage_id='three',topology_id='arch2',architecture_family='ResNet')
        self.assertTrue(pair_relation(a,c)['different_architecture'])
        self.assertTrue(pair_relation(a,c)['same_family'])
        descendant=dict(a,topology_id='arch1-pruned')
        self.assertTrue(pair_relation(a,descendant)['same_lineage'])

    def test_hash_order_independence_and_buffer_sensitivity(self):
        a={'weight':torch.arange(6.).reshape(2,3),'bn.num_batches_tracked':torch.tensor(3)}
        b=dict(reversed(list(a.items())))
        self.assertEqual(tensor_hash(a),tensor_hash(b))
        b['bn.num_batches_tracked']=torch.tensor(4)
        self.assertNotEqual(tensor_hash(a),tensor_hash(b))
        self.assertNotEqual(tensor_hash({'x':torch.ones(2)}),tensor_hash({'x':torch.ones(1,2)}))

    def test_b2_network_contract(self):
        model = SharpnessPreActResNet18().eval()
        with torch.inference_mode():
            output = model(torch.rand(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 10))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 11_172_170)

    def test_range_ignored_never_reads_archive_body(self):
        class Response:
            status=200
            headers={'Content-Length':str(2_328_714_690_560)}
            read_called=False
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,n):self.read_called=True;raise AssertionError('Body must not be read')
        response=Response()
        with tempfile.TemporaryDirectory() as tmp, patch.object(core,'ROOT',Path(tmp)), patch('urllib.request.urlopen',return_value=response):
            with self.assertRaisesRegex(RuntimeError,'Selective download refused'):
                core.fetch('https://example.invalid/archive','test',start=100,length=512)
            self.assertFalse(response.read_called)

    def test_range_wrong_offsets_rejected(self):
        class Response:
            status=206
            headers={'Content-Range':'bytes 0-511/1000000'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,n):raise AssertionError('Body must not be read')
        with tempfile.TemporaryDirectory() as tmp, patch.object(core,'ROOT',Path(tmp)), patch('urllib.request.urlopen',return_value=Response()):
            with self.assertRaisesRegex(RuntimeError,'Content-Range mismatch'):
                core.fetch('https://example.invalid/archive','test',start=100,length=512)

    def test_source_budget_blocks_network_request(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(core,'ROOT',Path(tmp)), patch('urllib.request.urlopen') as request:
            core.write_json(Path(tmp)/'metadata/transfer_ledger.json',{'test':core.SOURCE_LIMIT})
            with self.assertRaisesRegex(RuntimeError,'Source transfer budget'):
                core.fetch('https://example.invalid/archive','test',start=100,length=512)
            request.assert_not_called()

if __name__=='__main__':unittest.main()
