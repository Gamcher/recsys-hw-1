import unittest
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from models.item_knn import ItemKNN
from models.als import ExplicitALS
from models.custom_als import CustomALS
from models.ease import EASE
from models.fm import FM
from utils.evaluation import topk,ranking_metrics,cohort


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        rows=[]
        for u in range(12):
            for i in rng.choice(20,8,replace=False):
                rows.append((u,int(i),float(1+(u+i)%5),1000+int(i),str(i)))
        self.frame=pd.DataFrame(rows,columns=['user_idx','item_idx','overall','unixReviewTime','asin'])
        self.meta=pd.DataFrame({'asin':[str(i) for i in range(20)],'brand':['a','b']*10,
                                'category':[['Video Games','PC']]*20})

    def test_models_fit_score_and_mask(self):
        for model in [ItemKNN(neighbors=4),ExplicitALS(rank=4,iterations=3),
                      CustomALS(rank=4,iterations=3),EASE(),FM(rank=4,epochs=2)]:
            with self.subTest(model=type(model).__name__):
                model.fit(self.frame,12,20,self.meta)
                scores=model.score([0,1])
                self.assertEqual(scores.shape,(2,20))
                self.assertTrue(np.isfinite(scores).all())
                known=csr_matrix((np.ones(16),(np.repeat([0,1],8),np.tile(np.arange(8),2))),shape=(2,20))
                recs=topk(scores,known)
                self.assertEqual(recs.shape,(2,10))
                self.assertTrue((recs>=8).all())
                self.assertTrue(all(len(set(r))==10 for r in recs))

    def test_als_objective_decreases(self):
        for cls in [ExplicitALS,CustomALS]:
            model=cls(rank=4,iterations=5).fit(self.frame,12,20)
            self.assertTrue((np.diff(model.loss_history)<=1e-6).all())

    def test_ease_solution(self):
        model=EASE(regularization=3).fit(self.frame,12,20)
        self.assertTrue(np.allclose(model.weights.diagonal(),0))
        gram=(model.history.T@model.history).toarray()+3*np.eye(20)
        gradient=gram@model.weights-(model.history.T@model.history).toarray()
        np.fill_diagonal(gradient,0)
        self.assertTrue(np.allclose(gradient,0,atol=1e-8))

    def test_fm_fast_scores_equal_forward(self):
        model=FM(rank=4,epochs=2).fit(self.frame,12,20,self.meta)
        features=torch.tensor(model.encoder.transform([0,1],[3,4]))
        with torch.no_grad():
            direct=model.network(features).numpy()
        np.testing.assert_allclose(model.score([0,1])[[0,1],[3,4]],direct,atol=1e-5)
        self.assertTrue({'brand','platform'}<=set(model.encoder.vocabularies))
        self.assertGreater(len(model.encoder.vocabularies['brand']),1)

    def test_custom_target_changes_values_not_weights(self):
        m=CustomALS(alpha=.5,tau_days=1)
        a=m.target(self.frame)
        self.assertFalse(np.allclose(a,self.frame.overall))
        self.assertTrue((np.abs(a)<=1).all())
        self.assertTrue((a[self.frame.overall==3]==0).all())

    def test_metrics_known_values_and_ties(self):
        recs=np.arange(10)[None,:]
        self.assertEqual(ranking_metrics(recs,[{0}]),{'NDCG@10':1.,'Recall@10':1.})
        r=ranking_metrics(recs,[{1,15}])
        self.assertAlmostEqual(r['Recall@10'],.5)
        self.assertAlmostEqual(r['NDCG@10'],(1/np.log2(3))/(1+1/np.log2(3)))
        np.testing.assert_array_equal(topk(np.zeros((1,20)),csr_matrix((1,20))),recs)
        with self.assertRaises(ValueError):
            topk(np.full((1,20),np.nan),csr_matrix((1,20)))

    def test_cohort_excludes_known_and_low_ratings(self):
        truth=cohort(self.frame,self.frame.iloc[:1],20)
        self.assertNotIn(int(self.frame.iloc[0].item_idx),truth.get(0,set()))
        for u,items in truth.items():
            self.assertTrue(all(self.frame.loc[(self.frame.user_idx == u) & (self.frame.item_idx == i), 'overall'].iloc[0]>=4 for i in items))

    def test_metadata_vocab_ignores_nontraining_items(self):
        from utils.features import ItemFeatures
        a=ItemFeatures().fit(self.frame,self.meta,12,20)
        extra=pd.DataFrame({'asin':['not_in_train'],'brand':['future_brand'],
                            'category':[['Video Games','future_platform']]})
        b=ItemFeatures().fit(self.frame,pd.concat([self.meta,extra]),12,20)
        self.assertEqual(a.vocabularies,b.vocabularies)
        np.testing.assert_array_equal(a.item_fields,b.item_fields)

    def test_zero_custom_targets_remain_observed(self):
        frame=self.frame.copy()
        frame.overall=3.
        model=CustomALS(rank=4,iterations=2).fit(frame,12,20)
        self.assertTrue(np.allclose(model.score([0,1]),0))

    def test_evaluation_rejects_malformed_inputs(self):
        from utils.evaluation import evaluate
        for recs, truth in [(np.zeros((2,10),dtype=int),[{1}]),
                            (np.arange(10)[None,:],[set()]),
                            (np.zeros((1,10),dtype=int),[{1}])]:
            with self.assertRaises(ValueError):
                ranking_metrics(recs,truth)
        with self.assertRaises(ValueError):
            topk(np.zeros((1,20)),csr_matrix((1,19)))
        with self.assertRaises(ValueError):
            evaluate(None,{},self.frame,12,20)
        invalid=self.frame.copy()
        invalid.loc[0,'item_idx']=20
        with self.assertRaises(ValueError):
            cohort(invalid,self.frame,20)
        score=ranking_metrics(np.array([[1,2]]),[{1}],k=2)
        self.assertEqual(score,{'NDCG@2':1.,'Recall@2':1.})
