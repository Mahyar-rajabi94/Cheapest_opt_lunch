"""Sequential finite-pool ExtraTrees lower-confidence acquisition."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

class PoolTrees:
    def __init__(self,features,beta=1.5,seed=0):
        self.features=np.asarray(features,dtype=float)
        self.beta=beta
        self.seed=seed
        self.ids=[]
        self.y=[]
    def tell(self,index,y):
        self.ids.append(int(index));self.y.append(float(y))
    def choose(self,available):
        available=np.asarray(available,dtype=int)
        self.model=ExtraTreesRegressor(n_estimators=512,max_features=.5,min_samples_leaf=2,
                                     random_state=self.seed,n_jobs=1,bootstrap=False,max_depth=None)
        self.model.fit(self.features[self.ids],self.y)
        draws=np.asarray([tree.predict(self.features[available]) for tree in self.model.estimators_])
        mu=draws.mean(0);sd=draws.std(0)
        score=mu-self.beta*sd
        pos=int(np.argmin(score));index=int(available[pos])
        detail={'mean':float(mu[pos]),'sd':float(sd[pos]),'lcb':float(score[pos]),'beta':self.beta,
                'uncertainty':'spread_across_trees_not_calibrated_posterior',
                'nearest_observed_distance':float(np.linalg.norm(self.features[self.ids]-self.features[index],axis=1).min())}
        return index,detail
