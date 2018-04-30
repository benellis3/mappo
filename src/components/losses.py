import torch as th

from components.transforms import _to_batch, _from_batch

class EntropyRegularisationLoss():

    def __init__(self):
        pass

    def forward(self, policies, tformat):

        _policies, policies_params, policies_tformat = _to_batch(policies, tformat)

        # need batch scalar product as in COMA!!!
        entropy = -th.bmm(th.log(policies).unsqueeze(1),
                          policies.unsqueeze(2)).squeeze(2)

        ret = _from_batch(entropy, policies_params, policies_tformat)
        return ret