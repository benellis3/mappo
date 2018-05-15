from torch.autograd import Variable
import torch as th

from components.transforms import _vdim

def test1():
    """
    Does the critic loss go to zero if if I keep the replay buffer sample fixed?
    """
    inputs_level3_tformat = "a*bs*t*v"

    avail_actions1 = Variable(th.FloatTensor(1,5,1,3).uniform_(0, 1).bernoulli())
    avail_actions2 = Variable(th.FloatTensor(1,5,1,3).uniform_(0, 1).bernoulli())
    actions1 = th.LongTensor(1,5,1,1).random_(0, 3).float()
    actions2 = th.LongTensor(1,5,1,1).random_(0, 3).float()
    actions1[0,0,0,0] = float("nan")
    actions2[0,2,0,0] = float("nan")


    print("actions1:", actions1[0,0])
    print("actions2:", actions2[0,0])

    # Now check whether any of the pair_sampled_actions violate individual agent constraints on avail_actions
    actions1_masked = actions1.clone()
    actions1_mask = (actions1 != actions1)
    actions1_masked.masked_fill_(actions1_mask, 0.0)
    actions2_masked = actions2.clone()
    actions2_mask = (actions2 != actions2)
    actions2_masked.masked_fill_(actions2_mask, 0.0)
    actions1[avail_actions1.gather(_vdim(inputs_level3_tformat), Variable(actions1_masked.long())).data == 0.0] = float("nan")
    actions2[avail_actions2.gather(_vdim(inputs_level3_tformat), Variable(actions2_masked.long())).data == 0.0] = float("nan")
    actions1[actions1_mask] = float("nan")
    actions2[actions2_mask] = float("nan")

    print("aa1:", avail_actions1[0,0])
    print("actions1:", actions1[0,0])
    print("aa2:", avail_actions2[0,0])
    print("actions2:", actions2[0,0])
    pass


def main():
    test1()
    #test2()
    #test1b()
    #test2()
    #test3()
    pass

if __name__ == "__main__":
    main()