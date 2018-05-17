from copy import deepcopy
from functools import partial
from itertools import combinations
from models import REGISTRY as mo_REGISTRY
import numpy as np
from numpy.random import randint
import torch as th
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop

from debug.debug import IS_PYCHARM_DEBUG
from components.scheme import Scheme
from components.transforms import _adim, _bsdim, _tdim, _vdim, \
    _generate_input_shapes, _generate_scheme_shapes, _build_model_inputs, \
    _join_dicts, _seq_mean, _copy_remove_keys, _make_logging_str, _underscore_to_cap, _check_nan
from components.losses import EntropyRegularisationLoss
from components.transforms import _to_batch, \
    _from_batch, _naninfmean
from utils.xxx import _n_agent_pairings, _agent_ids_2_pairing_id, _pairing_id_2_agent_ids, _n_agent_pair_samples, _agent_ids_2_pairing_id

from .basic import BasicLearner

class XXXPolicyLoss(nn.Module):

    def __init__(self):
        super(XXXPolicyLoss, self).__init__()

    def forward(self, policies, advantages, actions, tformat):
        assert tformat in ["a*bs*t*v"], "invalid input format!"

        policy_mask = (policies == 0.0)
        log_policies = th.log(policies)
        log_policies = log_policies.masked_fill(policy_mask, 0.0)

        _adv = advantages.clone().detach()

        _act = actions.clone()
        _act[_act!=_act] = 0.0 # mask NaNs in _act

        _active_logits = th.gather(log_policies, _vdim(tformat), _act.long())
        _active_logits[actions != actions] = 0.0 # mask logits for actions that are actually NaNs
        _adv[actions != actions] = 0.0

        loss_mean = -(_active_logits.squeeze(_vdim(tformat)) * _adv.squeeze(_vdim(tformat))).mean(dim=_bsdim(tformat)) #DEBUG: MINUS?
        output_tformat = "a*t"

        return loss_mean, output_tformat

class XXXCriticLoss(nn.Module):

    def __init__(self):
        super(XXXCriticLoss, self).__init__()
    def forward(self, input, target, tformat):
        assert tformat in ["a*bs*t*v"], "invalid input format!"

        # targets may legitimately have NaNs - want to zero them out, and also zero out inputs at those positions
        # target = target.detach()
        nans = (target != target)
        target[nans] = 0.0
        input[nans] = 0.0

        # calculate mean-square loss
        ret = (input - target.detach())**2

        # sum over whole sequences
        ret = ret.mean(dim=_tdim(tformat), keepdim=True)

        #sum over agents
        ret = ret.mean(dim=_adim(tformat), keepdim=True)

        # average over batches
        ret = ret.mean(dim=_bsdim(tformat), keepdim=True)

        output_tformat = "s" # scalar
        return ret, output_tformat

class XXXLearner(BasicLearner):

    def __init__(self, multiagent_controller, logging_struct=None, args=None):
        self.args = args
        self.multiagent_controller = multiagent_controller
        # self.agents = multiagent_controller.agents # for now, do not use any other multiagent controller functionality!!
        self.n_agents = multiagent_controller.n_agents
        self.n_actions = self.multiagent_controller.n_actions
        for _i in range(1, 4):
            setattr(self, "T_policy_level{}".format(_i), 0)
            setattr(self, "T_critic_level{}".format(_i), 0)

        #self.T_target_critic_update_interval=args.target_critic_update_interval
        self.stats = {}
        # self.n_critic_learner_reps = args.n_critic_learner_reps
        self.logging_struct = logging_struct

        self.critic_level1 = mo_REGISTRY[self.args.xxx_critic_level1]
        self.critic_level2 = mo_REGISTRY[self.args.xxx_critic_level2]
        self.critic_level3 = mo_REGISTRY[self.args.xxx_critic_level3]

        self.critic_level1_scheme = Scheme([dict(name="observations",
                                                 select_agent_ids=list(range(self.n_agents))),
                                            dict(name="actions",
                                                 rename="past_actions",
                                                 select_agent_ids=list(range(self.n_agents)),
                                                 transforms=[("shift", dict(steps=1)),
                                                             ("one_hot", dict(range=(0, self.n_actions-1)))],
                                                 switch=self.args.xxx_critic_level1_use_past_actions),
                                            *[dict(name="actions_level1__sample{}".format(_i),
                                                   rename="past_actions_level1__sample{}".format(_i),
                                                   transforms=[("shift", dict(steps=1)),
                                                               # ("one_hot", dict(range=(0, self.n_actions - 1)))
                                                               ],
                                                   switch=self.args.xxx_critic_level2_use_past_actions_level1)
                                                for _i in range(_n_agent_pair_samples(self.n_agents))],
                                            *[dict(name="actions_level1__sample{}".format(_i),
                                                   rename="agent_actions__sample{}".format(_i))
                                              for _i in range(_n_agent_pair_samples(self.n_agents))],
                                            dict(name="policies_level1",
                                                 rename="agent_policy"),
                                            dict(name="state")
                                          ])
        self.target_critic_level1_scheme = self.critic_level1_scheme


        self.critic_scheme_level2_fn = \
            lambda _agent_id1, _agent_id2: Scheme([dict(name="agent_id",
                                                        rename="agent_ids",
                                                        transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                        select_agent_ids=[_agent_id1, _agent_id2],),
                                                   dict(name="observations",
                                                        select_agent_ids=[_agent_id1, _agent_id2]),
                                                   #dict(name="actions_level3",
                                                   #     rename="past_actions",
                                                   #     select_agent_ids=list(range(self.n_agents)),
                                                   #     transforms=[("shift", dict(steps=1)),
                                                   #                #("one_hot", dict(range=(0, self.n_actions - 1)))
                                                   #                ],
                                                   #   switch=self.args.xxx_critic_level2_use_past_actions),
                                                   *[dict(name="actions_level2__sample{}".format(0), # MIGHT BE PROBLEMATIC
                                                        rename="other_agent_actions_level2__sample{}".format(_i),
                                                        transforms=[] if _agent_ids_2_pairing_id((_agent_id1, _agent_id2),
                                                                                                 self.n_agents) != _i else [("mask", dict(fill=0.0))],
                                                        )
                                                     for _i in range(_n_agent_pairings(self.n_agents))],
                                                   dict(name="actions_level2__sample{}".format(0), # NEEDS TO BE SAMPLED IN
                                                          rename="past_action_level2__sample{}".format(0),
                                                          transforms=[("shift", dict(steps=1)),
                                                                     #("one_hot", dict(range=(0, self.n_actions - 1)))
                                                                     ],
                                                          ),
                                                   dict(name="actions_level2__sample{}".format(0),
                                                        # NEEDS TO BE SAMPLED IN
                                                        rename="agent_action".format(0),
                                                        ),
                                                   dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id1, _agent_id2]),
                                                   *[dict(name="policies_level2__sample{}".format(_i)) for _i in range(_n_agent_pair_samples(self.n_agents) if self.args.n_pair_samples is None else self.args.n_pair_samples)], #range(_n_agent_pair_samples(self.n_agents))], #select_agent_ids=[_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)])
                                                   dict(name="state"),
                                                   dict(name="avail_actions",
                                                        select_agent_ids=[_agent_id1]),
                                                   dict(name="avail_actions",
                                                        select_agent_ids=[_agent_id2])
                                                 ])


        self.critic_level3_scheme_fn = lambda _agent_id: Scheme([dict(name="agent_id",
                                                                      select_agent_ids=[_agent_id],
                                                                      # transforms=[("one_hot", dict(range=(0, self.n_agents-1)))],
                                                                     ),
                                                                 dict(name="observations",
                                                                      rename="agent_observation",
                                                                      select_agent_ids=[_agent_id],
                                                                     ),
                                                                 dict(name="actions_level3",
                                                                      rename="past_actions",
                                                                      select_agent_ids=list(range(self.n_agents)),
                                                                      transforms=[("shift", dict(steps=1, fill=0)),
                                                                                  #("one_hot", dict(range=(0, self.n_actions-1)))
                                                                                 ],
                                                                    ),
                                                                 dict(name="actions_level3",
                                                                      rename="other_agents_actions",
                                                                      select_agent_ids=list(range(0, self.n_agents)), #[_aid for _aid in range(0, self.n_agents) if _i != _aid],
                                                                      transforms=[("mask", dict(select_agent_ids=[_agent_id], fill=0.0)),
                                                                                #("one_hot", dict(range=(0, self.n_actions - 1)))
                                                                                 ]),
                                                                 dict(name="actions_level3",
                                                                      rename="agent_action",
                                                                      select_agent_ids=[_agent_id], # do NOT one-hot!
                                                                      ),
                                                                 dict(name="state"),
                                                                 dict(name="policies_level3",
                                                                      rename="agent_policy",
                                                                      select_agent_ids=[_agent_id],),
                                                                 dict(name="avail_actions",
                                                                     select_agent_ids=[_agent_id])
                                                                 ])
        self.target_critic_level3_scheme_fn = self.critic_level3_scheme_fn

        # Set up schemes
        self.scheme_level1 = {}
        # level 1
        self.scheme_level1["critic_level1"] = self.critic_level1_scheme
        self.scheme_level1["target_critic_level1"] = self.critic_level1_scheme

        self.schemes_level2 = {}
        # level 2
        for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
            self.schemes_level2["critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] = self.critic_scheme_level2_fn(_agent_id1,_agent_id2)
            self.schemes_level2["target_critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] = self.critic_scheme_level2_fn(_agent_id1,_agent_id2)
        # level 3
        self.schemes_level3 = {}
        for _agent_id in range(self.n_agents):
            self.schemes_level3["critic_level3__agent{}".format(_agent_id)] = self.critic_level3_scheme_fn(_agent_id)
            self.schemes_level3["target_critic_level3__agent{}".format(_agent_id)] = self.critic_level3_scheme_fn(_agent_id)

        # create joint scheme from the critics schemes
        self.joint_scheme_dict = _join_dicts(self.scheme_level1,
                                             self.schemes_level2,
                                             self.schemes_level3,
                                             self.multiagent_controller.joint_scheme_dict)

        # construct model-specific input regions

        # level 1
        assert self.n_agents <=4, "NOT IMPLEMENTED FOR >= 4 agents!"
        self.input_columns_level1 = {}
        self.input_columns_level1["critic_level1"] = {}
        #self.input_columns_level1["critic_input_level1"]["avail_actions"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id])]).agent_flatten()
        self.input_columns_level1["critic_level1"]["qfunction"] = Scheme([dict(name="state"),
                                                                          # dict(name="observations", select_agent_ids=list(range(self.n_agents))),
                                                                          *[dict(name="past_actions_level1__sample{}".format(_i))
                                                                            for _i in range(_n_agent_pair_samples(self.n_agents))],
                                                                          dict(name="past_actions",
                                                                               select_agent_ids=list(range(self.n_agents)),)])
        self.input_columns_level1["critic_level1"]["observations"] = Scheme([dict(name="observations", select_agent_ids=list(range(self.n_agents)))])
        self.input_columns_level1["critic_level1"]["agent_action"] = Scheme([dict(name="agent_actions__sample{}".format(_i))
                                                                                  for _i in range(1)],) # TODO: Expand for more than 4 agents!!
        self.input_columns_level1["critic_level1"]["agent_policy"] = Scheme([dict(name="agent_policy")])
        self.input_columns_level1["target_critic_level1"] = self.input_columns_level1["critic_level1"]

        # level 2

        self.input_columns_level2 = {}
        for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
            self.input_columns_level2["critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] = {}
            self.input_columns_level2["critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["avail_actions_id1"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id1])])
            self.input_columns_level2["critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["avail_actions_id2"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id2])])
            self.input_columns_level2["critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["qfunction"] = \
                Scheme([*[dict(name="other_agent_actions_level2__sample{}".format(_i))
                          for _i in range(_n_agent_pairings(self.n_agents))],
                        dict(name="state"),
                        dict(name="past_action_level2__sample{}".format(0)),
                        dict(name="agent_ids", select_agent_ids=[_agent_id1, _agent_id2]),
                        #dict(name="past_actions", select_agent_ids=list(range(self.n_agents)))
                        ])
            self.input_columns_level2["critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["observations"] = Scheme([dict(name="observations", select_agent_ids=[_agent_id1, _agent_id2])])
            self.input_columns_level2["critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["agent_action"] = Scheme([dict(name="agent_action")])
            self.input_columns_level2["critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["policies_level2"] = Scheme([dict(name="policies_level2__sample{}".format(_i)) for _i in range(_n_agent_pair_samples(self.n_agents) if self.args.n_pair_samples is None else self.args.n_pair_samples)]) #range(_n_agent_pair_samples(self.n_agents))])
            self.input_columns_level2["target_critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] = self.input_columns_level2["critic_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]


        self.input_columns_level3 = {}
        for _agent_id in range(self.n_agents):
            self.input_columns_level3["critic_level3__agent{}".format(_agent_id)] = {}
            self.input_columns_level3["critic_level3__agent{}".format(_agent_id)]["avail_actions"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id])]).agent_flatten()
            self.input_columns_level3["critic_level3__agent{}".format(_agent_id)]["qfunction"] = Scheme([dict(name="other_agents_actions", select_agent_ids=list(range(self.n_agents))), # select all agent ids here, as have mask=0 transform on current agent action
                                                                                                  dict(name="state"),
                                                                                                  dict(name="agent_observation", select_agent_ids=[_agent_id]),
                                                                                                  dict(name="agent_id", select_agent_ids=[_agent_id]),
                                                                                                  dict(name="past_actions", select_agent_ids=list(range(self.n_agents)))]).agent_flatten()
            self.input_columns_level3["critic_level3__agent{}".format(_agent_id)]["agent_action"] = Scheme([dict(name="agent_action", select_agent_ids=[_agent_id])]).agent_flatten()
            self.input_columns_level3["critic_level3__agent{}".format(_agent_id)]["agent_policy"] = Scheme([dict(name="agent_policy", select_agent_ids=[_agent_id])]).agent_flatten()
            self.input_columns_level3["target_critic_level3__agent{}".format(_agent_id)] = self.input_columns_level3["critic_level3__agent{}".format(_agent_id)]

        self.last_target_update_T_critic_level1 = 0
        self.last_target_update_T_critic_level2 = 0
        self.last_target_update_T_critic_level3 = 0
        pass


    def create_models(self, transition_scheme):

        self.scheme_shapes_level1 = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                            dict_of_schemes=self.scheme_level1)

        self.input_shapes_level1 = _generate_input_shapes(input_columns=self.input_columns_level1,
                                                          scheme_shapes=self.scheme_shapes_level1)

        self.scheme_shapes_level2 = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                            dict_of_schemes=self.schemes_level2)

        self.input_shapes_level2 = _generate_input_shapes(input_columns=self.input_columns_level2,
                                                          scheme_shapes=self.scheme_shapes_level2)

        self.scheme_shapes_level3 = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                            dict_of_schemes=self.schemes_level3)

        self.input_shapes_level3 = _generate_input_shapes(input_columns=self.input_columns_level3,
                                                          scheme_shapes=self.scheme_shapes_level3)

        # Set up critic models
        self.critic_models = {}
        self.target_critic_models = {}

        # set up models level 1
        self.critic_models["level1"] = self.critic_level1(input_shapes=self.input_shapes_level1["critic_level1"],
                                                          n_agents=self.n_agents,
                                                          n_actions=self.n_actions,
                                                          args=self.args)
        if self.args.use_cuda:
            self.critic_models["level1"] = self.critic_models["level1"].cuda()
        self.target_critic_models["level1"] = deepcopy(self.critic_models["level1"])

        # set up models level 2
        if self.args.critic_level2_share_params:
            critic_level2 = self.critic_level2(input_shapes=self.input_shapes_level2["critic_level2__agent0"],
                                               n_actions=self.n_actions,
                                               args=self.args)
            if self.args.use_cuda:
                critic_level2 = critic_level2.cuda()

            for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
                self.critic_models["level2_{}:{}".format(_agent_id1, _agent_id2)] = critic_level2
                self.target_critic_models["level2_{}:{}".format(_agent_id1, _agent_id2)] = deepcopy(critic_level2)
        else:
            assert False, "TODO"

        # set up models level 3
        if self.args.critic_level3_share_params:
            critic_level3 = self.critic_level3(input_shapes=self.input_shapes_level3["critic_level3__agent0"],
                                               n_actions=self.n_actions,
                                               args=self.args)
            if self.args.use_cuda:
                critic_level3 = critic_level3.cuda()

            for _agent_id in range(self.n_agents):
                self.critic_models["level3_{}".format(_agent_id)] = critic_level3
                self.target_critic_models["level3_{}".format(_agent_id)] = deepcopy(critic_level3)
        else:
            assert False, "TODO"

        # set up optimizers
        if self.args.agent_level1_share_params:
            self.agent_level1_parameters = self.multiagent_controller.get_parameters(level=1)
        else:
            assert False, "TODO"
        self.agent_level1_optimiser = RMSprop(self.agent_level1_parameters, lr=self.args.lr_agent_level1)

        if self.args.agent_level2_share_params:
            self.agent_level2_parameters = self.multiagent_controller.get_parameters(level=2)
        else:
            assert False, "TODO"
        self.agent_level2_optimiser = RMSprop(self.agent_level2_parameters, lr=self.args.lr_agent_level2)

        if self.args.agent_level3_share_params:
            self.agent_level3_parameters = self.multiagent_controller.get_parameters(level=3)
        else:
            assert False, "TODO"
        self.agent_level3_optimiser = RMSprop(self.agent_level3_parameters, lr=self.args.lr_agent_level3)

        self.critic_level1_parameters = []
        if self.args.critic_level1_share_params:
            self.critic_level1_parameters.extend(self.critic_models["level1"].parameters())
        else:
            assert False, "TODO"
        self.critic_level1_optimiser = RMSprop(self.critic_level1_parameters, lr=self.args.lr_critic_level1)

        self.critic_level2_parameters = []
        if self.args.critic_level2_share_params:
            self.critic_level2_parameters.extend(self.critic_models["level2_0:1"].parameters())
        else:
            assert False, "TODO"
        self.critic_level2_optimiser = RMSprop(self.critic_level2_parameters, lr=self.args.lr_critic_level2)

        self.critic_level3_parameters = []
        if self.args.critic_level3_share_params:
            self.critic_level3_parameters.extend(self.critic_models["level3_{}".format(0)].parameters())
        else:
            assert False, "TODO"
        self.critic_level3_optimiser = RMSprop(self.critic_level3_parameters, lr=self.args.lr_critic_level3)

        # this is used for joint retrieval of data from all schemes
        self.joint_scheme_dict_level1 = _join_dicts(self.scheme_level1, self.multiagent_controller.joint_scheme_dict_level1)
        self.joint_scheme_dict_level2 = _join_dicts(self.schemes_level2, self.multiagent_controller.joint_scheme_dict_level2)
        self.joint_scheme_dict_level3 = _join_dicts(self.schemes_level3, self.multiagent_controller.joint_scheme_dict_level3)

        self.args_sanity_check() # conduct XXX sanity check on arg parameters
        pass

    def args_sanity_check(self):
        """
        :return:
        """
        pass

    def train(self,
              batch_history,
              T_env=None):

        # -------------------------------------------------------------------------------
        # |  We follow the algorithmic description of XXX as supplied in Algorithm 1   |
        # |  (Counterfactual Multi-Agent Policy Gradients, Foerster et al 2018)         |
        # |  Note: Instead of for-looping backwards through the sample, we just run     |
        # |  repetitions of the optimization procedure sampling from the same batch     |
        # -------------------------------------------------------------------------------

        # Retrieve and view all data that can be retrieved from batch_history in a single step (caching efficient)

        # create one single batch_history view suitable for all
        # data_inputs_level1, data_inputs_tformat_level1 = batch_history.view(dict_of_schemes=self.joint_scheme_dict_level1,
        #                                                                     to_cuda=self.args.use_cuda,
        #                                                                     to_variable=True,
        #                                                                     bs_ids=None,
        #                                                                     fill_zero=True) # DEBUG: Should be True
        #
        # data_inputs_level2, data_inputs_tformat_level2 = batch_history.view(dict_of_schemes=self.joint_scheme_dict_level2,
        #                                                                     to_cuda=self.args.use_cuda,
        #                                                                     to_variable=True,
        #                                                                     bs_ids=None,
        #                                                                     fill_zero=True) # DEBUG: Should be True
        #
        # data_inputs_level3, data_inputs_tformat_level3 = batch_history.view(dict_of_schemes=self.joint_scheme_dict_level3,
        #                                                                     to_cuda=self.args.use_cuda,
        #                                                                     to_variable=True,
        #                                                                     bs_ids=None,
        #                                                                     fill_zero=True) # DEBUG: Should be True

        data_inputs, data_inputs_tformat = batch_history.view(dict_of_schemes=self.joint_scheme_dict,
                                                              to_cuda=self.args.use_cuda,
                                                              to_variable=True,
                                                              bs_ids=None,
                                                              fill_zero=True)
                                                              #fill_zero=True) # DEBUG: Should be True

        #a = {_k:_v.to_pd() for _k, _v in data_inputs.items()}
        #b = batch_history.to_pd()
        self.train_level1(batch_history, data_inputs, data_inputs_tformat, T_env)
        self.train_level2(batch_history, data_inputs, data_inputs_tformat, T_env) # DEBUG
        self.train_level3(batch_history, data_inputs, data_inputs_tformat, T_env)
        pass

    def train_level1(self, batch_history, data_inputs, data_inputs_tformat, T_env):
        # Update target if necessary
        if (self.T_critic_level1 - self.last_target_update_T_critic_level1) / self.args.T_target_critic_level1_update_interval > 1.0:
            self.update_target_nets(level=1)
            self.last_target_update_T_critic_level1 = self.T_critic_level1
            print("updating target net!")

        # actions_level1 = []
        # for _i in range(_n_agent_pair_samples(self.n_agents)):
        #     actions, actions_tformat = batch_history.get_col(bs=None,
        #                                                      col="actions_level1__sample{}".format(_i),
        #                                                      # agent_ids=list(range(0, self.n_agents)),
        #                                                      stack=True)
        #     actions_level1.append(actions)

        assert self.n_agents <= 4, "not implemented for >= 4 agents!"
        actions, actions_tformat = batch_history.get_col(bs=None,
                                                         col="actions_level1__sample{}".format(0),
                                                         # agent_ids=list(range(0, self.n_agents)),
                                                         stack=True)
        actions_level1 = actions.unsqueeze(0)
        #th.stack(actions_level1)


        # do single forward pass in critic
        xxx_model_inputs, xxx_model_inputs_tformat = _build_model_inputs(column_dict=self.input_columns_level1,
                                                                         inputs=data_inputs,
                                                                         inputs_tformat=data_inputs_tformat,
                                                                         to_variable=True,
                                                                         stack=False)
        # pimp up to agent dimension 1
        xxx_model_inputs = {_k1:{_k2:_v2.unsqueeze(0) for _k2, _v2 in _v1.items()} for _k1, _v1 in xxx_model_inputs.items()}
        xxx_model_inputs_tformat = "a*bs*t*v"

        #data_inputs = {_k1:{_k2:_v2.unsqueeze(0) for _k2, _v2 in _v1.items()} for _k1, _v1 in data_inputs.items()}
        #data_inputs_tformat = "a*bs*t*v"

        self._optimize(batch_history=batch_history,
                       xxx_model_inputs=xxx_model_inputs,
                       xxx_model_inputs_tformat=xxx_model_inputs_tformat,
                       data_inputs=data_inputs,
                       data_inputs_tformat=data_inputs_tformat,
                       agent_optimiser=self.agent_level1_optimiser,
                       agent_parameters=self.agent_level1_parameters,
                       critic_optimiser=self.critic_level1_optimiser,
                       critic_parameters=self.critic_level1_parameters,
                       critic=self.critic_models["level1"],
                       target_critic=self.target_critic_models["level1"],
                       xxx_critic_use_sampling=self.args.xxx_critic_level1_use_sampling,
                       xxx_critic_sample_size=self.args.xxx_critic_level1_sample_size,
                       T_critic_str="T_critic_level1",
                       T_policy_str="T_policy_level1",
                       T_env=T_env,
                       level=1,
                       actions=actions_level1)
        pass

    def train_level2(self, batch_history, data_inputs, data_inputs_tformat, T_env):
        # Update target if necessary
        if (self.T_critic_level2 - self.last_target_update_T_critic_level2) / self.args.T_target_critic_level2_update_interval > 1.0:
            self.update_target_nets(level=2)
            self.last_target_update_T_critic_level2 = self.T_critic_level2
            print("updating target net!")

        assert self.n_agents <= 3 or self.args.n_pair_samples == 1 , "only implemented for 3 or fewer agents, or if n_pair_samples == 1"
        actions, actions_tformat = batch_history.get_col(bs=None,
                                                         col="actions_level2__sample{}".format(0))
        actions = actions.unsqueeze(0)

        # do single forward pass in critic
        xxx_model_inputs, xxx_model_inputs_tformat = _build_model_inputs(column_dict=self.input_columns_level2,
                                                                         inputs=data_inputs,
                                                                         inputs_tformat=data_inputs_tformat,
                                                                         to_variable=True)

        # a = data_inputs["critic_level3__agent0"].to_pd()
        self._optimize(batch_history,
                       xxx_model_inputs,
                       xxx_model_inputs_tformat,
                       data_inputs,
                       data_inputs_tformat,
                       agent_optimiser=self.agent_level2_optimiser,
                       agent_parameters=self.agent_level2_parameters,
                       critic_optimiser=self.critic_level2_optimiser,
                       critic_parameters=self.critic_level2_parameters,
                       critic=self.critic_models["level2_0:1"],
                       target_critic=self.target_critic_models["level2_0:1"],
                       xxx_critic_use_sampling=self.args.xxx_critic_level2_use_sampling,
                       xxx_critic_sample_size=self.args.xxx_critic_level2_sample_size,
                       T_critic_str="T_critic_level2",
                       T_policy_str="T_policy_level2",
                       T_env=T_env,
                       level=2,
                       actions=actions)
        pass

    def train_level3(self, batch_history, data_inputs, data_inputs_tformat, T_env):
        # Update target if necessary
        if (self.T_critic_level3 - self.last_target_update_T_critic_level3) / self.args.T_target_critic_level3_update_interval > 1.0:
            self.update_target_nets(level=3)
            self.last_target_update_T_critic_level3 = self.T_critic_level3
            print("updating target net!")

        actions, actions_tformat = batch_history.get_col(bs=None,
                                                         col="actions_level3",
                                                         agent_ids=list(range(0, self.n_agents)),
                                                         stack=True)
        # do single forward pass in critic
        xxx_model_inputs, xxx_model_inputs_tformat = _build_model_inputs(column_dict=self.input_columns_level3,
                                                                         inputs=data_inputs,
                                                                         inputs_tformat=data_inputs_tformat,
                                                                         to_variable=True)

        self._optimize(batch_history,
                       xxx_model_inputs,
                       xxx_model_inputs_tformat,
                       data_inputs,
                       data_inputs_tformat,
                       agent_optimiser=self.agent_level3_optimiser,
                       agent_parameters=self.agent_level3_parameters,
                       critic_optimiser=self.critic_level3_optimiser,
                       critic_parameters=self.critic_level3_parameters,
                       critic=self.critic_models["level3_0"],
                       target_critic=self.target_critic_models["level3_0"],
                       xxx_critic_use_sampling=self.args.xxx_critic_level3_use_sampling,
                       xxx_critic_sample_size=self.args.xxx_critic_level3_sample_size,
                       T_critic_str="T_critic_level3",
                       T_policy_str="T_policy_level3",
                       T_env=T_env,
                       level=3,
                       actions=actions)
        pass

    def _optimize(self,
                  batch_history,
                  xxx_model_inputs,
                  xxx_model_inputs_tformat,
                  data_inputs,
                  data_inputs_tformat,
                  agent_optimiser,
                  agent_parameters,
                  critic_optimiser,
                  critic_parameters,
                  critic,
                  target_critic,
                  xxx_critic_use_sampling,
                  xxx_critic_sample_size,
                  T_critic_str,
                  T_policy_str,
                  T_env,
                  level,
                  actions
                  ):

        critic_loss_arr = []
        critic_mean_arr = []
        target_critic_mean_arr = []
        critic_grad_norm_arr = []

        def _optimize_critic(**kwargs):
            level = kwargs["level"]
            inputs_critic= kwargs["xxx_model_inputs"]["critic_level{}".format(level)]
            inputs_target_critic=kwargs["xxx_model_inputs"]["target_critic_level{}".format(level)]
            inputs_critic_tformat=kwargs["tformat"]
            inputs_target_critic_tformat = kwargs["tformat"]

            # construct target-critic targets and carry out necessary forward passes
            # same input scheme for both target critic and critic!
            output_target_critic, output_target_critic_tformat = target_critic.forward(inputs_target_critic,
                                                                                       tformat=xxx_model_inputs_tformat)



            target_critic_td_targets, \
            target_critic_td_targets_tformat = batch_history.get_stat("td_lambda_targets",
                                                                       bs_ids=None,
                                                                       td_lambda=self.args.td_lambda,
                                                                       gamma=self.args.gamma,
                                                                       value_function_values=output_target_critic["qvalue"].detach(),
                                                                       to_variable=True,
                                                                       n_agents=output_target_critic["qvalue"].shape[_adim(output_target_critic_tformat)],
                                                                       to_cuda=self.args.use_cuda)

            # sample!!
            if xxx_critic_use_sampling:
                critic_shape = inputs_critic[list(inputs_critic.keys())[0]].shape
                sample_ids = randint(critic_shape[_bsdim(inputs_target_critic_tformat)] \
                                        * critic_shape[_tdim(inputs_target_critic_tformat)],
                                     size = xxx_critic_sample_size)
                sampled_ids_tensor = th.from_numpy(sample_ids).long().cuda() if inputs_critic[list(inputs_critic.keys())[0]].is_cuda else th.from_numpy(sample_ids).long()
                _inputs_critic = {}
                for _k, _v in inputs_critic.items():
                    batch_sample = _v.view(
                        _v.shape[_adim(inputs_critic_tformat)],
                        -1,
                        _v.shape[_vdim(inputs_critic_tformat)])[:, sampled_ids_tensor, :]
                    _inputs_critic[_k] = batch_sample.view(_v.shape[_adim(inputs_critic_tformat)],
                                                      -1,
                                                      1,
                                                      _v.shape[_vdim(inputs_critic_tformat)])

                batch_sample_qtargets = target_critic_td_targets.view(target_critic_td_targets.shape[_adim(inputs_critic_tformat)],
                                                                      -1,
                                                                      target_critic_td_targets.shape[_vdim(inputs_critic_tformat)])[:, sampled_ids_tensor, :]
                qtargets = batch_sample_qtargets.view(target_critic_td_targets.shape[_adim(inputs_critic_tformat)],
                                                      -1,
                                                      1,
                                                      target_critic_td_targets.shape[_vdim(inputs_critic_tformat)])
            else:
                _inputs_critic = inputs_critic
                qtargets = target_critic_td_targets

            output_critic, output_critic_tformat = critic.forward(_inputs_critic,
                                                                       tformat=xxx_model_inputs_tformat)


            critic_loss, \
            critic_loss_tformat = XXXCriticLoss()(input=output_critic["qvalue"],
                                                   target=Variable(qtargets, requires_grad=False),
                                                   tformat=target_critic_td_targets_tformat)

            # optimize critic loss
            critic_optimiser.zero_grad()
            critic_loss.backward()

            critic_grad_norm = th.nn.utils.clip_grad_norm(critic_parameters, 50)
            critic_optimiser.step()

            # Calculate critic statistics and update
            target_critic_mean = _naninfmean(output_target_critic["qvalue"])

            critic_mean = _naninfmean(output_critic["qvalue"])

            critic_loss_arr.append(np.asscalar(critic_loss.data.cpu().numpy()))
            critic_mean_arr.append(critic_mean)
            target_critic_mean_arr.append(target_critic_mean)
            critic_grad_norm_arr.append(critic_grad_norm)

            setattr(self, T_critic_str, getattr(self, T_critic_str) + len(batch_history) * batch_history._n_t)

            return output_critic


        output_critic = None

        # optimize the critic as often as necessary to get the critic loss down reliably
        for _i in range(getattr(self.args, "n_critic_level{}_learner_reps".format(level))): #self.n_critic_learner_reps):
            _ = _optimize_critic(xxx_model_inputs=xxx_model_inputs,
                                 tformat=xxx_model_inputs_tformat,
                                 actions=actions,
                                 level=level)

        # get advantages
        output_critic, output_critic_tformat = critic.forward(xxx_model_inputs["critic_level{}".format(level)],
                                                              tformat=xxx_model_inputs_tformat)
        advantages = output_critic["advantage"]

        # only train the policy once in order to stay on-policy!
        policy_loss_function = partial(XXXPolicyLoss(),
                                       actions=Variable(actions),
                                       advantages=advantages)

        hidden_states, hidden_states_tformat = self.multiagent_controller.generate_initial_hidden_states(
            len(batch_history))

        agent_controller_output, \
        agent_controller_output_tformat = self.multiagent_controller.get_outputs(data_inputs,
                                                                                 hidden_states=hidden_states,
                                                                                 loss_fn=policy_loss_function,
                                                                                 loss_level=level,
                                                                                 tformat=data_inputs_tformat,
                                                                                 avail_actions=None,
                                                                                 test_mode=False,
                                                                                 batch_history=batch_history)
        XXX_loss = agent_controller_output["losses"]
        XXX_loss = XXX_loss.mean()

        if self.args.xxx_use_entropy_regularizer:
            XXX_loss += self.args.xxx_entropy_loss_regularization_factor * \
                         EntropyRegularisationLoss()(policies=agent_controller_output["policies"],
                                                     tformat="a*bs*t*v").sum()

        # carry out optimization for agents
        agent_optimiser.zero_grad()
        XXX_loss.backward()

        _check_nan(agent_parameters)
        policy_grad_norm = th.nn.utils.clip_grad_norm(agent_parameters, 50)
        agent_optimiser.step()

        #for p in self.agent_level1_parameters:
        #    print('===========\ngradient:\n----------\n{}'.format(p.grad))

        # increase episode counter (the fastest one is always)
        setattr(self, T_policy_str, getattr(self, T_policy_str) + len(batch_history) * batch_history._n_t)

        # Calculate policy statistics
        advantage_mean = _naninfmean(output_critic["advantage"])
        self._add_stat("advantage_mean_level{}".format(level), advantage_mean, T_env=T_env)
        self._add_stat("policy_grad_norm_level{}".format(level), policy_grad_norm, T_env=T_env)
        self._add_stat("policy_loss_level{}".format(level), XXX_loss.data.cpu().numpy(), T_env=T_env)
        self._add_stat("critic_loss_level{}".format(level), np.mean(critic_loss_arr), T_env=T_env)
        self._add_stat("critic_mean_level{}".format(level), np.mean(critic_mean_arr), T_env=T_env)
        self._add_stat("target_critic_mean_level{}".format(level), np.mean(target_critic_mean_arr), T_env=T_env)
        self._add_stat("critic_grad_norm_level{}".format(level), np.mean(critic_grad_norm_arr), T_env=T_env)
        self._add_stat(T_policy_str, getattr(self, T_policy_str), T_env=T_env)
        self._add_stat(T_critic_str, getattr(self, T_critic_str), T_env=T_env)

        pass

    def update_target_nets(self, level):
        if self.args.critic_level1_share_params and level==1:
            # self.target_critic.load_state_dict(self.critic.state_dict())
            self.critic_models["level1"].load_state_dict(self.critic_models["level1"].state_dict())
        if self.args.critic_level2_share_params and level==2:
            self.critic_models["level2_0:1"].load_state_dict(self.critic_models["level2_0:1"].state_dict())
        if self.args.critic_level3_share_params and level==3:
            self.critic_models["level3_{}".format(0)].load_state_dict(self.critic_models["level3_{}".format(0)].state_dict())
        pass

    def get_stats(self):
        if hasattr(self, "_stats"):
            return self._stats
        else:
            return []

    def log(self, log_directly = True):
        """
        Each learner has it's own logging routine, which logs directly to the python-wide logger if log_directly==True,
        and returns a logging string otherwise

        Logging is triggered in run.py
        """
        stats = self.get_stats()
        logging_dict = {}
        logging_str = ""
        for _i in range(1,4):
            logging_dict.update({"advantage_mean_level{}".format(_i): _seq_mean(stats["advantage_mean_level{}".format(_i)]),
                                 "critic_grad_norm_level{}".format(_i): _seq_mean(stats["critic_grad_norm_level{}".format(_i)]),
                                 "critic_loss_level{}".format(_i):_seq_mean(stats["critic_loss_level{}".format(_i)]),
                                 "policy_grad_norm_level{}".format(_i): _seq_mean(stats["policy_grad_norm_level{}".format(_i)]),
                                 "policy_loss_level{}".format(_i): _seq_mean(stats["policy_loss_level{}".format(_i)]),
                                 "target_critic_mean_level{}".format(_i): _seq_mean(stats["target_critic_mean_level{}".format(_i)]),
                                 "T_critic_level{}".format(_i): getattr(self, "T_critic_level{}".format(_i)),
                                 "T_policy_level{}".format(_i): getattr(self, "T_policy_level{}".format(_i))}
                                )
            logging_str = "T_policy_level{}={:g}, T_critic_level{}={:g}, ".format(_i, logging_dict["T_policy_level{}".format(_i)],
                                                                                  _i, logging_dict["T_critic_level{}".format(_i)])

        logging_str += _make_logging_str(_copy_remove_keys(logging_dict, ["T_policy_level1",
                                                                          "T_critic_level1",
                                                                          "T_policy_level2",
                                                                          "T_critic_level2",
                                                                          "T_policy_level3",
                                                                          "T_critic_level3"]))

        if log_directly:
            self.logging_struct.py_logger.info("{} LEARNER INFO: {}".format(self.args.learner.upper(), logging_str))

        return logging_str, logging_dict

    def save_models(self, path, token, T):

        self.multiagent_controller.save_models(path=path, token=token, T=T)
        th.save(self.critic.state_dict(),"results/models/{}/{}_critic__{}_T.weights".format(self.args.learner,
                                                                                            token,
                                                                                            T))
        th.save(self.target_critic.state_dict(), "results/models/{}/{}_target_critic__{}_T.weights".format(self.args.learner,
                                                                                                           token,
                                                                                                           T))
        pass