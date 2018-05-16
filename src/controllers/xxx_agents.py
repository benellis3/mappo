import numpy as np
from torch.autograd import Variable
import torch as th

from components.action_selectors import REGISTRY as as_REGISTRY
from components import REGISTRY as co_REGISTRY
from components.scheme import Scheme
from components.episode_buffer import BatchEpisodeBuffer
from components.transforms import _build_model_inputs, _join_dicts, \
    _generate_scheme_shapes, _generate_input_shapes, _adim, _bsdim, _tdim, _vdim, _agent_flatten, _check_nan, \
    _to_batch, _from_batch

from itertools import combinations
from models import REGISTRY as mo_REGISTRY
from utils.xxx import _n_agent_pair_samples, _agent_ids_2_pairing_id, _joint_actions_2_action_pair, \
    _pairing_id_2_agent_ids, _pairing_id_2_agent_ids__tensor, _n_agent_pairings, _agent_ids_2_pairing_id, _joint_actions_2_action_pair_aa
class XXXMultiagentController():
    """
    container object for a set of independent agents
    TODO: may need to propagate test_mode in here as well!
    """

    def __init__(self, runner, n_agents, n_actions, action_selector=None, args=None):
        self.args = args
        self.runner = runner
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_str = args.agent
        self.agent_output_type = "policies"


        self.model_level1 = mo_REGISTRY[args.xxx_agent_model_level1]
        self.model_level2 = mo_REGISTRY[args.xxx_agent_model_level2]
        self.model_level3 = mo_REGISTRY[args.xxx_agent_model_level3]

        # # Set up action selector
        if action_selector is None:
            self.action_selector = as_REGISTRY[args.action_selector](args=self.args)
        else:
            self.action_selector = action_selector

        self.agent_scheme_level1 = Scheme([*[dict(name="actions_level1__sample{}".format(_i),
                                                rename="past_actions_level1__sample{}".format(_i),
                                                transforms=[("shift", dict(steps=1)),
                                                            ("one_hot", dict(range=(0, self.n_actions-1)))],
                                                switch=self.args.xxx_agent_level1_use_past_actions)
                                             for _i in range(_n_agent_pair_samples(self.n_agents))],
                                           dict(name="xxx_epsilons_central_level1",
                                                scope="episode"),
                                           dict(name="observations",
                                                select_agent_ids=list(range(self.n_agents)))
                                           if not self.args.xxx_use_obs_intersections else
                                           dict(name="obs_intersection_all")
                                           ])


        self.agent_scheme_level2_fn = lambda _agent_id1, _agent_id2: Scheme([dict(name="agent_id",
                                                                                  rename="agent_ids",
                                                                                  transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                                                  select_agent_ids=[_agent_id1, _agent_id2],),
                                                                             dict(name="observations",
                                                                                  select_agent_ids=[_agent_id1, _agent_id2]),
                                                                             dict(name="actions_level2__sample{}".format(0),
                                                                                  rename="past_actions_level2",
                                                                                  transforms=[("shift", dict(steps=1)),
                                                                                              ("one_hot_pairwise", dict(range=(0, self.n_actions-1)))],
                                                                                  switch=self.args.xxx_agent_level2_use_past_actions),
                                                                             # TODO: transform to split into two actions (with maybe one-hot encoding each)!
                                                                             dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id1, _agent_id2]),
                                                                             dict(name="xxx_epsilons_central_level2",
                                                                                  scope="episode"),
                                                                             dict(name="avail_actions",
                                                                                  select_agent_ids=[_agent_id1, _agent_id2]),
                                                                             dict(name="observations",
                                                                                   select_agent_ids=[_agent_id1, _agent_id2])
                                                                                if not self.args.xxx_use_obs_intersections else
                                                                             dict(name="obs_intersection_pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))),
                                                                             dict(name="avail_actions_pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)))])

        self.agent_scheme_level3_fn = lambda _agent_id: Scheme([dict(name="agent_id",
                                                                     transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                                     select_agent_ids=[_agent_id],),
                                                                dict(name="observations",
                                                                     select_agent_ids=[_agent_id]),
                                                                dict(name="actions_level3",
                                                                     rename="past_actions_level3",
                                                                     select_agent_ids=[_agent_id],
                                                                     transforms=[("shift", dict(steps=1)),
                                                                                 ("one_hot", dict(range=(0, self.n_actions-1)))], # DEBUG!
                                                                     switch=self.args.xxx_agent_level3_use_past_actions),
                                                                dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id]),
                                                                dict(name="xxx_epsilons_central_level3", scope="episode"),
                                                                * [dict(name="actions_level1__sample{}".format(_i),
                                                                        rename="past_actions_level1__sample{}".format(_i),
                                                                        transforms=[("one_hot", dict(range=(0, self.n_actions - 1)))],
                                                                        switch=self.args.xxx_agent_level3_use_past_actions_level1)
                                                                   for _i in range(_n_agent_pair_samples(self.n_agents))],
                                                                dict(name="avail_actions", select_agent_ids=[_agent_id]) ])

        # Set up schemes
        self.schemes = {}
        # level 1
        self.schemes_level1 = {}
        self.schemes_level1["agent_input_level1"] = self.agent_scheme_level1

        # level 2
        self.schemes_level2 = {}
        for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
            self.schemes_level2["agent_input_level2__agent{}"
                .format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] \
                = self.agent_scheme_level2_fn(_agent_id1, _agent_id2)
        # level 3
        self.schemes_level3 = {}
        for _agent_id in range(self.n_agents):
            self.schemes_level3["agent_input_level3__agent{}".format(_agent_id)] = self.agent_scheme_level3_fn(_agent_id).agent_flatten()

        # create joint scheme from the agents schemes
        self.joint_scheme_dict_level1 = _join_dicts(self.schemes_level1)
        self.joint_scheme_dict_level2 = _join_dicts(self.schemes_level2)
        self.joint_scheme_dict_level3 = _join_dicts(self.schemes_level3)

        self.joint_scheme_dict = _join_dicts(self.schemes_level1, self.schemes_level2, self.schemes_level3)
        # construct model-specific input regions

        # level 1
        self.input_columns_level1 = {}
        self.input_columns_level1["agent_input_level1"] = {}
        self.input_columns_level1["agent_input_level1"]["main"] = \
            Scheme([#dict(name="observations", select_agent_ids=list(range(self.n_agents))),
                    dict(name="observations", select_agent_ids=list(range(self.n_agents)))
                        if not self.args.xxx_use_obs_intersections else
                    dict(name="obs_intersection_all"),
                    *[dict(name="past_actions_level1__sample{}".format(_i),
                          switch=self.args.xxx_agent_level1_use_past_actions)
                      for _i in range(_n_agent_pair_samples(self.n_agents))]
                  ])
        self.input_columns_level1["agent_input_level1"]["epsilons_central_level1"] = \
            Scheme([dict(name="xxx_epsilons_central_level1",
                         scope="episode")])
        #self.input_columns_level1["agent_input_level1"]["epsilons_level1"] = \
        #    Scheme([dict(name="xxx_epsilons_level1")])

        # level 2
        self.input_columns_level2 = {}
        for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] = {}
            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["main"] = \
                Scheme([dict(name="observations", select_agent_ids=[_agent_id1, _agent_id2])
                        if not self.args.xxx_use_obs_intersections else
                        dict(name="obs_intersection_pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2),self.n_agents))),
                        dict(name="past_actions_level2",
                             switch=self.args.xxx_agent_level2_use_past_actions),
                        dict(name="agent_ids", select_agent_ids=[_agent_id1, _agent_id2])])
            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["epsilons_central_level2"] = \
                Scheme([dict(name="xxx_epsilons_central_level2",
                             scope="episode")])
            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["avail_actions_id1"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id1])])
            self.input_columns_level2[
                "agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["avail_actions_id2"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id2])])
            if self.args.xxx_use_obs_intersections:
                self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["obs_intersection_pair"] = Scheme([dict(name="obs_intersection_pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)),
                                                                                                                                                                                 switch=self.args.xxx_use_obs_intersections)])
                self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] \
                                            ["avail_actions_pair"] = Scheme([dict(name="avail_actions_pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)),
                                                        switch=self.args.xxx_use_obs_intersections)])
            #self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["epsilons_level2"] = \
            #    Scheme([dict(name="xxx_epsilons_level2")])

        # level 3
        self.input_columns_level3 = {}
        for _agent_id in range(self.n_agents):
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)] = {}
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["main"] = \
                Scheme([dict(name="observations", select_agent_ids=[_agent_id]),
                        dict(name="past_actions_level3",
                             select_agent_ids=[_agent_id],
                             switch=self.args.xxx_agent_level3_use_past_actions),
                        dict(name="agent_id", select_agent_ids=[_agent_id])],
                        dict(name="actions_level1"))
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["epsilons_central_level3"] = \
                Scheme([dict(name="xxx_epsilons_central_level3",
                             scope="episode")])
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["avail_actions"] = \
                Scheme([dict(name="avail_actions",
                             select_agent_ids=[_agent_id])])
           # self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["epsilons_level3"] = \
           #     Scheme([dict(name="xxx_epsilons_level3")])

        pass

    def get_parameters(self, level):
        if level == 1:
            return list(self.models["level1"].parameters())
        elif level == 2:
            param_list = []
            for _i in range(_n_agent_pairings(self.n_agents)):
                param_list.extend(self.models["level2_{}".format(_i)].parameters())
            return param_list
        elif level == 3:
            param_list = []
            for _agent_id in range(self.n_agents):
                param_list.extend(self.models["level3_{}".format(_agent_id)].parameters())
            return param_list

    def select_actions(self, inputs, avail_actions, tformat, info, test_mode=False):

        selected_actions_list = []
        for _i in range(_n_agent_pair_samples(self.n_agents)):
            selected_actions_list += [dict(name="actions_level1__sample{}".format(_i),
                                           data=self.actions_level1[_i])]
        for _i in range(_n_agent_pair_samples(self.n_agents)):
            selected_actions_list += [dict(name="actions_level2__sample{}".format(_i),
                                           data=self.actions_level2[_i])]
        selected_actions_list += [dict(name="actions_level3",
                                       select_agent_ids=list(range(self.n_agents)),
                                       data=self.actions_level3)]
        selected_actions_list += [dict(name="actions",
                                       select_agent_ids=list(range(self.n_agents)),
                                       data=self.final_actions)]

        modified_inputs_list = []
        modified_inputs_list += [dict(name="policies_level1",
                                      data=self.policies_level1)]
        for _i in range(_n_agent_pair_samples(self.n_agents)):
            modified_inputs_list += [dict(name="policies_level2__sample{}".format(_i),
                                          data=self.policies_level2[_i])]
        modified_inputs_list += [dict(name="policies_level3",
                                      select_agent_ids=list(range(self.n_agents)),
                                      data=self.policies_level3)]
        return selected_actions_list, modified_inputs_list, self.selected_actions_format

    def create_model(self, transition_scheme):

        self.scheme_shapes_level1 = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                        dict_of_schemes=self.schemes_level1)

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


        # TODO: Set up agent models
        self.models = {}

        # set up models level 1
        self.models["level1"] = self.model_level1(input_shapes=self.input_shapes_level1["agent_input_level1"],
                                                  n_agents=self.n_agents,
                                                  output_type=self.agent_output_type,
                                                  args=self.args)
        if self.args.use_cuda:
            self.models["level1"] = self.models["level1"].cuda()

        # set up models level 2
        if self.args.agent_level2_share_params:
            model_level2 = self.model_level2(input_shapes=self.input_shapes_level2["agent_input_level2__agent0"],
                                             n_actions=self.n_actions,
                                             output_type=self.agent_output_type,
                                             args=self.args)
            if self.args.use_cuda:
                model_level2 = model_level2.cuda()

            for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
                self.models["level2_{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] = model_level2
        else:
            assert False, "TODO"

        # set up models level 3
        self.models_level3 = {}
        if self.args.agent_level3_share_params:
            model_level3 = self.model_level3(input_shapes=self.input_shapes_level3["agent_input_level3__agent0"],
                                             n_actions=self.n_actions,
                                             output_type=self.agent_output_type,
                                             args=self.args)
            if self.args.use_cuda:
                model_level3 = model_level3.cuda()

            for _agent_id in range(self.n_agents):
                self.models["level3_{}".format(_agent_id)] = model_level3
        else:
            assert False, "TODO"

        return

    def generate_initial_hidden_states(self, batch_size):
        """
        generates initial hidden states for each agent
        """

        # Set up hidden states for all levels - and propagate througn the runner!
        hidden_dict = {}
        hidden_dict["level1"] = th.stack([Variable(th.zeros(batch_size, 1, self.args.agents_hidden_state_size)) for _
                                        in range(1)])
        hidden_dict["level2"] = th.stack([Variable(th.zeros(batch_size, 1, self.args.agents_hidden_state_size)) for _
                                        in range(len(sorted(combinations(list(range(self.n_agents)), 2))))])
        hidden_dict["level3"] = th.stack([Variable(th.zeros(batch_size, 1, self.args.agents_hidden_state_size)) for _
                                        in range(self.n_agents)])
        if self.args.use_cuda:
            hidden_dict = {_k:_v.cuda() for _k, _v in hidden_dict.items()}

        return hidden_dict, "?*bs*v*t"

    def share_memory(self):
        assert False, "TODO"
        pass

    def get_outputs(self, inputs, hidden_states, tformat, loss_fn=None, loss_level=None, **kwargs):

        avail_actions = kwargs["avail_actions"]
        test_mode = kwargs["test_mode"]
        batch_history = kwargs.get("batch_history", None)

        if self.args.share_agent_params:

            # top level: aa' ~ Pi_c sample which pair to coordinate
            # second level: pick up correct pair (given aa'), sample u^a, u^a' from the pair coordinator
            # either decode u^a, u^a' from the sampled action, or refer to level

            # --------------------- LEVEL 1

            if loss_level == 1 or loss_level is None:
                inputs_level1, inputs_level1_tformat = _build_model_inputs(self.input_columns_level1,
                                                                           inputs,
                                                                           to_variable=True,
                                                                           inputs_tformat=tformat)
                _check_nan(inputs_level1)
                out_level1, hidden_states_level1, losses_level1, tformat_level1 = self.models["level1"](inputs_level1["agent_input_level1"],
                                                                                                        hidden_states=hidden_states["level1"],
                                                                                                        loss_fn=loss_fn if loss_level == 1 else None,
                                                                                                        tformat=inputs_level1_tformat,
                                                                                                        n_agents=self.n_agents,
                                                                                                        **kwargs)
                _check_nan(out_level1)
                if loss_level == 1:
                    return dict(losses=losses_level1), tformat_level1

                sampled_pair_ids, modified_inputs_level1, selected_actions_format_level1 = self.action_selector.select_action({"policies":out_level1},
                                                                                                                              avail_actions=None,
                                                                                                                              tformat=tformat_level1,
                                                                                                                              test_mode=test_mode)
                _check_nan(sampled_pair_ids)
                if self.args.debug_mode in ["level2_actions_fixed_pair"]:
                    """
                    DEBUG MODE: LEVEL2 ACTIONS FIXED PAIR
                    Here we pick level2 actions from a fixed agent pair (0,1) and the third action from IQL
                    """
                    assert self.n_agents == 3, "only makes sense in n_agents=3 scenario"
                    sampled_pair_ids.fill_(0.0)

                # sample which pairs should be selected
                self.actions_level1 = sampled_pair_ids.clone()
                self.selected_actions_format = selected_actions_format_level1
                self.policies_level1 = modified_inputs_level1.squeeze(0).clone()

            elif loss_level == 2 or loss_level == 3:
                assert self.n_agents <= 3, "only implemented for 3 or fewer agents!!"
                for _i in  range(_n_agent_pair_samples(self.n_agents)):
                    sampled_pair_ids, _ = batch_history.get_col(col="actions_level1__sample{}".format(_i),)
                sampled_pair_ids[sampled_pair_ids != sampled_pair_ids] = 0.0
                sampled_pair_ids = sampled_pair_ids.unsqueeze(0).contiguous()

            if loss_level is None or loss_level == 2:
                # --------------------- LEVEL 2
                assert self.n_agents < 5, "pair selection only implemented for up to 4 agents yet!!"

                inputs_level2, inputs_level2_tformat = _build_model_inputs(self.input_columns_level2,
                                                                           inputs,
                                                                           to_variable=True,
                                                                           inputs_tformat=tformat,
                                                                           )

                assert self.args.agent_level2_share_params, "not implemented!"

                # # create pairwise avail actions
                # avail_actions1, params_aa1, tformat_aa1 = _to_batch(inputs_level2["agent_input_level2"]["avail_actions_id1"], inputs_level2_tformat)
                # avail_actions2, params_aa2, _ = _to_batch(inputs_level2["agent_input_level2"]["avail_actions_id2"], inputs_level2_tformat)

                ########## HOW TO DETERMINE AVAILABLE ACTIONS ##########################################################
                # One cannot a priori say anything about the other agent's available actions based
                # on the common knowledge. This would need to be learnt. We are therefore hard-coding it for
                #
                # In the last step, we then check whether the actions sent to the individual agents are legal.
                # If not so, then those actions have to be chosen from the independent sampler.
                ########################################################################################################

                # if self.args.env in ["pred_prey", "matrix_game"]:
                #     pairwise_avail_actions = th.bmm(avail_actions1.unsqueeze(2), avail_actions2.unsqueeze(1))
                # elif self.args.env in ["sc1", "sc2"]:
                #     tmp = (avail_actions1  * avail_actions2)
                #     pairwise_avail_actions = th.bmm(tmp.unsqueeze(2), tmp.unsqueeze(1))
                # else:
                #     assert False, "pairwise_avail_actions not determined for this environment!"
                #
                #
                # ttype = th.cuda.FloatTensor if pairwise_avail_actions.is_cuda else th.FloatTensor
                # delegation_avails = Variable(ttype(pairwise_avail_actions.shape[0], 1).fill_(1.0), requires_grad=False)
                # pairwise_avail_actions = pairwise_avail_actions.view(pairwise_avail_actions.shape[0], -1)
                #
                #
                # # if self.args.xxx_use_obs_intersections:
                # #     # if there is no observation intersection, pair-wise can choose any conceivable action
                # #     # (if this results in invalid actions, then those will be picked in a decentralized fashion)
                # #     obs_intersections, i_params, i_format = _to_batch(inputs_level2["agent_input_level2"]["obs_intersection_pair"], tformat=inputs_level2_tformat)
                # #     obs_intersections_mask = (obs_intersections.sum(dim=1, keepdim=True) == 0)
                # #     tmp = obs_intersections_mask.repeat(1, pairwise_avail_actions.shape[1])
                # #     pairwise_avail_actions[tmp.data] = 1.0
                #
                # pairwise_avail_actions = th.cat([delegation_avails, pairwise_avail_actions], dim=1)
                # pairwise_avail_actions = _from_batch(pairwise_avail_actions, params_aa2, tformat_aa1)

                pairwise_avail_actions = inputs_level2["agent_input_level2"]["avail_actions_pair"]
                ttype = th.cuda.FloatTensor if pairwise_avail_actions.is_cuda else th.FloatTensor
                delegation_avails = Variable(ttype(pairwise_avail_actions.shape[0],
                                                   pairwise_avail_actions.shape[1],
                                                   pairwise_avail_actions.shape[2], 1).fill_(1.0), requires_grad=False)
                pairwise_avail_actions = th.cat([delegation_avails, pairwise_avail_actions], dim=_vdim(tformat))

                out_level2, hidden_states_level2, losses_level2, tformat_level2 = self.models["level2_0"](inputs_level2["agent_input_level2"],
                                                                                                          hidden_states=hidden_states["level2"],
                                                                                                          loss_fn=loss_fn if loss_level == 2 else None,
                                                                                                          tformat=inputs_level2_tformat,
                                                                                                          sampled_pair_ids=sampled_pair_ids,
                                                                                                          pairwise_avail_actions=pairwise_avail_actions,
                                                                                                          **kwargs)

                if loss_level == 2:
                    return dict(losses=losses_level2), tformat_level2

                pair_sampled_actions, \
                modified_inputs_level2, \
                selected_actions_format_level2 = self.action_selector.select_action({"policies":out_level2},
                                                                                    avail_actions=pairwise_avail_actions.data,
                                                                                    tformat=tformat_level2,
                                                                                    test_mode=test_mode)

                self.actions_level2 = pair_sampled_actions.clone()
                self.selected_actions_format_level2 = selected_actions_format_level2
                self.policies_level2 = modified_inputs_level2.clone()

            elif loss_level == 3:
                assert self.n_agents <= 3, "only implemented for 3 or fewer agents!!"
                for _i in  range(_n_agent_pair_samples(self.n_agents)):
                    pair_sampled_actions, _ = batch_history.get_col(col="actions_level2__sample{}".format(_i))
                    pair_sampled_actions[pair_sampled_actions != pair_sampled_actions] = 0.0
                pair_sampled_actions = pair_sampled_actions.unsqueeze(0).contiguous()

            if loss_level == 3 or loss_level is None:
                # --------------------- LEVEL 3
                # TODO: np.nan in actions1 !!

                inputs_level3, inputs_level3_tformat = _build_model_inputs(self.input_columns_level3,
                                                                           inputs,
                                                                           to_variable=True,
                                                                           inputs_tformat=tformat,
                                                                           )

                pair_id1, pair_id2 = _pairing_id_2_agent_ids__tensor(sampled_pair_ids, self.n_agents, "a*bs*t*v") # sampled_pair_ids.squeeze(0).squeeze(2).view(-1), self.n_agents)

                avail_actions1 = inputs_level3["agent_input_level3"]["avail_actions"].gather(
                    _adim(inputs_level3_tformat), Variable(pair_id1.repeat(1, 1, 1, inputs_level3["agent_input_level3"][
                        "avail_actions"].shape[_vdim(inputs_level3_tformat)])))
                avail_actions2 = inputs_level3["agent_input_level3"]["avail_actions"].gather(
                    _adim(inputs_level3_tformat), Variable(pair_id2.repeat(1, 1, 1, inputs_level3["agent_input_level3"][
                        "avail_actions"].shape[_vdim(inputs_level3_tformat)])))

                # selected_level_2_actions = pair_sampled_actions.gather(0, sampled_pair_ids.long())
                if loss_level is None:
                    pair_sampled_actions = pair_sampled_actions.gather(0, sampled_pair_ids.long())

                actions1, actions2 = _joint_actions_2_action_pair_aa(pair_sampled_actions, self.n_actions, avail_actions1, avail_actions2)



                # #tmp = Variable(pair_id1).unsqueeze(2).unsqueeze(3).repeat(1,1,1,inputs_level3["agent_input_level3"]["avail_actions"].shape[_vdim(inputs_level3_tformat)])
                # avail_actions1 = inputs_level3["agent_input_level3"]["avail_actions"].gather(_adim(inputs_level3_tformat), Variable(pair_id1.repeat(1,1,1,inputs_level3["agent_input_level3"]["avail_actions"].shape[_vdim(inputs_level3_tformat)])))
                # avail_actions2 = inputs_level3["agent_input_level3"]["avail_actions"].gather(_adim(inputs_level3_tformat), Variable(pair_id2.repeat(1,1,1,inputs_level3["agent_input_level3"]["avail_actions"].shape[_vdim(inputs_level3_tformat)])))

                # Now check whether any of the pair_sampled_actions violate individual agent constraints on avail_actions

                ttype = th.cuda.FloatTensor if self.args.use_cuda else th.FloatTensor
                action_matrix = ttype(self.n_agents,
                                      pair_sampled_actions.shape[_bsdim(tformat)]*
                                      pair_sampled_actions.shape[_tdim(tformat)]).fill_(float("nan"))

                action_matrix.scatter_(0, pair_id1.squeeze(-1).view(pair_id1.shape[0],-1), actions1.squeeze(-1).view(actions1.shape[0],-1))
                action_matrix.scatter_(0, pair_id2.squeeze(-1).view(pair_id2.shape[0],-1), actions2.squeeze(-1).view(actions2.shape[0],-1))

                out_level3, hidden_states_level3, losses_level3, tformat_level3 = self.models["level3_0"](inputs_level3["agent_input_level3"],
                                                                                                          hidden_states=hidden_states["level3"],
                                                                                                          loss_fn=loss_fn if loss_level==3 else None,
                                                                                                          tformat=inputs_level3_tformat,
                                                                                                          **kwargs)
                if loss_level == 3:
                    return dict(losses=losses_level3), tformat_level3

                # extract available actions
                avail_actions_level3 = inputs_level3["agent_input_level3"]["avail_actions"]

                individual_actions, \
                modified_inputs_level3, \
                selected_actions_format_level3 = self.action_selector.select_action({"policies":out_level3},
                                                                                    avail_actions=avail_actions_level3.data,
                                                                                    tformat=tformat_level3,
                                                                                    test_mode=test_mode)

                # fill into action matrix all the actions that are not NaN
                individual_actions_sq = individual_actions.squeeze(_vdim(tformat_level3)).view(individual_actions.shape[_adim(tformat_level3)], -1)

                tmp = individual_actions_sq.clone()
                tmp[action_matrix == action_matrix] = float("nan")
                self.actions_level3 = tmp.view(individual_actions.shape[_adim(tformat_level3)],
                                               individual_actions.shape[_bsdim(tformat_level3)],
                                               individual_actions.shape[_tdim(tformat_level3)],
                                               1)


                action_matrix[action_matrix != action_matrix] = individual_actions_sq

                if self.args.debug_mode in ["level3_actions_only"]:
                    """
                    DEBUG MODE: LEVEL3 ACTIONS ONLY
                    Here we just pick actions from level3 - should therefore just correspond to vanilla COMA!
                    """
                    action_matrix  = individual_actions_sq

                self.final_actions = action_matrix.view(individual_actions.shape[_adim(tformat_level3)],
                                                        individual_actions.shape[_bsdim(tformat_level3)],
                                                        individual_actions.shape[_tdim(tformat_level3)],
                                                        1)

                #self.actions_level3 = individual_actions.clone()
                self.selected_actions_format_level3 = selected_actions_format_level3
                self.policies_level3 = modified_inputs_level3.clone()

            losses = {1:losses_level1,
                      2:losses_level2,
                      3:losses_level3}
            return dict(hidden_states = dict(level1=hidden_states_level1,
                                             level2=hidden_states_level2,
                                             level3=hidden_states_level3),
                        losses = losses[loss_level] if loss_level is not None else None), tformat_level3
        else:
            assert False, "Not yet implemented."

    def save_models(self, path, token, T):
        if self.args.share_agent_params:
            th.save(self.agent_model.state_dict(),
                    "results/models/{}/{}_agentsp__{}_T.weights".format(token, self.args.learner, T))
        else:
            for _agent_id in range(self.args.n_agents):
                th.save(self.agent_models["agent__agent{}".format(_agent_id)].state_dict(),
                        "results/models/{}/{}_agent{}__{}_T.weights".format(token, self.args.learner, _agent_id, T))
        pass

    pass




