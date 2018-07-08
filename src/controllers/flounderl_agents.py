from copy import deepcopy
import numpy as np
from torch.autograd import Variable
import torch as th

from components.action_selectors import REGISTRY as as_REGISTRY
from components import REGISTRY as co_REGISTRY
from components.scheme import Scheme
from components.episode_buffer import BatchEpisodeBuffer
from components.transforms import _build_model_inputs, _join_dicts, \
    _generate_scheme_shapes, _generate_input_shapes, _adim, _bsdim, _tdim, _vdim, _agent_flatten, _check_nan, \
    _to_batch, _from_batch, _vdim, _join_dicts, _underscore_to_cap, _copy_remove_keys, _make_logging_str, _seq_mean, \
    _pad_nan

from itertools import combinations
from models import REGISTRY as mo_REGISTRY
from utils.mackrel import _n_agent_pair_samples, _agent_ids_2_pairing_id, _joint_actions_2_action_pair, \
    _pairing_id_2_agent_ids, _pairing_id_2_agent_ids__tensor, _n_agent_pairings, \
    _agent_ids_2_pairing_id, _joint_actions_2_action_pair_aa, _ordered_agent_pairings, _excluded_pair_ids

class FLOUNDERLMultiagentController():
    """
    container object for a set of independent agents
    TODO: may need to propagate test_mode in here as well!
    """

    def __init__(self, runner, n_agents, n_actions, action_selector=None, args=None, logging_struct=None):
        self.args = args
        self.runner = runner
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_output_type = "policies"
        self.logging_struct = logging_struct

        self._stats = {}

        self.model_class = mo_REGISTRY[args.flounderl_agent_model]

        # # Set up action selector
        if action_selector is None:
            self.action_selector = as_REGISTRY[args.action_selector](args=self.args)
        else:
            self.action_selector = action_selector

        # b = _n_agent_pairings(self.n_agents)
        self.agent_scheme_level1 = Scheme([#dict(name="actions",
        #                                         rename="past_actions",
        #                                         select_agent_ids=list(range(self.n_agents)),
        #                                         transforms=[("shift", dict(steps=1)),
        #                                                       ("one_hot", dict(range=(0, self.n_actions)))],
        #                                         switch=self.args.flounderl_agent_use_past_actions),
                                           dict(name="flounderl_epsilons_central_level1",
                                                scope="episode"),
                                           #dict(name="observations",
                                           #     select_agent_ids=list(range(self.n_agents)))
                                           #if not self.args.flounderl_use_obs_intersections else
                                           dict(name="obs_intersection_all")
                                           ])


        self.agent_scheme_level2_fn = lambda _agent_id1, _agent_id2: Scheme([dict(name="agent_id",
                                                                                  rename="agent_ids",
                                                                                  transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                                                  select_agent_ids=[_agent_id1, _agent_id2],),
                                                                             dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id1, _agent_id2]),
                                                                             dict(name="flounderl_epsilons_central",
                                                                                  scope="episode"),
                                                                             dict(name="avail_actions",
                                                                                  select_agent_ids=[_agent_id1, _agent_id2]),
                                                                             # dict(name="observations",
                                                                             #      select_agent_ids=[_agent_id1, _agent_id2],
                                                                             #      switch=not self.args.flounderl_use_obs_intersections),
                                                                             dict(name="obs_intersection__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)),
                                                                                  ),
                                                                             dict(name="avail_actions__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)),
                                                                                  ),
                                                                             dict(name="flounderl_epsilons_central_level2",
                                                                                  scope="episode"),
                                                                             dict(name="actions",
                                                                                  rename="past_actions",
                                                                                  select_agent_ids=list(range(self.n_agents)),
                                                                                  transforms=[("shift", dict(steps=1)),
                                                                                              ("one_hot", dict(range=(0, self.n_actions)))],
                                                                                  switch=self.args.flounderl_agent_use_past_actions),
                                                                             ])

        self.agent_scheme_level3_fn = lambda _agent_id: Scheme([dict(name="agent_id",
                                                                     transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                                     select_agent_ids=[_agent_id],),
                                                                dict(name="observations",
                                                                     select_agent_ids=[_agent_id]),
                                                                dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id]),
                                                                dict(name="avail_actions", select_agent_ids=[_agent_id]),
                                                                dict(name="flounderl_epsilons_central_level3",
                                                                     scope="episode"),
                                                                dict(name="actions",
                                                                     rename="past_actions",
                                                                     transforms=[("shift", dict(steps=1)),
                                                                                 ("one_hot", dict(range=(0, self.n_actions)))],
                                                                     select_agent_ids=list(range(self.n_agents)),
                                                                     switch=self.args.flounderl_agent_use_past_actions),
                                                                ])

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
            Scheme([dict(name="observations", select_agent_ids=list(range(self.n_agents)))
                        if not self.args.flounderl_use_obs_intersections else
                    dict(name="obs_intersection_all"),
                  ])
        self.input_columns_level1["agent_input_level1"]["epsilons_central_level1"] = \
             Scheme([dict(name="flounderl_epsilons_central_level1",
                          scope="episode")])

        # level 2
        self.input_columns_level2 = {}
        for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] = {}
            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["main"] = \
                Scheme([dict(name="observations", select_agent_ids=[_agent_id1, _agent_id2])
                        if not self.args.flounderl_use_obs_intersections else
                        dict(name="obs_intersection__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2),self.n_agents))),
                        dict(name="agent_ids", select_agent_ids=[_agent_id1, _agent_id2]),
                        ])
            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["epsilons_central_level2"] = \
                 Scheme([dict(name="flounderl_epsilons_central_level2",
                              scope="episode")])

            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["avail_actions_id1"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id1])])
            self.input_columns_level2[
                "agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["avail_actions_id2"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id2])])
            if self.args.flounderl_use_obs_intersections:
                self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] \
                                            ["avail_actions_pair"] = Scheme([dict(name="avail_actions__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)),
                                                        switch=self.args.flounderl_use_obs_intersections)])

        # level 3
        self.input_columns_level3 = {}
        for _agent_id in range(self.n_agents):
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)] = {}
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["main"] = \
                Scheme([dict(name="observations", select_agent_ids=[_agent_id]),
                        dict(name="agent_id", select_agent_ids=[_agent_id]),
                        ])
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["epsilons_central_level3"] = \
                Scheme([dict(name="flounderl_epsilons_central_level3",
                             scope="episode")])
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["avail_actions"] = \
                Scheme([dict(name="avail_actions",
                             select_agent_ids=[_agent_id])])

        pass

    def get_parameters(self):
        return list(self.model.parameters())

    def select_actions(self, inputs, avail_actions, tformat, info, hidden_states=None, test_mode=False, **kwargs):
        """
        sample from the FLOUNDERL tree
        """
        # cd = inputs["agent_input_level2__agent0"].to_pd() # DEBUG

        T_env = info["T_env"]
        test_suffix = "" if not test_mode else "_test"

        if self.args.agent_level1_share_params:

            # --------------------- LEVEL 1

            inputs_level1, inputs_level1_tformat = _build_model_inputs(self.input_columns_level1,
                                                                       inputs,
                                                                       to_variable=True,
                                                                       inputs_tformat=tformat)
            if self.args.debug_mode:
                _check_nan(inputs_level1)

            out_level1, hidden_states_level1, losses_level1, tformat_level1 = self.model.model_level1(inputs_level1["agent_input_level1"],
                                                                                                      hidden_states=hidden_states["level1"],
                                                                                                      loss_fn=None,
                                                                                                      tformat=inputs_level1_tformat,
                                                                                                      n_agents=self.n_agents,
                                                                                                      test_mode=test_mode,
                                                                                                      **kwargs)


            if self.args.debug_mode:
                _check_nan(inputs_level1)


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


            inputs_level2, inputs_level2_tformat = _build_model_inputs(self.input_columns_level2,
                                                                       inputs,
                                                                       to_variable=True,
                                                                       inputs_tformat=tformat,
                                                                       )

            assert self.args.agent_level2_share_params, "not implemented!"


            if "avail_actions_pair" in inputs_level2["agent_input_level2"]:
                pairwise_avail_actions = inputs_level2["agent_input_level2"]["avail_actions_pair"]
            else:
                avail_actions1, params_aa1, tformat_aa1 = _to_batch(inputs_level2["agent_input_level2"]["avail_actions_id1"], inputs_level2_tformat)
                avail_actions2, params_aa2, _ = _to_batch(inputs_level2["agent_input_level2"]["avail_actions_id2"], inputs_level2_tformat)
                pairwise_avail_actions = th.bmm(avail_actions1.unsqueeze(2), avail_actions2.unsqueeze(1))
                pairwise_avail_actions = _from_batch(pairwise_avail_actions, params_aa2, tformat_aa1)

            ttype = th.cuda.FloatTensor if pairwise_avail_actions.is_cuda else th.FloatTensor
            delegation_avails = Variable(ttype(pairwise_avail_actions.shape[0],
                                               pairwise_avail_actions.shape[1],
                                               pairwise_avail_actions.shape[2], 1).fill_(1.0), requires_grad=False)
            pairwise_avail_actions = th.cat([delegation_avails, pairwise_avail_actions], dim=_vdim(tformat))

            # mask = pairwise_avail_actions.data.clone().fill_(0.0)
            # mask[:,:,:,-1] = 1.0
            # mask.scatter_(_adim(inputs_level2_tformat), sampled_pair_ids.repeat(1,1,1,pairwise_avail_actions.shape[-1]).long(), pairwise_avail_actions.data)
            # pairwise_avail_actions = Variable(mask, requires_grad=False)

            out_level2, hidden_states_level2, losses_level2, tformat_level2 = self.model.models["level2_{}".format(0)](inputs_level2["agent_input_level2"],
                                                                                                                        hidden_states=hidden_states["level2"],
                                                                                                                        loss_fn=None,
                                                                                                                        tformat=inputs_level2_tformat,
                                                                                                                        sampled_pair_ids=sampled_pair_ids,
                                                                                                                        pairwise_avail_actions=pairwise_avail_actions,
                                                                                                                        test_mode=test_mode,
                                                                                                                        **kwargs)


            pair_sampled_actions, \
            modified_inputs_level2, \
            selected_actions_format_level2 = self.action_selector.select_action({"policies":out_level2},
                                                                                avail_actions=pairwise_avail_actions.data,
                                                                                tformat=tformat_level2,
                                                                                test_mode=test_mode)
            # if th.sum(pair_sampled_actions == 26.0) > 0.0:
            #     a = 5

            self.actions_level2 = pair_sampled_actions.clone()
            self.actions_level2_sampled = pair_sampled_actions.gather(0, sampled_pair_ids.long())
            self.selected_actions_format_level2 = selected_actions_format_level2
            self.policies_level2 = modified_inputs_level2.clone()


            inputs_level3, inputs_level3_tformat = _build_model_inputs(self.input_columns_level3,
                                                                       inputs,
                                                                       to_variable=True,
                                                                       inputs_tformat=tformat,
                                                                       )

            pair_id1, pair_id2 = _pairing_id_2_agent_ids__tensor(sampled_pair_ids, self.n_agents,
                                                                 "a*bs*t*v")  # sampled_pair_ids.squeeze(0).squeeze(2).view(-1), self.n_agents)

            avail_actions1 = inputs_level3["agent_input_level3"]["avail_actions"].gather(
                _adim(inputs_level3_tformat), Variable(pair_id1.repeat(1, 1, 1, inputs_level3["agent_input_level3"][
                    "avail_actions"].shape[_vdim(inputs_level3_tformat)])))
            avail_actions2 = inputs_level3["agent_input_level3"]["avail_actions"].gather(
                _adim(inputs_level3_tformat), Variable(pair_id2.repeat(1, 1, 1, inputs_level3["agent_input_level3"][
                    "avail_actions"].shape[_vdim(inputs_level3_tformat)])))

            # selected_level_2_actions = pair_sampled_actions.gather(0, sampled_pair_ids.long())
            pair_sampled_actions = pair_sampled_actions.gather(0, sampled_pair_ids.long())

            actions1, actions2 = _joint_actions_2_action_pair_aa(pair_sampled_actions,
                                                                 self.n_actions,
                                                                 avail_actions1,
                                                                 avail_actions2)
            # count how often level2 actions are un-available at level 3
            pair_action_unavail_rate = (th.mean((actions1 != actions1).float()).item() + th.mean((actions2 != actions2).float()).item()) / 2.0
            self._add_stat("pair_action_unavail_rate__runner",
                           pair_action_unavail_rate,
                           T_env=T_env,
                           suffix=test_suffix,
                           to_sacred=False)

            # Now check whether any of the pair_sampled_actions violate individual agent constraints on avail_actions
            ttype = th.cuda.FloatTensor if self.args.use_cuda else th.FloatTensor
            #action_matrix = ttype(self.n_agents,
            #                      pair_sampled_actions.shape[_bsdim(tformat)] *
            #                      pair_sampled_actions.shape[_tdim(tformat)]).fill_(float("nan"))

            #action_matrix.scatter_(0, pair_id1.squeeze(-1).view(pair_id1.shape[0], -1),
            #                       actions1.squeeze(-1).view(actions1.shape[0], -1))
            #action_matrix.scatter_(0, pair_id2.squeeze(-1).view(pair_id2.shape[0], -1),
            #                       actions2.squeeze(-1).view(actions2.shape[0], -1))

            action_tensor = ttype(self.n_agents,
                                  pair_sampled_actions.shape[_bsdim(tformat)],
                                  pair_sampled_actions.shape[_tdim(tformat)],
                                  1).fill_(float("nan"))
            action_tensor.scatter_(0, pair_id1, actions1)
            action_tensor.scatter_(0, pair_id2, actions2)

            dbg = action_tensor.clone() # DEBUG
            dbg[dbg!=dbg] = -float("inf") # DEBUG
            mmm = dbg.max().cpu().numpy() # DEBUG
            assert mmm < self.n_actions, "no-op in env action at mmm!" # DEBUG

            avail_actions_level3 = inputs_level3["agent_input_level3"]["avail_actions"].clone().data
            self.avail_actions = avail_actions_level3.clone()

            # active = action_tensor.clone() # action_matrix.view(self.n_agents, pair_sampled_actions.shape[_bsdim(tformat)], pair_sampled_actions.shape[_tdim(tformat)], -1).clone() #.unsqueeze(2).clone()
            # active[active == active] = 0.0
            # active[active != active] = 1.0
            # avail_actions_level3[active.repeat(1, 1, 1, avail_actions_level3.shape[_vdim(tformat)]) == 0.0] = \
            # avail_actions_level3[active.repeat(1, 1, 1, avail_actions_level3.shape[_vdim(tformat)]) == 0.0].fill_(0.0)
            # avail_actions_level3[:, :, :, -1:] = 1.0 - active
            #
            #
            # av = avail_actions_level3.clone().view(-1, avail_actions_level3.shape[3]).cpu().numpy() # DEBUG
            # at_old = action_tensor.clone().cpu().view(-1, action_tensor.shape[3]).numpy() # DEBUG
            # at_old_shp = action_tensor.clone().cpu().view(action_tensor.shape[0], -1, action_tensor.shape[3])
            # act = active.clone().cpu().view(-1, action_tensor.shape[3]).numpy() # DEBUG

            inputs_level3["agent_input_level3"]["avail_actions"] = Variable(avail_actions_level3,
                                                                            requires_grad=False)

            out_level3, hidden_states_level3, losses_level3, tformat_level3 = self.model.models["level3_{}".format(0)](inputs_level3["agent_input_level3"],
                                                                                                                       hidden_states=hidden_states["level3"],
                                                                                                                       loss_fn=None,
                                                                                                                       tformat=inputs_level3_tformat,
                                                                                                                       test_mode=test_mode,
                                                                                                                       **kwargs)
            # extract available actions
            avail_actions_level3 = inputs_level3["agent_input_level3"]["avail_actions"]

            individual_actions, \
            modified_inputs_level3, \
            selected_actions_format_level3 = self.action_selector.select_action({"policies":out_level3},
                                                                                avail_actions=avail_actions_level3.data,
                                                                                tformat=tformat_level3,
                                                                                test_mode=test_mode)

            # fill into action matrix all the actions that are not NaN
            #individual_actions_sq = individual_actions.squeeze(_vdim(tformat_level3)).view(individual_actions.shape[_adim(tformat_level3)], -1)
            #self.actions_level3 = individual_actions_sq.view(individual_actions.shape[_adim(tformat_level3)],
            #                                                 individual_actions.shape[_bsdim(tformat_level3)],
            #                                                 individual_actions.shape[_tdim(tformat_level3)],
            #                                                 1)
            self.actions_level3 = individual_actions
            action_tensor[action_tensor != action_tensor] = individual_actions[action_tensor != action_tensor]

            # set states beyond episode termination to NaN
            action_tensor = _pad_nan(action_tensor, tformat=tformat_level3, seq_lens=inputs["agent_input_level1"].seq_lens) # DEBUG

            dbg = action_tensor.clone() # DEBUG
            dbg[dbg!=dbg] = -float("inf") # DEBUG
            mmm2 = dbg.max().cpu().numpy() # DEBUG
            try:
                assert mmm2 < self.n_actions, "no-op in env action at mmm2!" # DEBUG
            except:
                at_new = action_tensor.clone().cpu().view(-1, action_tensor.shape[3]).numpy()
                indv = individual_actions.clone().cpu().view(-1, individual_actions.shape[3]).numpy()
                av_env = self.avail_actions.clone().view(-1, self.avail_actions.shape[3]).cpu().numpy()
                pass

            # l2 = action_tensor.squeeze()  # DEBUG
            if self.args.debug_mode in ["level3_actions_only"]:
                """
                DEBUG MODE: LEVEL3 ACTIONS ONLY
                Here we just pick actions from level3 - should therefore just correspond to vanilla COMA!
                """
                action_tensor  = individual_actions

            #self.final_actions = action_matrix.view(individual_actions.shape[_adim(tformat_level3)], # self.n_agents, #
            #                                        individual_actions.shape[_bsdim(tformat_level3)],
            #                                        individual_actions.shape[_tdim(tformat_level3)],
            #                                        1)
            self.final_actions = action_tensor.clone()

            #self.actions_level3 = individual_actions.clone()
            self.selected_actions_format_level3 = selected_actions_format_level3
            self.policies_level3 = modified_inputs_level3.clone()
            self.avail_actions_active = avail_actions_level3.data

            selected_actions_list = []
            for _i in range(_n_agent_pair_samples(self.n_agents) if self.args.n_pair_samples is None else self.args.n_pair_samples): #_n_agent_pair_samples(self.n_agents)):
                selected_actions_list += [dict(name="actions_level1__sample{}".format(_i),
                                               data=self.actions_level1[_i])]
            for _i in range(_n_agent_pair_samples(self.n_agents) if self.args.n_pair_samples is None else self.args.n_pair_samples):
                selected_actions_list += [dict(name="actions_level2__sample{}".format(_i),
                                               data=self.actions_level2_sampled[_i])] # TODO: BUG!?
            selected_actions_list += [dict(name="actions_level2",
                                           select_agent_ids=list(range(_n_agent_pairings(self.n_agents))),
                                           data=self.actions_level2)]
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
            modified_inputs_list += [dict(name="avail_actions_active",
                                          select_agent_ids=list(range(self.n_agents)),
                                          data=self.avail_actions_active)]
            modified_inputs_list += [dict(name="avail_actions",
                                          select_agent_ids=list(range(self.n_agents)),
                                          data=self.avail_actions)]

            #modified_inputs_list += [dict(name="avail_actions",
            #                              select_agent_ids=list(range(self.n_agents)),
            #                              data=self.avail_actions)]

            hidden_states = dict(level1=hidden_states_level1,
                                 level2=hidden_states_level2,
                                 level3=hidden_states_level3)

            return hidden_states, selected_actions_list, modified_inputs_list, self.selected_actions_format

            pass

        else:
            assert False, "Not implemented"

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

        self.model = self.model_class(input_shapes=dict(level1=self.input_shapes_level1["agent_input_level1"],
                                                        level2=self.input_shapes_level2["agent_input_level2__agent0"],
                                                        level3=self.input_shapes_level3["agent_input_level3__agent0"]),
                                      n_agents=self.n_agents,
                                      n_actions=self.n_actions,
                                      model_classes=dict(level1=mo_REGISTRY[self.args.flounderl_agent_model_level1],
                                                         level2=mo_REGISTRY[self.args.flounderl_agent_model_level2],
                                                         level3=mo_REGISTRY[self.args.flounderl_agent_model_level3]),
                                      args=self.args)

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

    def get_outputs(self, inputs, hidden_states, tformat, loss_fn=None, actions=None, **kwargs):

        if loss_fn is None or actions is None:
            assert False, "not implemented - always need loss function and selected actions!"

        avail_actions = kwargs["avail_actions"]
        test_mode = kwargs["test_mode"]
        batch_history = kwargs.get("batch_history", None)

        if self.args.share_agent_params:
            inputs_level1, inputs_level1_tformat = _build_model_inputs(self.input_columns_level1,
                                                                       inputs,
                                                                       inputs_tformat=tformat,
                                                                       to_variable=True)

            inputs_level2, inputs_level2_tformat = _build_model_inputs(self.input_columns_level2,
                                                                       inputs,
                                                                       to_variable=True,
                                                                       inputs_tformat=tformat,
                                                                       )

            inputs_level3, inputs_level3_tformat = _build_model_inputs(self.input_columns_level3,
                                                                       inputs,
                                                                       to_variable=True,
                                                                       inputs_tformat=tformat,
                                                                       )

            out, hidden_states, losses, tformat = self.model(inputs=dict(level1=inputs_level1,
                                                                         level2=inputs_level2,
                                                                         level3=inputs_level3),
                                                             hidden_states=hidden_states,
                                                             loss_fn=loss_fn,
                                                             tformat=dict(level1=inputs_level1_tformat,
                                                                          level2=inputs_level2_tformat,
                                                                          level3=inputs_level3_tformat),
                                                             n_agents=self.n_agents,
                                                             actions=actions,
                                                             **kwargs)

            ret_dict = dict(hidden_states = hidden_states,
                            losses = losses)
            ret_dict[self.agent_output_type] = out
            return ret_dict, tformat #losses[loss_level] if loss_level is not None else None), tformat_level3

        else:
            assert False, "Not yet implemented."

        pass

    def save_models(self, path, token, T):
        if self.args.share_agent_params:
            th.save(self.agent_model.state_dict(),
                    "results/models/{}/{}_agentsp__{}_T.weights".format(token, self.args.learner, T))
        else:
            for _agent_id in range(self.args.n_agents):
                th.save(self.agent_models["agent__agent{}".format(_agent_id)].state_dict(),
                        "results/models/{}/{}_agent{}__{}_T.weights".format(token, self.args.learner, _agent_id, T))
        pass

    def _add_stat(self, name, value, T_env, suffix="", to_sacred=True, to_tb=True):
        name += suffix

        if isinstance(value, np.ndarray) and value.size == 1:
            value = float(value)

        if not hasattr(self, "_stats"):
            self._stats = {}

        if name not in self._stats:
            self._stats[name] = []
            self._stats[name+"_T_env"] = []
        self._stats[name].append(value)
        self._stats[name+"_T_env"].append(T_env)

        if hasattr(self, "max_stats_len") and len(self._stats) > self.max_stats_len:
            self._stats[name].pop(0)
            self._stats[name+"_T_env"].pop(0)

        # log to sacred if enabled
        if hasattr(self.logging_struct, "sacred_log_scalar_fn") and to_sacred:
            self.logging_struct.sacred_log_scalar_fn(key=_underscore_to_cap(name), val=value)

        # log to tensorboard if enabled
        if hasattr(self.logging_struct, "tensorboard_log_scalar_fn") and to_tb:
            self.logging_struct.tensorboard_log_scalar_fn(_underscore_to_cap(name), value, T_env)

        return

    def log(self, test_mode=None, T_env=None, log_directly = True):
        """
        Each learner has it's own logging routine, which logs directly to the python-wide logger if log_directly==True,
        and returns a logging string otherwise

        Logging is triggered in run.py
        """
        test_suffix = "" if not test_mode else "_test"

        stats = self.get_stats()
        stats["pair_action_unavail_rate"+test_suffix] = _seq_mean(stats["pair_action_unavail_rate__runner"+test_suffix])

        self._add_stat("pair_action_unavail_rate",
                       stats["pair_action_unavail_rate"+test_suffix],
                       T_env=T_env,
                       suffix=test_suffix,
                       to_sacred=True)

        if stats == {}:
            self.logging_struct.py_logger.warning("Stats is empty... are you logging too frequently?")
            return "", {}

        logging_dict =  dict(T_env=T_env)

        logging_dict["pair_action_unavail_rate"+test_suffix] =stats["pair_action_unavail_rate"+test_suffix]

        logging_str = ""
        logging_str += _make_logging_str(_copy_remove_keys(logging_dict, ["T_env"+test_suffix]))


        if log_directly:
            self.logging_struct.py_logger.info("{} MC INFO: {}".format("TEST" if self.test_mode else "TRAIN",
                                                                           logging_str))
        return logging_str, logging_dict


    def get_stats(self):
        if hasattr(self, "_stats"):
            tmp = deepcopy(self._stats)
            self._stats={}
            return tmp
        else:
            return []


    pass




