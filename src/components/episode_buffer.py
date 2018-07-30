import torch as th
import numpy as np
from types import SimpleNamespace as SN

#env thread: dict of tensors (keys must match (some of) those in scheme)
#runner: EpBatch(t,bs)
#buffer: Buffer(EpBatch(t,many))


class EpisodeBatch:
    def __init__(self, scheme, groups, max_seq_length, batch_size, data=None):
        self.scheme = scheme.copy()
        self.groups = groups
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        if data is not None:
            self.data = data
        else:
            self._setup_data(self.scheme, self.groups, max_seq_length, batch_size)

    def _setup_data(self, scheme, groups, max_seq_length, batch_size):
        self.data = SN()
        self.data.transition_data = {}
        self.data.episode_data = {}

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,)},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.max_seq_length, self.batch_size)

    def update(self, data, bs=slice(None), ts=slice(None), episode_const=False):
        if not episode_const:
            self.update_transition_data(data, bs, ts)
        else:
            self.update_episode_data(data, bs)

    def update_transition_data(self, data, bs=slice(None), ts=slice(None)):
        slices = self._parse_slices((bs, ts))

        # This only applies when inserting environmental data.
        # Will be overwritten when adding EpisodeBatch data to the replay buffer
        self.data.transition_data["filled"][slices] = 1

        if isinstance(data, dict):
            data_items = data.items()
        elif isinstance(data, EpisodeBatch):
            data_items = data.data.transition_data.items()
        else:
            raise ValueError("Must update transition data with dict or EpisodeBatch, not {}".format(str(type(data))))

        for k, v in data_items:
            if k in self.data.transition_data:
                #TODO: guard to make sure we're only viewing to add singleton b/v dims if needed.
                self.data.transition_data[k][slices] = v.view_as(self.data.transition_data[k][slices])

    def update_episode_data(self, data, bs=slice(None)):
        bs = self._parse_slices((bs, slice(None)))

        if isinstance(data, dict):
            data_items = data.items()
        elif isinstance(data, EpisodeBatch):
            data_items = data.data.episode_data.items()
        else:
            raise ValueError("Must update episode data with dict or EpisodeBatch, not {}".format(str(type(data))))

        for k, v in data_items:
            if k in self.data.episode_data:
                self.data.episode_data[k][bs] = v.view_as(self.data.episode_data[k][bs])

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.max_seq_length, self.batch_size, data=new_data)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_max_t, ret_bs, data=new_data)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            # TODO: Is there a cleaner way to do this?
            if indexing_item.start is None and indexing_item.stop is None:
                ret_bs = max_size
            elif indexing_item.start is None:
                ret_bs = indexing_item.stop
            elif indexing_item.stop is None:
                ret_bs = max_size - indexing_item.start
            else:
                ret_bs = indexing_item.stop - indexing_item.start
            return ret_bs

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, max_seq_length, buffer_size):
        super(ReplayBuffer, self).__init__(scheme, groups, max_seq_length, buffer_size)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update_transition_data(ep_batch.data.transition_data,
                                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                                        slice(0, ep_batch.max_seq_length))
            self.update_episode_data(ep_batch.data.episode_data,
                                     slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        # Uniform sampling only atm
        ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
        return self[ep_ids]

    # def _check_slice(self, slice, max_size):
    #     if slice.step is not None:
    #         return slice.step > 0  # pytorch doesn't support negative steps so neither do we
    #     if slice.start is None and slice.stop is None:
    #         return True
    #     elif slice.start is None:
    #         return 0 < slice.stop <= max_size
    #     elif slice.stop is None:
    #         return 0 <= slice.start < max_size
    #     else:
    #         return (0 < slice.stop <= max_size) and (0 <= slice.start < max_size)

if __name__ == "__main__":
    bs = 4
    groups = {"agents": 2,}
    # "input": {"vshape": (shape), "episode_const": bool, "group": (name), "dtype": dtype}
    scheme = {
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "obs": {"vshape": (3,), "group": "agents"},
        "state": {"vshape": (3,3)},
        "epsilon": {"vshape": (1,), "episode_const": True}
    }

    ep_batch = EpisodeBatch(scheme, groups, 3, bs)

    env_data = {
        "obs": th.ones(2, 3),
        "state": th.eye(3)
    }
    batch_data = {
        "obs": th.ones(2, 3).unsqueeze(0).repeat(bs,1,1),
        "state": th.eye(3).unsqueeze(0).repeat(bs,1,1),
    }
    # bs=4 x t=3 x v=3*3

    ep_batch.update_transition_data(env_data, 0, 0)

    ep_batch.update({"epsilon": th.ones(bs)*.05}, episode_const=True)

    ep_batch[:, 1].update_transition_data(batch_data)
    ep_batch.update_transition_data(batch_data, ts=2)

    print(ep_batch["filled"])

    ep_batch.update_transition_data(env_data, 0, 2)

    env_data = {
        "obs": th.ones(2, 3),
        "state": th.eye(3)*2
    }
    ep_batch.update_transition_data(env_data, 3, 0)

    b2 = ep_batch[0, 1]
    b2.update(env_data, 0, 0)

    replay_buffer = ReplayBuffer(scheme, groups, 3, 5)

    replay_buffer.insert_episode_batch(ep_batch)

    replay_buffer.insert_episode_batch(ep_batch)

    sampled = replay_buffer.sample(3)

    print("sampled bs", sampled.batch_size)