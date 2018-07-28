import torch as th
import numpy as np
from types import SimpleNamespace as SN

#env thread: dict of tensors (keys must match (some of) those in scheme)
#runner: EpBatch(t,bs)
#buffer: Buffer(EpBatch(t,many))

def parse_getitem(items):
    # assert time slice is contiguous!
    if id is None:
        return slice(None, None, None)
    if isinstance(id, slice):
        return id
    elif isinstance(id, (tuple, list)):
        return id
    else:
        return slice(id, id + 1)


class EpisodeBatch:
    def __init__(self, scheme, groups, max_seq_length, batch_size, data=None):
        self.scheme = scheme
        self.groups = groups
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        if data is not None:
            self.data = data
        else:
            self._setup_data(scheme, groups, max_seq_length, batch_size)

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
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype)

    #TODO: expose these with __setitem__, support more indexing options?
    def update_transition_data(self, data, bs=slice(None), ts=slice(None)):
        slices = self._parse_slices([bs, ts])
        self.data.transition_data["filled"][slices] = 1
        for k, v in data.items():
            if k in self.data.transition_data:
                #TODO: guard to make sure we're only viewing to add singleton b/v dims if needed.
                self.data.transition_data[k][slices] = v.view_as(self.data.transition_data[k][slices])

    def update_episode_data(self, data, bs):
        bs = self._parse_slices([bs])
        for k, v in data.items():
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
        elif isinstance(item, list) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))
            ret = EpisodeBatch(self.scheme, self.groups, self.max_seq_length, self.batch_size, data=new_data)
            #TODO: update scheme? do we need scheme after initialisation?
            return ret
        else:
            item = self._parse_slices(item)
            assert 1 <= len(item) <= 2
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret = EpisodeBatch(self.scheme, self.groups, self.max_seq_length, self.batch_size, data=new_data)
            return ret

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        for item in items:
            if isinstance(item, int):
                parsed.append(slice(item, item+1))
            else:
                parsed.append(item)
        return parsed


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, max_seq_length, buffer_size):
        super(ReplayBuffer, self).__init__(scheme, groups, max_seq_length, buffer_size)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_size - self.buffer_index <= ep_batch.batch_size:
            self.update_transition_data(ep_batch.transition_data,
                                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                                        slice(0, ep_batch.max_seq_length))
            self.update_episode_data(ep_batch.episode_data,
                                     slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size) % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left])
            self.insert_episode_batch(ep_batch[buffer_left:])


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

    ep_batch[:, 1].update_transition_data(batch_data)
    ep_batch.update_transition_data(batch_data, slice(None), 2)

    print(ep_batch["filled"])

    ep_batch.update_transition_data(env_data, slice(0,1), slice(2,3))

    env_data = {
        "obs": th.ones(2, 3),
        "state": th.eye(3)*2
    }
    ep_batch.update_transition_data(env_data, slice(3,4), slice(0,1))

    b2 = ep_batch[slice(0,1),slice(1,2)]
    b2.update_transition_data(env_data, slice(0,1), slice(0,1))