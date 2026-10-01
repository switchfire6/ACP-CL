"""Separate reservoir age from lifetime IDs; retain disjoint recent/old stores."""

import copy

import numpy as np

from acp_cl.conditional.learner import PacketMemory


RESET_ARMS = ("keep", "clear", "rebase", "clear_rebase", "rng_reset", "full_reset")
POLICIES = ("uniform", "recent", "split")


class ReservoirMemory(PacketMemory):
    def __init__(self, capacity, seed):
        super().__init__(capacity, seed)
        self.age = 0

    def add(self, packet):
        index = self.seen
        self.seen += 1
        self.age += 1
        if not self.capacity:
            return
        if len(self.packets) < self.capacity:
            self.packets.append(packet)
            self.ids.append(index)
        else:
            slot = int(self.membership_rng.integers(0, self.age))
            if slot < self.capacity:
                self.packets[slot], self.ids[slot] = packet, index


class RecentHistoricalMemory:
    def __init__(self, capacity, recent, seed):
        if not 0 <= recent <= capacity:
            raise ValueError("invalid recent capacity")
        self.capacity, self.recent_capacity = capacity, recent
        self.seen = self.historical_seen = 0
        self.recent_packets, self.recent_ids = [], []
        self.historical_packets, self.historical_ids = [], []
        self.membership_rng = np.random.default_rng(seed+47201)
        self.sampling_rng = np.random.default_rng(seed+62071)

    @property
    def packets(self):
        return self.recent_packets+self.historical_packets

    @property
    def ids(self):
        return self.recent_ids+self.historical_ids

    def add(self, packet):
        index = self.seen
        self.seen += 1
        self.recent_packets.append(packet)
        self.recent_ids.append(index)
        if len(self.recent_packets) <= self.recent_capacity:
            return
        old, old_id = self.recent_packets.pop(0), self.recent_ids.pop(0)
        self.historical_seen += 1
        capacity = self.capacity-self.recent_capacity
        if len(self.historical_packets) < capacity:
            self.historical_packets.append(old)
            self.historical_ids.append(old_id)
        elif capacity:
            slot = int(self.membership_rng.integers(0, self.historical_seen))
            if slot < capacity:
                self.historical_packets[slot], self.historical_ids[slot] = old, old_id

    def sample(self):
        packets = self.packets
        return packets[int(self.sampling_rng.integers(0, len(packets)))] if packets else None

    def nbytes(self):
        return sum(p.nbytes() for p in self.packets)+8*len(self.ids)


def make_memory(policy, config, seed):
    if policy == "uniform":
        return ReservoirMemory(config["memory_packets"], seed)
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    recent = config["memory_packets"] if policy == "recent" else config["recent_packets"]
    return RecentHistoricalMemory(config["memory_packets"], recent, seed)


def reset_memory(memory, arm, seed):
    if arm not in RESET_ARMS or not isinstance(memory, ReservoirMemory):
        raise ValueError("invalid reset")
    result = copy.deepcopy(memory)
    if arm in ("clear", "clear_rebase", "full_reset"):
        result.packets, result.ids = [], []
    if arm in ("rebase", "clear_rebase", "full_reset"):
        result.age = len(result.packets)
    if arm in ("rng_reset", "full_reset"):
        fresh = ReservoirMemory(memory.capacity, seed)
        result.membership_rng, result.sampling_rng = fresh.membership_rng, fresh.sampling_rng
    return result


def memory_state(memory, include_packets=True):
    state = dict(type=type(memory).__name__, capacity=memory.capacity, seen=memory.seen,
        ids=list(memory.ids), membership=memory.membership_rng.bit_generator.state,
        sampling=memory.sampling_rng.bit_generator.state)
    for name in ("age", "recent_capacity", "historical_seen", "recent_ids", "historical_ids"):
        if hasattr(memory, name):
            state[name] = copy.deepcopy(getattr(memory, name))
    if include_packets:
        state["packets"] = [dict(support=None if p.support is None else p.support.fingerprint(),
            query=p.query.fingerprint(), hidden=p.oracle_modes is not None) for p in memory.packets]
    return state
