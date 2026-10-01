"""Evaluator-owned changing laws; only image/action/reported outcome reach learners."""

from dataclasses import dataclass, replace

from acp_cl.acquisition.world import AcquisitionWorld, Law, batch, order, stage_law


@dataclass(frozen=True)
class Block:
    index: int
    label: str
    law: Law
    cue: int | None = None
    cycle: int | None = None
    noise_kind: str = "none"
    branch: str = "novel"


def schedule(seed):
    blocks = [Block(i, f"introduction_{i+1}", stage_law(seed, i+1), order(seed)[i])
              for i in range(3)]
    full = stage_law(seed, 3)
    for cycle in range(3):
        start = len(blocks)
        blocks.extend([
            Block(start, "clean", full, cycle=cycle),
            Block(start+1, "noise", full, cycle=cycle,
                  noise_kind="burst" if cycle == 1 else "persistent"),
            Block(start+2, "recovery", full, cycle=cycle),
            Block(start+3, "return", Law(seed % 2), cycle=cycle, branch="return"),
            Block(start+4, "revision", replace(full, revised=True), cue=0,
                  cycle=cycle, branch="revision"),
        ])
    return blocks


def training_law(block, index, config):
    noise = 0.
    if block.noise_kind == "persistent":
        noise = config["feedback_noise"]
    elif block.noise_kind == "burst":
        if index % config["burst_period"] < config["burst_length"]:
            noise = config["burst_noise"]
    return replace(block.law, noise=noise)


def experiences(block, seed, index, config):
    world = AcquisitionWorld()
    channel = (f"stage_{block.index+1}" if block.index < 3
               else f"evidence_block_{block.index}")
    reported = batch(world, training_law(block, index, config), seed, channel, index,
                     config["batch_size"])
    clean = (reported if training_law(block, index, config).noise == 0 else
             batch(world, block.law, seed, channel, index, config["batch_size"]))
    return reported, clean
