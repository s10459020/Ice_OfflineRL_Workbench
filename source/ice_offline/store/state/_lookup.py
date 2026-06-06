from ice_offline.store.state.halfcheetah import HalfCheetahConverter, HalfCheetahState, HalfCheetahStateIO
from ice_offline.store.state.hopper import HopperConverter, HopperState, HopperStateIO
from ice_offline.store.state.walker2d import Walker2dConverter, Walker2dState, Walker2dStateIO


STATE_OPS = {
    "HalfCheetah-v5": (HalfCheetahState, HalfCheetahStateIO, HalfCheetahConverter),
    "Hopper-v5": (HopperState, HopperStateIO, HopperConverter),
    "Walker2d-v5": (Walker2dState, Walker2dStateIO, Walker2dConverter),
}
