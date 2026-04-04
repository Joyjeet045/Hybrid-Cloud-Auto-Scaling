"""Quick smoke test for all NFG-DiagScale components."""
from nfg_diagscale.config import load_config
c = load_config()

from nfg_diagscale.forecasting.kalman_filter import KalmanFilterRPS
from nfg_diagscale.decision.themis_latency import ThemisLatencyModel
from nfg_diagscale.decision.fuzzy_rules import build_rule_base
from nfg_diagscale.decision.anfis import ANFISEngine
from nfg_diagscale.optimizer.scaling_plane import ScalingPlane
from nfg_diagscale.optimizer.rebalance_penalty import RebalancePenalty
from nfg_diagscale.optimizer.nsga2 import NSGA2Optimizer
from nfg_diagscale.monitoring.heat_accumulator import HeatAccumulator
from nfg_diagscale.monitoring.metrics_collector import MetricsCollector
from nfg_diagscale.simulation.cloud_env import CloudEnvironment
from nfg_diagscale.baselines.hpa import HPABaseline
from nfg_diagscale.baselines.vpa import VPABaseline
from nfg_diagscale.baselines.diagonal_scale import DiagonalScaleBaseline
from nfg_diagscale.evaluation.metrics import compute_all_metrics
print("All imports OK")

# Kalman filter
kf = KalmanFilterRPS(c)
assert kf.update(100) == 100
v2 = kf.update(110)
print(f"Kalman: update(100)->100, update(110)->{v2:.2f}")

# Themis latency
tm = ThemisLatencyModel(c)
lat = tm.processing_latency(1, 4)
print(f"Themis l(b=1,c=4) = {lat:.2f} ms")
risk = tm.slo_risk(1, 4, 500, 5)
print(f"Themis SLO risk (rps=500,n=5) = {risk}")

# Scaling Plane
sp = ScalingPlane(c)
nlat = sp.node_latency(4, 8, 1, 1000)
print(f"ScalingPlane node_latency(c=4) = {nlat:.2f}")
coord = sp.coordination_latency(3)
print(f"ScalingPlane coord_latency(H=3) = {coord:.2f}")

# Rebalance Penalty
rp = RebalancePenalty(c)
pen = rp.compute(3, (4, 8, 1, 1000), 5, (6, 8, 1, 1000))
print(f"RebalancePenalty (3,4)->(5,6) = {pen:.2f}")

# Heat Accumulator
ha = HeatAccumulator(c)
ha.update("UP"); ha.update("UP"); ha.update("UP")
print(f"Heat after 3 UPs = {ha.get_heat()}, trigger = {ha.should_trigger()}")

# Fuzzy Rules
rules = build_rule_base()
print(f"Fuzzy rules: {len(rules)}")

# ANFIS
anfis = ANFISEngine(c)
d = anfis.decide(psi=1.5, omega=0.3, phi=0.6, rho=0.0,
                  n_current=2, cores_current=2, predicted_rps=150)
print(f"ANFIS: mode={d['mode']}, dc={d['delta_c']}, dn={d['delta_n']}")

# Cloud Environment
env = CloudEnvironment(c)
state = env.step(100)
print(f"CloudEnv step(100): lat={state['app_latency']:.2f}, cpu={state['cpu_utilization']:.2f}")
env.execute_scaling("vertical", 2, 0)
state2 = env.step(200)
print(f"CloudEnv after +2 cores: lat={state2['app_latency']:.2f}")

# Baselines
hpa = HPABaseline(c)
vpa = VPABaseline(c)
ds = DiagonalScaleBaseline(c)
print(f"Baselines created: {hpa.name}, {vpa.name}, {ds.name}")

# NSGA-II (quick run)
nsga = NSGA2Optimizer(c)
# Override for speed
c_fast = dict(c)
c_fast["nsga2"] = dict(c["nsga2"])
c_fast["nsga2"]["population_size"] = 10
c_fast["nsga2"]["generations"] = 5
nsga_fast = NSGA2Optimizer(c_fast)
front = nsga_fast.optimize(2, 2, 150)
print(f"NSGA-II Pareto front size: {len(front)}")
if front:
    cp = nsga_fast.get_nearest_checkpoint(2, 2)
    print(f"NSGA-II nearest checkpoint: H={cp[0]}, c={cp[1]}")

print("\nAll smoke tests PASSED")
