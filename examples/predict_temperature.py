"""Small standalone API example; run after installing podguard."""

from podguard import Parameters, ThermalPlate, fit_pod
from podguard.study import training_parameters

plate = ThermalPlate(25)
fit = fit_pod(plate, training_parameters(), rank=12)
parameters = Parameters(kx=0.8, ky=1.7, reaction=4, source_a=0.35, source_b=0.65)
prediction = fit.model.predict(parameters)
error = plate.norm(plate.solve(parameters) - fit.model.reconstruct(prediction))
print(f"Discrete mean temperature: {prediction.mean:.8f}")
print(f"Discrete L2 error: {error:.3e}; bound: {prediction.l2_bound:.3e}")
assert error <= prediction.l2_bound + 1e-11
