# Suggested 5-minute presentation
1. Introduce the problem: correlated process sensors can shift because of faults.
2. Explain the synthetic four-sensor baseline and PCA dimensionality reduction.
3. Explain GMM as a model of multiple normal operating clusters.
4. Explain T² (variation in retained PCA space) and Q (residual variation).
5. Inject each fault and observe detection and diagnosis.
6. Explain posterior probabilities as model estimates, not certainties.
7. State limitations: simulated stream, illustrative thresholds, no plant integration or safety certification.

# Common questions
- Why PCA? It compresses correlated variables and separates dominant variation.
- Why GMM? It can represent multiple clusters / operating regimes.
- Why T² and Q? They monitor retained score-space and residual-space behaviour.
- Is it real-time? It refreshes a simulated stream in the dashboard; it is not connected to live plant sensors.
- What is needed for deployment? Validated plant data, engineering validation, alarm testing, drift monitoring, cybersecurity and safety review.
