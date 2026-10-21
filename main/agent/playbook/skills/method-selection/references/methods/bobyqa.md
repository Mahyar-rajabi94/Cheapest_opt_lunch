# bobyqa — Powell's BOBYQA (quadratic-model trust region)
**Good for:** smooth or mildly rough local refinement without gradients;
nearly as fast as lbfgsb on smooth problems and more robust to small noise.
**Weak at:** strongly multimodal problems; it needs 2·dim+1 evaluations
before its first model step.
**Finished:** `last_event = exhausted`.
**Use as:** `polish`, or the local candidate in a race.
