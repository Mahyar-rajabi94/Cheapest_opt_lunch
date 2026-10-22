
## Evaluation 32
**Change:** Suspected heavy-tail outlier reading y=1.782e+05 at x=[0.6729564371701394, 4.6852498529648345, -1.8101924285105389, 5.0, -5.0] (median=4294, MAD=2174); re-evaluated same x, got y2=1.782e+05; feeding GP-BO the mean (1.782e+05) instead of the raw single reading.
**Reason:** |y - median| = 1.739e+05 > 6.0 * MAD = 1.304e+04, matching the pre-registered outlier-guard criterion from roadmap_initial.md.

## Evaluation 34
**Change:** Suspected heavy-tail outlier reading y=2.754e+04 at x=[-4.781749378808622, -0.4907243843208119, -2.8222934103222888, 0.8866827594026143, 5.0] (median=5600, MAD=3583); re-evaluated same x, got y2=2.754e+04; feeding GP-BO the mean (2.754e+04) instead of the raw single reading.
**Reason:** |y - median| = 2.194e+04 > 6.0 * MAD = 2.15e+04, matching the pre-registered outlier-guard criterion from roadmap_initial.md.

## Evaluation 47
**Change:** Suspected heavy-tail outlier reading y=1.449e+05 at x=[-4.71229595270857, -2.300753915917047, 1.4492866391556312, -4.792668428539127, -5.0] (median=6701, MAD=4639); re-evaluated same x, got y2=1.449e+05; feeding GP-BO the mean (1.449e+05) instead of the raw single reading.
**Reason:** |y - median| = 1.382e+05 > 6.0 * MAD = 2.784e+04, matching the pre-registered outlier-guard criterion from roadmap_initial.md.

## Evaluation 47
**Change:** Ending GP-BO phase early and moving to confirm_best phase.
**Reason:** Running best has not improved over the last 30 evaluations (best_so_far history window shows no change beyond 1e-9). Reserving remaining budget (53) for replication/confirmation of top candidates instead of further unconfirmed GP proposals.
