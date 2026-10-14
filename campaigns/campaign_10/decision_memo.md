# Campaign 10 Decision Memo (plain English)

**Did the random twin stay awake?** Yes. We measured how strongly the real
donor sector stirs the signal inside its own model (a factor of about
1.03), then scaled the random twin's blocks until it stirred just as
strongly (from 0.002 up to 0.98). It stayed active after training, too.

**Did the pretrained donor beat that awake twin?** Yes, at the main
training length, twice: by **10.4%** and **4.4%**. So Campaign 9's effect
was not *only* "the twin was asleep."

**But here is the catch, in plain terms:** staying awake made the random
twin *destructive* — random blocks stirring at full donor strength mostly
inject noise, so it became the worst child in every test. Beating it is a
low bar. Against the ordinary (sleeping) twin, the pretrained margin was
much thinner: **5.0%** and **1.0%**. And with triple training time, the
margin vanished and flipped: the ordinary twin actually won by **1.7%**.

**Did the new output pathway improve the real child's quality?** A little
— these are the best pretrained children in this pairing so far (8.31–8.37
at the short budget) — but still far from the usefulness gates (needed:
6.00 or better; language within 1.5× of the host; we're at ~1.85×).

**Was the donor's contribution causal?** Yes — removing or zeroing it
consistently hurt the pretrained children. **Did anything replicate?**
The win over the awake twin did; the win over the ordinary twin is thin
and budget-fragile. **Was held-out testing run?** No —
`NOT_RUN_NO_PROMOTION`.

**What the evidence establishes:** the pretrained slice contributes a
small, real, donor-specific signal when its geometry is respected — but
the effect is strongest exactly when the comparison twin is handicapped,
and training longer erases it. **What remains unproven:** that this signal
can ever grow into useful capability in this pairing.

**Recommendation (not launched):** before any Campaign 11, build the
control this programme still lacks — a random sector that is active *and
benign* (e.g., briefly trained to stability, or weight-shuffled pretrained
blocks that keep donor statistics but destroy learned structure). If the
pretrained advantage survives that control, it deserves real investment;
if not, ten campaigns will have mapped the boundary precisely, and the
honest move is to change the transfer unit or stop.
