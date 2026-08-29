# 3-5 minute demonstration script

1. Open **New arrival** and state the prominent boundary: clinical decision support only, nurse confirmation required, not for production use.
2. Load a held-out low-risk fixture. Explain that the expected ESI is hidden from the nurse-facing workflow. With no trained local artifact, the demo safely requests clinician review instead of fabricating a score; after governed local training, the same screen shows calibrated score, urgent risk, and three SHAP factors.
3. Load **SIMULATED: pediatric rules-only escalation**. Show that it is immediately blocked from adult ML and Fast-Track.
4. Load a case with the chest-pain red-flag cue. Show that the cue is selected by the nurse, rather than inferred from untrusted complaint text, and it triggers clinician review.
5. Use **SIMULATED: zero-history intake** with complete current vitals. Point out that zero history is visible but is not a block by itself.
6. Go to **Nurse decision**, choose override, and show that a structured reason is mandatory. Record the disposition and describe the local append-only audit event and mock EHR write.
7. In **Waiting queue**, turn on 3x surge. Explain that the display expands capacity pressure only: urgency ordering and safety thresholds remain unchanged. Apply simulated deterioration to produce a reassessment alert.
8. Turn on **Demonstrate model failure** and reevaluate. The app uses rules-only fallback, displays no score or confidence, and never permits Fast-Track.
9. Finish in **Audit & model card** by search-filtering the timeline and verifying the hash chain. Highlight the adult-only and label-leakage limitations.
