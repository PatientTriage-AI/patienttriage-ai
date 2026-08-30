# Demo Script (3-5 mins)

1. **Open dashboard & show safety banner:** Emphasize this is decision-support only.
2. **Upload multiple patients:** Use "Batch Intake" tab, upload template CSV to show scaling.
3. **Show Queue:** Display validated patient table, point out urgency sorting.
4. **Select low-risk patient:** Show ESI 4 recommendation + SHAP factors + Fast-Track eligibility.
5. **Show ambiguous/pediatric case:** Intake a pediatric case. Show that the model does not run and escalates to a clinician.
6. **Override:** Accept a recommendation, then override another with a required reason. Show audit trail.
7. **Wait Limit / Deterioration:** Show queue reassessment alerts.
8. **ML Failure:** Toggle "Simulate ML Failure" in the sidebar. Show safe fallback.
9. **Surge:** Toggle "3x Surge". Show increased queue pressure but identical safety logic.
