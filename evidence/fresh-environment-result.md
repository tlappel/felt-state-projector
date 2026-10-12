# Fresh-environment laptop check — passed October 12, 2026

Travis confirmed launching with `.demo-venv/bin/python portable/launch.py live`,
without a training-interpreter override, after following the fresh-install guide
and applying the Python 3.12 audioop-lts correction. This launcher selects the new
UI and inference environments in the extracted bundle and its included model run.
He supplied the candidate and projector receipt below. This completes the fresh
package-environment generation check on the existing laptop with user evidence.

> I'm pretty hungry and not tired, but I've got little energy.

The source is focus-final-07:body, whose packaged state is hunger .75, energy .35,
fatigue 0. The text matches the conversational saved reference graded faithful.
This is a reproduction observation, not a new held-out score or retrospective
regrading. Mechanical checks are valid with no errors or abstention, EOS true and
no token-cap cutoff. Startup: 11.361541549995309 seconds; generation:
7.102157732006162 seconds. These first-call timings are not warmed throughput or
end-to-end receiver latency.

[Receipt and provenance](fresh-environment-receipt-20261012.json). The evidence is
Travis's pasted output and explicit launcher confirmation; full installation logs
and interpreter paths were not independently collected. Existing OS, NVIDIA driver,
system tooling and cached base model were reused. Another machine or cloud GPU
has not been verified. No receiver send was needed for this test; the earlier
live Nebius receipt remains separate evidence.

Both portability checks are now complete on the laptop: relocation with existing
environments, then live generation with the new environments. The Python 3.12 UI
lock issue was fixed; the earlier build/fix documents retain their historical
pending-test labels. Existing downloadable zips are preserved and their embedded
BUILD_STATUS.json reflects assembly time; this dated result supersedes that test
status without silently rewriting the archives.

No further local generation or retraining is needed for this milestone. Next work
is the short submission video and public repository/license/access preparation.
The Discord question on GPU/judging access remains unanswered in this conversation.
Do not call the application publicly deployed or judge-ready until those access
and publication decisions are resolved.
