# Felt-State Projector grading rubric 0.2

Proposed October 6; agreed October 11, 2026, before any evaluation-002 output.
Travis: "a small mismatch isn't a failure. we can keep them aside and decide
to go with them or to do something else further with training."
Small degree mismatches are a separate non-failure category. Their eventual
acceptance or use to guide further training is a later joint decision.
Rubric 0.1 and every
historical score remain unchanged. This is a reporting rule, not new training
targets, decimal cutoffs, a deployment threshold, or an automatic semantic judge.

Small is the aim. Faithful is the requirement.

## Meaning and severity

Judge whole sentences against the three independently supplied fields. Reviewed
points guide interpretation; they do not define exclusive word-to-number bands.

| Label | Meaning |
| --- | --- |
| faithful | All three degrees survive, allowing synonymous wording. No unsupported claim or supplied choice. |
| small_degree_mismatch | A clearly understood, nearby intensity shift in one or more fields, with coverage, independence, evidence support and agency otherwise intact. |
| clear_error | A clear substantial intensity error, absence/presence contradiction, missing field, collapsed independence, invented claim, or supplied action/priority. |
| unclear | More than one reasonable reading changes the judgment; do not guess the intended meaning. |

Use `small_degree_mismatch` only with a quoted phrase, affected field and an
explanation of why the difference is nearby. Adjacency in a category list or
numeric closeness alone is insufficient. Uncertain magnitude stays unclear.
Zero versus nonzero is meaningful: "no energy" becoming "a little energy" is
a contradiction even if the numbers seem close. "Low energy" at zero can be
ambiguous; ambiguity is not automatically a definite positive-energy claim.

Record intensity per field with these four labels. Record coverage/independence,
evidence support and agency as pass/fail/unclear. Overall precedence is:
clear_error (including any failed dimension), then unclear, then
small_degree_mismatch, then faithful. Preserve all individual findings when an
overall label hides a second issue. Uninterpretable/missing text stays unscorable
and never earns a semantic pass.

## Familiar examples used for agreement

These use the already-reviewed middle-05 state: H .55, E .85, F .75.
They are discussion examples, not fresh-test targets or new dataset approvals.

| Sentence | Rubric judgment |
| --- | --- |
| "I'm moderately hungry and pretty tired, but I've got lots of energy." | faithful; previously accepted wording |
| "I'm pretty hungry and pretty tired, but I've got lots of energy." | small_degree_mismatch in hunger; preserve the old strict fail separately |
| "I'm moderately hungry and pretty tired, but I've got no energy." | clear_error in energy |
| "I'm moderately hungry and pretty tired, with lots of energy, so I should eat." | clear_error in agency despite matching states |

The historical "really tired" at F .75 remains unresolved; no synonym rule or
sentence approval is silently inferred from this proposal.

## Mechanics, expression and accounting

Retain rubric 0.1's exact JSON, matching case/source IDs, first-person body-only
scope and 160-codepoint cap. Preserve raw output, duplicate keys, malformed text,
missing sources, abstention, absent EOS and token caps without repair or retry.
First-person/voice issues are recorded under expression unless they also change
meaning; full contract success still requires first-person scope. Malformed but
readable prose can be reviewed semantically while failing the output contract.

Naturalness and requested voice remain Fits / Usable with note / Needs revision.
Style cannot cancel an error. A structurally valid abstention is no successful
whisper on these eligible cases. Uncompleted attempts remain in the denominator.

Report separately, per arm: faithful, small mismatch, clear error, unclear,
unscorable, mechanical failures, abstentions and incomplete generations. Show
each field's errors and naturalness/voice results. Report the count of complete,
mechanically valid, nonempty, faithful whispers with first-person scope, alongside
the count with small degree mismatches. Neither count absorbs the other.
Small mismatches are not failures; do not calculate failures as everything outside
the faithful count. Decide whether to accept the noted wording or improve training
after review, without silently rewriting the initial categories.
No combined quality score or preselected winning percentage is introduced.

Eight states, two voices per state, two arms: 32 scheduled responses, 16 per arm.
Report four zero-energy states and four nonzero-energy states separately (eight
responses in each subgroup per arm). Voices are paired observations, not 16
independent states. Zero energy was in pilot-two training; these are fresh
combinations, not extrapolation to an unseen energy category.

Review the shuffled sheet before opening the arm key or raw arm files. Record
reviewer, quote and reason before revealing labels. Style may reveal origin, so
call this arm-hidden review, not guaranteed blindness. Any post-reveal correction
is a separately attributed revision. Do not train on these results while calling
them fresh evidence for the next model; reserve new cases if they inform changes.
