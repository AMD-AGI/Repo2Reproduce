The deterministic Python executor has finished the selected experiment. This is a fresh,
evidence-only finalization stage; no prior Claude session state is authoritative.

Frozen paper target: {{TARGET_PATH}}
Canonical scientific-configuration audit: {{AUDIT_PATH}}
Environment manifest: {{ENVIRONMENT_PATH}}
Execution manifest: {{MANIFEST_PATH}}
Execution result: {{RESULT_PATH}}
Complete execution log: {{LOG_PATH}}
Artifacts directory: {{ARTIFACTS_PATH}}

Read those files. Inspect any expected result files named by the manifest. Then provide the
final reproduction status:

- selected experiment and paper-grounded configuration;
- exact command represented by the container launch script;
- paper-reported value;
- achieved value;
- whether the values are comparable and whether the result matched;
- every paper/code conflict, paper-missing code fallback, partially implemented or
  unimplemented paper detail, unverified evidence, and protocol/environment deviation;
- any execution failure, missing output, limitation, or assumption.

Do not launch or repeat the experiment. Base the final status only on the persisted manifest,
execution result, log, repository, and output artifacts.

The selected experiment, metric name, paper target, execution status, and achieved value must
exactly match the frozen target and deterministic execution result. Return only the structured
FinalReport requested by the output schema.
