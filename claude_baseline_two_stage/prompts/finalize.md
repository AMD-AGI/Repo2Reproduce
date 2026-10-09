The deterministic Python executor has finished the selected experiment. This is a fresh,
evidence-only finalization stage; no prior Claude session state is authoritative.

Frozen paper target: {{TARGET_PATH}}
Bidirectional configuration audit: {{AUDIT_PATH}}
Execution specification: {{SPEC_PATH}}
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
- every code-only assumption, paper detail not implemented, or protocol deviation;
- any execution failure, missing output, limitation, or assumption.

Do not launch or repeat the experiment. Base the final status only on the persisted manifest,
execution result, log, repository, and output artifacts.
