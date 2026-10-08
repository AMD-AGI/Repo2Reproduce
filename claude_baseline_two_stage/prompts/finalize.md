The deterministic Python executor has finished the selected experiment.

Execution manifest: {{MANIFEST_PATH}}
Execution result: {{RESULT_PATH}}
Complete execution log: {{LOG_PATH}}
Artifacts directory: {{ARTIFACTS_PATH}}

Read those files. Inspect any expected result files named by the manifest. Then provide the
final reproduction status:

- selected experiment and configuration;
- exact command represented by the container launch script;
- paper-reported value;
- achieved value;
- whether the result matched;
- any execution failure, missing output, limitation, or assumption.

Do not launch or repeat the experiment. Base the final status only on the persisted manifest,
execution result, log, repository, and output artifacts.
