## Image Generation Plan (Target: 500 images per class)

Current dataset counts (from analyze):

- bird_drop: 201
- clean: 191
- dusty: 182
- electrical_damage: 90
- physical_damage: 66
- snow_covered: 114

Goal: Each class totals 500 images. Generate the missing images per class and split across the three methods with Structural Consistency prioritized.

Method guidance:
- Structural Consistency (SC): Use cleaned images as structural references; preserve panel layout/geometry, inject target condition.
- Domain Adaptation (DA): Use images from other classes as sources; adapt style/appearance to the selected class.
- Text-to-Image (T2I): Direct text prompts to synthesize realistic solar panel scenes with the target condition.

Split policy:
- Prioritize SC; base split around approximately 60% SC, ~13.33% DA, remainder to T2I.
- For bird_drop, align SC closely to the provided example while hitting the exact total of 500.

### Per-class generation counts (to reach 500)

| Class               | Current | Missing | Structural Consistency | Domain Adaptation | Text-to-Image | Final Total |
|---------------------|---------|---------|------------------------|-------------------|---------------|-------------|
| bird_drop           | 201     | 299     | 180                    | 40                | 79            | 500         |
| clean               | 191     | 309     | 185                    | 41                | 83            | 500         |
| dusty               | 182     | 318     | 190                    | 42                | 86            | 500         |
| electrical_damage   | 90      | 410     | 246                    | 55                | 109           | 500         |
| physical_damage     | 66      | 434     | 260                    | 58                | 116           | 500         |
| snow_covered        | 114     | 386     | 231                    | 51                | 104           | 500         |

Totals to generate: 2,156

- Structural Consistency: 1,292
- Domain Adaptation: 287
- Text-to-Image: 577

### Execution notes

- Structure references: sample from `datasets/clean/` for SC guidance.
- Domain adaptation: sample sources from classes other than the target class.
- Save outputs under `generated/<class>/<method>/...` and track in a manifest.
- If any class needs slight rounding adjustments, keep SC priority and adjust T2I first.


