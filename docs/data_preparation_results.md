# WBS 2.0–2.7 — Data preparation evidence

Generated from the instructor-provided files by `run_data_preparation`; numeric summaries are computed, not hand-entered. Source files remain unchanged in `data/raw/`. The public test labels are not available and were not reconstructed.

## Input and contract checks (2.0)

| Split | Baskets | Distinct observed items | Mean basket size | Median | Min | Q25 | Q75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 8,300 | 168 | 4.244 | 3 | 1 | 1.0 | 6.0 | 29 |
| Test observed after target removal | 1,535 | 164 | 5.075 | 4 | 1 | 3.0 | 7.0 | 31 |

Both CSVs have exactly `basket_id, items`; their row counts match the assignment. IDs are unique within each split and the train/test ID intersection is zero. Test basket-size statistics describe only the remaining visible items after the target was removed, not original test basket sizes.

## Transaction parsing and matrix (2.0–2.1)

`items` is split on `;`, whitespace around tokens is trimmed, and empty tokens or missing values fail fast. Duplicate items in a basket are rejected rather than silently collapsed. The train vocabulary has 168 labels and the Boolean matrix has shape (8300, 168); each row sum equals the parsed original basket size. `basket_id` is the row index/identifier, never a feature.

The union vocabulary across the supplied train and visible test items has 169 labels. The test-only visible item(s) are baby food; supervised feature vocabularies must still be fitted from train only, and unseen test tokens must be recorded/handled without refitting on test.

## EDA evidence (2.2)

The top 20 train items and their basket supports are in `outputs/train_item_frequency.csv`. The basket-size count table is in `outputs/train_basket_size_distribution.csv`. Figures:

- `outputs/train_basket_size_distribution.png`
- `outputs/train_top20_items.png`
- `outputs/train_item_frequency_long_tail.png`

Train averages 4.244 items per basket, with median 3; the observed range is 1–29. These are descriptive statistics for the released split, not customer or time trends.

## Quality checks and decisions (2.3)

| issue | count | denominator | rate_percent | evidence | handling |
| --- | --- | --- | --- | --- | --- |
| Single-item train baskets | 2159 | 8300 | 26.0120 | Counted from original train basket sizes. | Retain for EDA/association; exclude only when simulating the provided test protocol. |
| Very large train baskets | 231 | 8300 | 2.7831 | Very-large threshold is the Tukey upper fence Q3 + 1.5×IQR = 13.50; counted strictly above the fence. | Inspect; do not delete automatically. |
| Repeated item within a train basket | 0 | 8300 | 0.0000 | Exact repeated tokens within one parsed basket. | Would need review; no automatic deduplication. |
| Duplicate train basket contents | 2608 | 8300 | 31.4217 | Order-insensitive item-set duplicates beyond the first; distinct basket IDs are retained. | Keep as separate transactions unless provenance proves an ingestion error. |
| Train/test basket_id overlap | 0 | 1535 | 0.0000 | Set intersection of the supplied identifiers. | Must be zero; otherwise stop before modeling. |
| Tokens with leading/trailing whitespace | 0 | 43013 | 0.0000 | Measured before strip(); parsed labels are trimmed for consistent tokenization. | Normalize boundary whitespace only; retain the raw CSV unchanged. |
| Item labels containing uppercase characters | 2 | 169 | 1.1834 | Exact labels differing from casefold(); observed labels: Instant food products, UHT-milk. UHT-milk uses an acronym; Instant food products is title-cased although the dictionary says lowercase. | Preserve exact source labels; confirm the dictionary convention with the group before any normalization. |

An order-insensitive content duplicate is reported as an additional occurrence after the first; such rows retain distinct basket IDs and are **not deleted**. Candidate similar-name pairs are listed in `outputs/similar_name_review_candidates.csv`; these are manual review prompts, not asserted duplicates. Exact spelling/case is preserved in the mapping. The two labels containing uppercase characters are `Instant food products` and `UHT-milk`; the latter is an acronym and should not be lowercased mechanically.

## Target prevalence and selection effect (2.4)

| population | baskets | target_positive_baskets | target_prevalence | target_prevalence_percent | difference_from_all_train_pp |
| --- | --- | --- | --- | --- | --- |
| All train baskets | 8300 | 1549 | 0.1866 | 18.6627 | 0.0000 |
| Train baskets with original size >= 2 | 6141 | 1487 | 0.2421 | 24.2143 | 5.5516 |

The provided test examples originate from original baskets with at least two items. Therefore validation first selects train baskets with original size ≥2, then separates the target label and removes `other vegetables` from the observed features. It must not select based on the post-removal size. The eligible-train prevalence is the more relevant prior for validation, threshold discussion, and expected Average Precision; it is not the hidden test prevalence and must not be presented as such. A threshold selected on all-train prevalence can be miscalibrated for this selected population.

## Product taxonomy (2.5–2.6)

The one-to-one mapping in `data/mappings/item_to_group.csv` covers exactly 169 observed labels with 23 groups. Grouping uses the product use/type implied by the supplied English item label; exact labels are preserved. Baby goods, household goods, media and garden products remain separate from food aisles.

| group | display_name | mapped_items | grouping_rule |
| --- | --- | --- | --- |
| alcoholic_beverages | Alcoholic beverages | 12 | Beer, spirits and wine categories. |
| baby_products | Baby products | 2 | Baby food and baby-care items retained together as a small dedicated aisle group. |
| bakery_bread | Bakery and bread | 10 | Bread, rolls and other bakery products, including shelf-stable bakery. |
| canned_preserved | Canned and preserved foods | 7 | Canned, pickled, preserved and sweet-spread products. |
| condiments_cooking | Condiments and cooking ingredients | 11 | Cooking ingredients, sauces, seasoning and dressing. |
| dairy_eggs | Dairy and eggs | 19 | Milk, cultured dairy, cheese and eggs; margarine is grouped with dairy-case spreads. |
| fish_seafood | Fish and seafood | 2 | Fish products not explicitly labeled as canned. |
| fresh_fruit | Fresh fruit | 5 | Fresh fruit categories named in the source labels. |
| fresh_vegetables | Fresh vegetables | 5 | Vegetable categories and potato products, grouped by the main produce ingredient. |
| frozen_foods | Frozen foods | 7 | Items explicitly labeled frozen plus ice cream. |
| garden_plants | Garden and plants | 3 | Planting seeds, soil/fertilizer and pot plants. |
| household_cleaning | Household cleaning | 10 | Cleaning agents and household cleaning/maintenance products. |
| household_kitchen | Household and kitchen | 10 | Kitchen consumables, utensils, household utility items and bags. |
| meat_poultry | Meat and poultry | 11 | Fresh or processed meat and poultry, including sausages. |
| media_stationery | Media and stationery | 3 | Newspapers and photo/film/sound-storage media. |
| nonalcoholic_beverages | Non-alcoholic beverages | 9 | Non-alcoholic drinks and beverage ingredients; miscellaneous beverages are provisionally placed here. |
| packaged_produce | Packaged fruit and vegetables | 1 | Mixed packaged fruit/vegetable item kept separate because its contents/form are underspecified. |
| pantry_baking_grains | Pantry, baking and grains | 8 | Dry grains, flour, baking ingredients and shelf-stable meal base. |
| personal_care_hygiene | Personal care and hygiene | 8 | Personal cosmetics, hygiene and sanitary-care items. |
| pet_products | Pet products | 3 | Pet food and pet-care products. |
| ready_to_eat_meals | Ready-to-eat foods and soups | 5 | Prepared foods, soups and ready-to-eat spreads/products. |
| snacks_sweets | Snacks and sweets | 15 | Confectionery, packaged snacks, snack bars and sweet toppings. |
| specialty_miscellaneous | Specialty and miscellaneous groceries | 3 | Source labels whose product identity is not specified; kept as a small explicit group rather than spread into arbitrary aisles. |

Items with plausible alternative assignments, rationale, and the planned quantitative comparison are in `data/mappings/item_group_alternatives.csv`. Examples include `frozen dessert`, `finished products`, `rubbing alcohol`, `organic products`, `misc. beverages`, `margarine`, `potato products`, and `syrup`. These choices can change category-level support and rules; no numerical effect is claimed here. The actual rule comparison is deferred to WBS 3.5, where primary and alternate mappings must be run on the same transactions and support settings.

## Data Gate / handoff (2.7)

| check | status | evidence |
| --- | --- | --- |
| Provided train/test schemas and expected row counts | PASS | train=8,300 and test=1,535; both have exactly basket_id,items. |
| basket_id unique within each split | PASS | train unique=True; test unique=True. |
| Train/test basket_id disjoint | PASS | intersection size=0. |
| Items parse and matrix row sums match train basket sizes | PASS | Boolean matrix shape=(8300, 168); every row sum equals original train basket size. |
| Taxonomy covers exact observed union vocabulary | PASS | 169 unique labels mapped across 23 groups; union vocabulary=169. |
| Test contains no target token | PASS | Checks only the released items field; this is not a test label and does not infer the hidden target. |
| Test target labels are absent from input schema | PASS | Raw fields=['basket_id', 'items']; target label column is not present. |
| No test-only vocabulary used to fit supervised item vocabulary | PASS | Train matrix uses train-only vocabulary (168 items); test-only items recorded for OOV handling: ['baby food']. |
| basket_id excluded from transaction features | PASS | basket_id is retained only as the matrix row index/identifier. |
| Target removed from simulated supervised features | PASS | simulate_test_observation selects original size >=2 before removing target; it returns labels separately. |
| Preprocessing fit inside each future CV fold | NOT VERIFIED | No supervised model/CV pipeline is part of WBS 2.0-2.7; verify in WBS 4.x before reporting model metrics. |
| No external or public labeled data used | NOT VERIFIED | This is a provenance/process requirement; preserve the instructor-provided source files and team activity log. |
| Original data files unchanged | NOT VERIFIED | The project pipeline reads raw CSVs without writing to data/raw; verify original hashes if files are moved or replaced. |

Reusable entry points are `load_assignment_data`, `build_transaction_matrix`, `simulate_test_observation`, and `make_group_presence_matrix` in `groceries_group3.preparation.transactions`. `train_baskets.csv.gz`, `test_baskets.csv.gz`, and Boolean matrices are reproducible local derivatives under `data/processed/`, ignored by Git; regenerate them from the instructor files rather than sharing raw data. The train transaction matrix includes the target for association/EDA; a supervised model must explicitly exclude the target from its feature matrix. Group-presence summaries are descriptive only and must be rebuilt with target exclusion for classification.

### Remaining gate items

- Group review of the prepared artifacts is pending.
- External-data provenance and preservation of original source files are process checks and cannot be proven from the numeric pipeline alone.
- Fold-local learned preprocessing must be demonstrated in the later classification task.
- Quantitative taxonomy sensitivity is pending WBS 3.5.
