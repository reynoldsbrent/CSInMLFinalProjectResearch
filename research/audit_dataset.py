
"""
RAGTruth Dataset Audit
======================

Run from the research project root:

    python research/audit_dataset.py

Optional custom paths:

    python research/audit_dataset.py \
        --dataset-dir RAGTruth\dataset \
        --output-dir research\audit_results

This script does not modify the original dataset.
It writes diagnostic reports to the output directory.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_RESPONSE_FIELDS = {
    "id",
    "source_id",
    "model",
    "temperature",
    "labels",
    "split",
    "quality",
    "response",
}

REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "task_type",
    "source",
    "source_info",
    "prompt",
}

EXPECTED_SPLITS = {"train", "test"}
EXPECTED_TASKS = {"QA", "Summary", "Data2txt"}


def normalize_id(value):
    """Normalize IDs for comparison without changing the source files."""
    if value is None:
        return None
    return str(value).strip()


def read_jsonl(file_path, issues):
    """Read a JSONL file and record parsing problems."""
    records = []

    if not file_path.exists():
        issues.append(f"Missing file: {file_path}")
        return records

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                issues.append(
                    f"{file_path.name}: blank line {line_number}"
                )
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(
                    f"{file_path.name}: invalid JSON at line "
                    f"{line_number}: {exc}"
                )
                continue

            if not isinstance(record, dict):
                issues.append(
                    f"{file_path.name}: line {line_number} "
                    "is not a JSON object"
                )
                continue

            record["_audit_line"] = line_number
            records.append(record)

    return records


def write_csv(file_path, rows):
    """Write dictionaries to a CSV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Write an empty file when no issues were found.
        file_path.write_text("", encoding="utf-8")
        return

    columns = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))

    with file_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def count_duplicates(values):
    """Return values that occur more than once."""
    counts = Counter(v for v in values if v is not None)
    return {
        str(value): count
        for value, count in counts.items()
        if count > 1
    }


def check_required_fields(records, required_fields):
    """Return records with missing required fields."""
    problems = []

    for record in records:
        missing = required_fields - set(record)

        if missing:
            problems.append({
                "line": record.get("_audit_line"),
                "id": record.get("id"),
                "source_id": record.get("source_id"),
                "missing_fields": ", ".join(sorted(missing)),
            })

    return problems


def audit_annotations(responses):
    """Check annotation structures, offsets, and text consistency."""
    total_spans = 0
    malformed_label_rows = 0
    malformed_spans = 0
    missing_fields = 0
    offsets_out_of_range = 0
    text_mismatches = 0
    implicit_true_spans = 0
    invalid_implicit_true_values = 0

    label_type_counts = Counter()
    annotation_issues = []

    for row in responses:
        response_id = normalize_id(row.get("id"))
        response_text = row.get("response")
        labels = row.get("labels")

        if not isinstance(labels, list):
            malformed_label_rows += 1
            annotation_issues.append({
                "response_id": response_id,
                "issue": "labels_not_list",
                "details": f"Actual type: {type(labels).__name__}",
            })
            continue

        for index, label in enumerate(labels):
            total_spans += 1

            if not isinstance(label, dict):
                malformed_spans += 1
                annotation_issues.append({
                    "response_id": response_id,
                    "issue": "annotation_not_object",
                    "details": f"Annotation index: {index}",
                })
                continue

            required = {"start", "end", "text", "label_type"}
            missing = required - set(label)

            if missing:
                missing_fields += 1
                annotation_issues.append({
                    "response_id": response_id,
                    "issue": "missing_annotation_fields",
                    "details": ", ".join(sorted(missing)),
                })

            label_type_counts[str(label.get("label_type"))] += 1

            implicit_true = label.get("implicit_true")
            if implicit_true is True:
                implicit_true_spans += 1
            elif implicit_true not in (None, False):
                invalid_implicit_true_values += 1
                annotation_issues.append({
                    "response_id": response_id,
                    "issue": "unexpected_implicit_true_value",
                    "details": repr(implicit_true),
                })

            start = label.get("start")
            end = label.get("end")
            text = label.get("text")

            # bool is a subclass of int in Python, so exclude it explicitly.
            valid_offsets = (
                isinstance(start, int) and not isinstance(start, bool)
                and isinstance(end, int) and not isinstance(end, bool)
            )

            if not valid_offsets:
                if start is not None or end is not None:
                    annotation_issues.append({
                        "response_id": response_id,
                        "issue": "invalid_offset_type",
                        "details": f"start={start!r}, end={end!r}",
                    })
                continue

            if not isinstance(response_text, str):
                continue

            if start < 0 or end < start or end > len(response_text):
                offsets_out_of_range += 1
                annotation_issues.append({
                    "response_id": response_id,
                    "issue": "offset_out_of_range",
                    "details": (
                        f"start={start}, end={end}, "
                        f"response_length={len(response_text)}"
                    ),
                })
                continue

            if isinstance(text, str):
                extracted_text = response_text[start:end]

                if extracted_text != text:
                    text_mismatches += 1
                    annotation_issues.append({
                        "response_id": response_id,
                        "issue": "annotation_text_mismatch",
                        "details": (
                            f"start={start}, end={end}; "
                            f"annotated={text!r}; "
                            f"extracted={extracted_text!r}"
                        ),
                    })

    summary = {
        "total_annotation_spans": total_spans,
        "response_rows_with_non_list_labels": malformed_label_rows,
        "malformed_annotation_spans": malformed_spans,
        "annotations_missing_required_fields": missing_fields,
        "offsets_out_of_range": offsets_out_of_range,
        "annotation_text_mismatches": text_mismatches,
        "implicit_true_spans": implicit_true_spans,
        "unexpected_implicit_true_values": invalid_implicit_true_values,
        "label_type_counts": dict(label_type_counts),
    }

    return summary, annotation_issues


def main():
    parser = argparse.ArgumentParser(
        description="Audit the RAGTruth dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default="RAGTruth/dataset",
        help="Directory containing response.jsonl and source_info.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="research/audit_results",
        help="Directory for generated audit reports",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsing_issues = []

    responses = read_jsonl(
        dataset_dir / "response.jsonl", parsing_issues
    )
    sources = read_jsonl(
        dataset_dir / "source_info.jsonl", parsing_issues
    )

    if not responses or not sources:
        print("ERROR: One or both dataset files are empty or unavailable.")
        print(f"Dataset directory: {dataset_dir.resolve()}")
        print("Check the file paths and try again.")
        return

    # ---------------------------------------------------------
    # 1. Schema checks
    # ---------------------------------------------------------
    response_schema_issues = check_required_fields(
        responses, REQUIRED_RESPONSE_FIELDS
    )
    source_schema_issues = check_required_fields(
        sources, REQUIRED_SOURCE_FIELDS
    )

    # ---------------------------------------------------------
    # 2. IDs and metadata joins
    # ---------------------------------------------------------
    response_ids = [
        normalize_id(row.get("id")) for row in responses
    ]
    metadata_ids = [
        normalize_id(row.get("source_id")) for row in sources
    ]

    duplicate_response_ids = count_duplicates(response_ids)
    duplicate_metadata_ids = count_duplicates(metadata_ids)

    source_map = defaultdict(list)

    for source in sources:
        source_map[normalize_id(source.get("source_id"))].append(source)

    missing_metadata = sorted({
        normalize_id(row.get("source_id"))
        for row in responses
        if normalize_id(row.get("source_id")) not in source_map
    }, key=str)

    duplicated_metadata_keys = sorted(
        [
            source_id
            for source_id, rows in source_map.items()
            if len(rows) > 1
        ],
        key=str,
    )

    # ---------------------------------------------------------
    # 3. Split, quality, task, and model distributions
    # ---------------------------------------------------------
    split_counts = Counter()
    quality_counts = Counter()
    model_counts = Counter()
    task_counts = Counter()
    split_task_counts = Counter()
    split_quality_counts = Counter()
    label_counts = Counter()
    good_quality_label_counts = Counter()

    response_counts_by_source = Counter()
    splits_by_source = defaultdict(set)
    task_by_source = {}
    invalid_splits = []
    invalid_tasks = []
    missing_or_empty_responses = []
    invalid_label_rows = []

    for row in responses:
        response_id = normalize_id(row.get("id"))
        source_id = normalize_id(row.get("source_id"))
        split = row.get("split")
        quality = row.get("quality")
        model = row.get("model")
        response_text = row.get("response")
        labels = row.get("labels")

        split_counts[str(split)] += 1
        quality_counts[str(quality)] += 1
        model_counts[str(model)] += 1
        split_quality_counts[(str(split), str(quality))] += 1

        if split not in EXPECTED_SPLITS:
            invalid_splits.append({
                "response_id": response_id,
                "split": split,
            })

        if not isinstance(response_text, str) or not response_text.strip():
            missing_or_empty_responses.append({
                "response_id": response_id,
                "issue": "missing_or_empty_response",
            })

        if not isinstance(labels, list):
            invalid_label_rows.append({
                "response_id": response_id,
                "issue": "labels_not_list",
            })
        else:
            has_hallucination = bool(labels)
            label_counts[(str(split), has_hallucination)] += 1

            if quality == "good":
                good_quality_label_counts[
                    (str(split), has_hallucination)
                ] += 1

        response_counts_by_source[source_id] += 1
        splits_by_source[source_id].add(split)

        matching_sources = source_map.get(source_id, [])

        if matching_sources:
            # Don't arbitrarily choose one if metadata IDs are duplicated.
            if len(matching_sources) == 1:
                task = matching_sources[0].get("task_type")
                task_by_source[source_id] = task
                task_counts[str(task)] += 1
                split_task_counts[(str(split), str(task))] += 1

                if task not in EXPECTED_TASKS:
                    invalid_tasks.append({
                        "response_id": response_id,
                        "source_id": source_id,
                        "task_type": task,
                    })

    # ---------------------------------------------------------
    # 4. Official split overlap
    # ---------------------------------------------------------
    train_response_ids = {
        normalize_id(row.get("id"))
        for row in responses if row.get("split") == "train"
    }
    test_response_ids = {
        normalize_id(row.get("id"))
        for row in responses if row.get("split") == "test"
    }

    train_source_ids = {
        normalize_id(row.get("source_id"))
        for row in responses if row.get("split") == "train"
    }
    test_source_ids = {
        normalize_id(row.get("source_id"))
        for row in responses if row.get("split") == "test"
    }

    response_overlap = train_response_ids & test_response_ids
    source_overlap = train_source_ids & test_source_ids

    mixed_split_sources = [
        {
            "source_id": source_id,
            "task_type": task_by_source.get(
                source_id, "UNKNOWN_OR_DUPLICATE_METADATA"
            ),
            "response_count": response_counts_by_source[source_id],
            "splits": ", ".join(
                sorted(str(x) for x in splits_by_source[source_id])
            ),
        }
        for source_id in sorted(splits_by_source, key=str)
        if len(splits_by_source[source_id]) > 1
    ]

    # ---------------------------------------------------------
    # 5. Source group sizes
    # ---------------------------------------------------------
    source_group_rows = []

    for source_id in sorted(response_counts_by_source, key=str):
        source_group_rows.append({
            "source_id": source_id,
            "task_type": task_by_source.get(
                source_id, "UNKNOWN_OR_DUPLICATE_METADATA"
            ),
            "response_count": response_counts_by_source[source_id],
            "splits": ", ".join(
                sorted(str(x) for x in splits_by_source[source_id])
            ),
        })

    group_size_counts = Counter(
        response_counts_by_source.values()
    )

    # ---------------------------------------------------------
    # 6. Annotation validation
    # ---------------------------------------------------------
    annotation_summary, annotation_issues = audit_annotations(responses)

    # ---------------------------------------------------------
    # 7. Save CSV reports
    # ---------------------------------------------------------
    write_csv(
        output_dir / "response_schema_issues.csv",
        response_schema_issues,
    )
    write_csv(
        output_dir / "source_schema_issues.csv",
        source_schema_issues,
    )
    write_csv(
        output_dir / "annotation_issues.csv",
        annotation_issues,
    )
    write_csv(
        output_dir / "mixed_split_sources.csv",
        mixed_split_sources,
    )
    write_csv(
        output_dir / "invalid_split_rows.csv",
        invalid_splits,
    )
    write_csv(
        output_dir / "invalid_task_rows.csv",
        invalid_tasks,
    )
    write_csv(
        output_dir / "missing_or_empty_responses.csv",
        missing_or_empty_responses,
    )
    write_csv(
        output_dir / "invalid_label_rows.csv",
        invalid_label_rows,
    )
    write_csv(
        output_dir / "source_groups.csv",
        source_group_rows,
    )

    # ---------------------------------------------------------
    # 8. Save JSON summary
    # ---------------------------------------------------------
    def counter_to_dict(counter):
        return {str(key): value for key, value in counter.items()}

    summary = {
        "record_counts": {
            "responses": len(responses),
            "source_metadata_records": len(sources),
            "unique_response_ids": len(set(response_ids)),
            "unique_source_ids_in_metadata": len(set(metadata_ids)),
            "unique_source_ids_in_responses": len(
                set(normalize_id(r.get("source_id")) for r in responses)
            ),
        },
        "schema": {
            "response_rows_missing_required_fields": len(
                response_schema_issues
            ),
            "source_rows_missing_required_fields": len(
                source_schema_issues
            ),
        },
        "ids_and_metadata": {
            "duplicate_response_ids": duplicate_response_ids,
            "duplicate_metadata_source_ids": duplicate_metadata_ids,
            "responses_missing_metadata_count": len(missing_metadata),
            "responses_missing_metadata_sample": missing_metadata[:100],
            "duplicate_metadata_keys": duplicated_metadata_keys,
        },
        "official_splits": {
            "response_counts": counter_to_dict(split_counts),
            "train_test_response_id_overlap_count": len(response_overlap),
            "train_test_response_id_overlap_sample": sorted(
                response_overlap
            )[:100],
            "train_test_source_id_overlap_count": len(source_overlap),
            "train_test_source_id_overlap_sample": sorted(
                source_overlap, key=str
            )[:100],
            "sources_appearing_in_multiple_splits": len(
                mixed_split_sources
            ),
            "invalid_split_rows": len(invalid_splits),
        },
        "quality": {
            "quality_counts": counter_to_dict(quality_counts),
            "split_quality_counts": {
                f"{split}|{quality}": count
                for (split, quality), count
                in split_quality_counts.items()
            },
        },
        "task_and_model_distributions": {
            "task_counts": counter_to_dict(task_counts),
            "split_task_counts": {
                f"{split}|{task}": count
                for (split, task), count in split_task_counts.items()
            },
            "model_counts": counter_to_dict(model_counts),
            "invalid_task_rows": len(invalid_tasks),
        },
        "response_level_label_proxy": {
            "definition": (
                "has_hallucination = labels is a nonempty list; "
                "this is a proxy based on annotations, not a new "
                "independent ground-truth assessment"
            ),
            "all_quality_label_counts": {
                f"{split}|has_hallucination={label}": count
                for (split, label), count in label_counts.items()
            },
            "good_quality_label_counts": {
                f"{split}|has_hallucination={label}": count
                for (split, label), count in good_quality_label_counts.items()
            },
        },
        "source_group_sizes": {
            "responses_per_source_distribution": {
                str(size): count
                for size, count in sorted(group_size_counts.items())
            },
            "sources_with_unexpected_response_count": sum(
                count for size, count in group_size_counts.items()
                if size != 6
            ),
        },
        "annotation_validation": annotation_summary,
        "response_text_validation": {
            "missing_or_empty_responses": len(missing_or_empty_responses),
            "invalid_label_rows": len(invalid_label_rows),
        },
        "parsing_issues": parsing_issues,
    }

    with (output_dir / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------
    # 9. Print concise summary
    # ---------------------------------------------------------
    print("\n" + "=" * 58)
    print("RAGTRUTH DATASET AUDIT")
    print("=" * 58)
    print(f"Response records: {len(responses):,}")
    print(f"Source metadata records: {len(sources):,}")
    print(f"Duplicate response IDs: {len(duplicate_response_ids)}")
    print(f"Duplicate metadata source IDs: {len(duplicate_metadata_ids)}")
    print(f"Responses missing metadata: {len(missing_metadata)}")
    print(f"Train/test response ID overlap: {len(response_overlap)}")
    print(f"Train/test source ID overlap: {len(source_overlap)}")
    print(f"Sources appearing in multiple splits: {len(mixed_split_sources)}")
    print(
        "Annotation spans: "
        f"{annotation_summary['total_annotation_spans']:,}"
    )
    print(
        "Annotation offsets out of range: "
        f"{annotation_summary['offsets_out_of_range']}"
    )
    print(
        "Annotation text mismatches: "
        f"{annotation_summary['annotation_text_mismatches']}"
    )
    print(f"JSONL parsing issues: {len(parsing_issues)}")
    print(f"\nReports saved to: {output_dir.resolve()}")
    print("Review audit_summary.json and the CSV issue reports.")
    print(
        "\nNote: findings are diagnostics. Investigate anomalies "
        "before changing data or finalizing the experiment."
    )


if __name__ == "__main__":
    main()
