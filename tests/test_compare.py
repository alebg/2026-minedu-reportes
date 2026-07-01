"""Tests for the plagiarism detection engine in compare.py."""

from pathlib import Path

import pytest

import compare
from compare import (
    AlignedPassage,
    AlignmentSummary,
    ComparisonResult,
    FullReport,
    Passage,
    _build_ngram_index,
    _collect_matching_positions,
    _dedup_aligned,
    _extract_passages,
    _jaccard,
    _original_preview,
    _pct,
    _ratio,
    _slugify,
    _split_paragraphs,
    _strip_bibliography,
    _write_full_report,
    align_passages,
    compare_texts,
    extract_ngrams,
    normalize_with_spans,
)

# ---------------------------------------------------------------------------
# _pct / _ratio
# ---------------------------------------------------------------------------


class TestPctAndRatio:
    def test_pct_basic(self) -> None:
        assert _pct(1, 3) == 33.3

    def test_pct_zero_denominator(self) -> None:
        assert _pct(5, 0) == 0.0

    def test_pct_exact(self) -> None:
        assert _pct(1, 4) == 25.0

    def test_ratio_basic(self) -> None:
        assert _ratio(1, 3) == pytest.approx(1 / 3)

    def test_ratio_zero_denominator(self) -> None:
        assert _ratio(5, 0) == 0.0


# ---------------------------------------------------------------------------
# normalize_with_spans
# ---------------------------------------------------------------------------


class TestNormalizeWithSpans:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("HELLO World", ("hello", "world")),
            ("café résumé", ("cafe", "resume")),
            ("hello, world! (test)", ("hello", "world", "test")),
            ("", ()),
            ("...!!??", ()),
            ("  spaces   everywhere  ", ("spaces", "everywhere")),
        ],
    )
    def test_normalization(
        self,
        raw: str,
        expected: tuple[str, ...],
    ) -> None:
        assert normalize_with_spans(raw).words == expected

    def test_original_text_preserved(self) -> None:
        original = "Café Résumé"
        assert normalize_with_spans(original).original_text == original

    def test_all_spans_round_trip(self) -> None:
        original = "¡Hola, señor! ¿Cómo está?"
        result = normalize_with_spans(original)
        for word, (start, end) in zip(
            result.words, result.original_spans, strict=True,
        ):
            fragment = original[start:end]
            reconstructed = normalize_with_spans(fragment).words
            assert len(reconstructed) == 1
            assert reconstructed[0] == word

    def test_span_mapping_points_to_original(self) -> None:
        original = "Hello World"
        result = normalize_with_spans(original)
        start, end = result.original_spans[0]
        assert original[start:end] == "Hello"

    def test_span_mapping_with_accents(self) -> None:
        original = "más allá"
        result = normalize_with_spans(original)
        start, end = result.original_spans[0]
        assert original[start:end] == "más"

    def test_spans_cover_all_words(self) -> None:
        original = "Análisis de «plagio» — según método (n-gramas)"
        result = normalize_with_spans(original)
        assert len(result.words) == len(result.original_spans)
        for word, (start, end) in zip(
            result.words, result.original_spans, strict=True,
        ):
            assert start < end
            assert normalize_with_spans(original[start:end]).words[0] == word


# ---------------------------------------------------------------------------
# extract_ngrams
# ---------------------------------------------------------------------------


class TestExtractNgrams:
    @pytest.mark.parametrize(
        ("words", "n", "expected_len"),
        [
            (("a", "b", "c", "d", "e", "f", "g", "h", "i"), 8, 2),
            (("a", "b", "c"), 3, 1),
            (("a", "b"), 3, 0),
            ((), 8, 0),
        ],
    )
    def test_ngram_count(
        self,
        words: tuple[str, ...],
        n: int,
        expected_len: int,
    ) -> None:
        assert len(extract_ngrams(words, n=n)) == expected_len

    def test_ngram_content_and_positions(self) -> None:
        words = ("a", "b", "c", "d", "e", "f", "g", "h", "i")
        ngrams = extract_ngrams(words, n=8)
        assert ngrams[0] == (("a", "b", "c", "d", "e", "f", "g", "h"), 0)
        assert ngrams[1] == (("b", "c", "d", "e", "f", "g", "h", "i"), 1)


# ---------------------------------------------------------------------------
# _build_ngram_index
# ---------------------------------------------------------------------------


class TestBuildNgramIndex:
    def test_single_occurrence(self) -> None:
        ngrams = ((("a", "b", "c"), 0),)
        index = _build_ngram_index(ngrams)
        assert index[("a", "b", "c")] == (0,)

    def test_multiple_occurrences(self) -> None:
        gram = ("a", "b", "c")
        ngrams = ((gram, 0), (gram, 5))
        index = _build_ngram_index(ngrams)
        assert index[gram] == (0, 5)

    def test_distinct_grams(self) -> None:
        ngrams = ((("a", "b"), 0), (("c", "d"), 1))
        index = _build_ngram_index(ngrams)
        assert len(index) == 2


# ---------------------------------------------------------------------------
# _collect_matching_positions
# ---------------------------------------------------------------------------


class TestCollectMatchingPositions:
    def test_no_matches(self) -> None:
        gram_s: tuple[str, ...] = ("a", "b", "c")
        gram_t: tuple[str, ...] = ("x", "y", "z")
        source = ((gram_s, 0),)
        target_index: dict[tuple[str, ...], tuple[int, ...]] = {
            gram_t: (0,),
        }
        s_pos, t_pos, count = _collect_matching_positions(
            source, target_index, 3,
        )
        assert count == 0
        assert s_pos == frozenset()
        assert t_pos == frozenset()

    def test_one_match(self) -> None:
        gram: tuple[str, ...] = ("a", "b", "c")
        source = ((gram, 0),)
        target_index: dict[tuple[str, ...], tuple[int, ...]] = {gram: (5,)}
        s_pos, t_pos, count = _collect_matching_positions(
            source, target_index, 3,
        )
        assert count == 1
        assert s_pos == frozenset({0, 1, 2})
        assert t_pos == frozenset({5, 6, 7})

    def test_multiple_target_positions(self) -> None:
        gram: tuple[str, ...] = ("a", "b", "c")
        source = ((gram, 0),)
        target_index: dict[tuple[str, ...], tuple[int, ...]] = {
            gram: (5, 20),
        }
        s_pos, t_pos, count = _collect_matching_positions(
            source, target_index, 3,
        )
        assert count == 1
        assert t_pos == frozenset({5, 6, 7, 20, 21, 22})


# ---------------------------------------------------------------------------
# _original_preview
# ---------------------------------------------------------------------------


class TestOriginalPreview:
    def test_extracts_from_original(self) -> None:
        original = "Hola, ¿cómo estás hoy aquí en la ciudad?"
        normalized = normalize_with_spans(original)
        preview = _original_preview(normalized, 0, len(normalized.words) - 1)
        assert "Hola" in preview
        assert "ciudad" in preview

    def test_truncates_at_max_chars(self) -> None:
        original = "palabra " * 200
        normalized = normalize_with_spans(original.strip())
        preview = _original_preview(
            normalized, 0, len(normalized.words) - 1, max_chars=50,
        )
        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")


# ---------------------------------------------------------------------------
# _extract_passages
# ---------------------------------------------------------------------------


class TestExtractPassages:
    def test_empty_positions(self) -> None:
        normalized = normalize_with_spans("one two three four five")
        passages = _extract_passages(frozenset(), normalized, min_words=2)
        assert passages == ()

    def test_contiguous_block(self) -> None:
        text = "one two three four five six seven eight nine ten"
        normalized = normalize_with_spans(text)
        positions = frozenset(range(10))
        passages = _extract_passages(positions, normalized, min_words=5)
        assert len(passages) == 1
        assert passages[0].word_count == 10

    def test_gap_within_tolerance(self) -> None:
        text = "a b c d e f g h i j k l m n o p q r s t"
        normalized = normalize_with_spans(text)
        positions = frozenset({0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14})
        passages = _extract_passages(
            positions, normalized, min_words=5, max_gap=3,
        )
        assert len(passages) == 1

    def test_gap_exceeds_tolerance(self) -> None:
        text = "a b c d e f g h i j k l m n o p q r s t"
        normalized = normalize_with_spans(text)
        positions = frozenset({0, 1, 2, 3, 4, 10, 11, 12, 13, 14})
        passages = _extract_passages(
            positions, normalized, min_words=5, max_gap=3,
        )
        assert len(passages) == 2

    def test_filters_short_passages(self) -> None:
        text = "a b c d e f g h i j"
        normalized = normalize_with_spans(text)
        positions = frozenset({0, 1, 2})
        passages = _extract_passages(positions, normalized, min_words=5)
        assert passages == ()

    def test_sorted_by_length_descending(self) -> None:
        text = " ".join(f"w{i}" for i in range(30))
        normalized = normalize_with_spans(text)
        positions = frozenset(set(range(0, 10)) | set(range(20, 30)))
        passages = _extract_passages(positions, normalized, min_words=5)
        assert len(passages) == 2
        assert passages[0].word_count >= passages[1].word_count

    def test_passage_text_preview_comes_from_original(self) -> None:
        text = "El rápido zorro marrón salta sobre el perro perezoso lento"
        normalized = normalize_with_spans(text)
        positions = frozenset(range(len(normalized.words)))
        passages = _extract_passages(positions, normalized, min_words=5)
        assert len(passages) == 1
        assert "rápido" in passages[0].text_preview


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------


class TestJaccard:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (frozenset({("a", "b"), ("c", "d")}),
             frozenset({("a", "b"), ("c", "d")}),
             1.0),
            (frozenset({("a", "b")}),
             frozenset({("c", "d")}),
             0.0),
            (frozenset(),
             frozenset(),
             0.0),
        ],
    )
    def test_exact_values(
        self,
        a: frozenset[tuple[str, ...]],
        b: frozenset[tuple[str, ...]],
        expected: float,
    ) -> None:
        assert _jaccard(a, b) == expected

    def test_partial_overlap(self) -> None:
        a: frozenset[tuple[str, ...]] = frozenset({("a",), ("b",)})
        b: frozenset[tuple[str, ...]] = frozenset({("a",), ("c",)})
        assert _jaccard(a, b) == pytest.approx(1 / 3)

    def test_symmetric(self) -> None:
        a: frozenset[tuple[str, ...]] = frozenset({("a",), ("b",), ("c",)})
        b: frozenset[tuple[str, ...]] = frozenset({("b",), ("c",), ("d",)})
        assert _jaccard(a, b) == _jaccard(b, a)


# ---------------------------------------------------------------------------
# _split_paragraphs
# ---------------------------------------------------------------------------


class TestSplitParagraphs:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("first paragraph\n\nsecond paragraph",
             ("first paragraph", "second paragraph")),
            ("line one\nline two",
             ("line one", "line two")),
            ("hello\n\n\n\nworld",
             ("hello", "world")),
            ("", ()),
            ("   \n\n   ", ()),
        ],
    )
    def test_splitting(
        self,
        text: str,
        expected: tuple[str, ...],
    ) -> None:
        assert _split_paragraphs(text) == expected


# ---------------------------------------------------------------------------
# _strip_bibliography
# ---------------------------------------------------------------------------


class TestStripBibliography:
    def test_removes_from_last_occurrence_in_final_40pct(self) -> None:
        lines = ["content"] * 60 + ["REFERENCIAS BIBLIOGRÁFICAS", "ref1"]
        text = "\n".join(lines)
        result = _strip_bibliography(text)
        assert "ref1" not in result
        assert "content" in result

    def test_ignores_early_occurrence(self) -> None:
        lines = (
            ["REFERENCIAS BIBLIOGRÁFICAS"]
            + ["content"] * 100
        )
        text = "\n".join(lines)
        result = _strip_bibliography(text)
        assert result == text

    def test_no_bibliography_header(self) -> None:
        text = "just some text\nwith multiple lines\nno bibliography"
        assert _strip_bibliography(text) == text

    @pytest.mark.parametrize(
        "header",
        [
            "Bibliografía",
            "REFERENCIAS",
            "REFERENCIAS BIBLIOGRÁFICAS",
            "referencias bibliograficas",
            "bibliography",
        ],
    )
    def test_recognizes_header_variants(self, header: str) -> None:
        lines = ["content"] * 80 + [header, "ref1"]
        text = "\n".join(lines)
        result = _strip_bibliography(text)
        assert "ref1" not in result

    def test_picks_last_header_when_multiple_in_final_40pct(self) -> None:
        lines = (
            ["content"] * 60
            + ["REFERENCIAS", "early ref"]
            + ["more content"] * 5
            + ["Bibliografía", "late ref"]
        )
        text = "\n".join(lines)
        result = _strip_bibliography(text)
        assert "late ref" not in result
        assert "early ref" in result


# ---------------------------------------------------------------------------
# compare_texts (integration)
# ---------------------------------------------------------------------------


class TestCompareTexts:
    def test_identical_texts(self) -> None:
        text = "the quick brown fox jumps over the lazy dog repeatedly"
        result = compare_texts(text, text)
        assert result.source_overlap_pct == 100.0
        assert result.target_overlap_pct == 100.0
        assert result.matching_ngrams > 0

    def test_completely_different(self) -> None:
        source = "alpha beta gamma delta epsilon zeta eta theta"
        target = "one two three four five six seven eight"
        result = compare_texts(source, target)
        assert result.matching_ngrams == 0
        assert result.source_overlap_pct == 0.0
        assert result.target_overlap_pct == 0.0

    def test_partial_overlap(self) -> None:
        shared = "the quick brown fox jumps over the lazy dog"
        source = shared + " extra source words here today"
        target = shared + " different target words there now"
        result = compare_texts(source, target)
        assert result.source_overlap_pct > 0.0
        assert result.target_overlap_pct > 0.0
        assert result.matching_ngrams > 0

    def test_asymmetric_overlap(self) -> None:
        shared = "one two three four five six seven eight nine ten"
        source = shared
        target = shared + " " + " ".join(f"extra{i}" for i in range(50))
        result = compare_texts(source, target)
        assert result.source_overlap_pct > result.target_overlap_pct

    def test_empty_source(self) -> None:
        result = compare_texts("", "some text here")
        assert result.matching_ngrams == 0

    def test_passages_contain_original_text(self) -> None:
        shared = (
            "el rápido zorro marrón salta sobre el perro perezoso"
            " una y otra vez en el campo verde durante la mañana"
            " de un día cualquiera de primavera en el hemisferio"
        )
        source = shared
        target = shared
        result = compare_texts(source, target)
        assert len(result.passages) >= 1
        assert "rápido" in result.passages[0].text_preview
        assert result.passages[0].start_word == 0
        assert result.passages[0].word_count > 0

    def test_returns_correct_types(self) -> None:
        text = "a b c d e f g h i j k l m n o p q r s t"
        result = compare_texts(text, text)
        assert isinstance(result, ComparisonResult)
        assert isinstance(result.passages, tuple)
        for p in result.passages:
            assert isinstance(p, Passage)


# ---------------------------------------------------------------------------
# align_passages (integration)
# ---------------------------------------------------------------------------


class TestAlignPassages:
    def test_identical_paragraphs(self) -> None:
        text = (
            "el rápido zorro marrón salta sobre el perro perezoso una y otra vez\n\n"
            "la lluvia cae sobre los campos verdes del valle durante el invierno"
        )
        summary, aligned = align_passages(text, text)
        assert isinstance(summary, AlignmentSummary)
        assert summary.matched_paragraphs > 0
        assert summary.avg_similarity > 90.0

    def test_near_copy_with_edits(self) -> None:
        shared_a = "el rápido zorro marrón salta sobre el perro perezoso"
        shared_b = "una y otra vez en el campo verde durante la mañana"
        shared_c = "caminando por el sendero largo que cruza el bosque"
        source = f"{shared_a} {shared_b} {shared_c} al amanecer"
        target = f"{shared_a} {shared_b} {shared_c} al atardecer"
        summary, aligned = align_passages(source, target)
        assert summary.matched_paragraphs == 1
        assert 15.0 < aligned[0].similarity_pct < 100.0

    def test_no_match(self) -> None:
        source = "alfa beta gamma delta epsilon zeta eta theta iota kappa"
        target = "uno dos tres cuatro cinco seis siete ocho nueve diez"
        summary, aligned = align_passages(source, target)
        assert summary.matched_paragraphs == 0
        assert aligned == ()

    def test_empty_input(self) -> None:
        summary, aligned = align_passages("", "some text")
        assert summary.source_paragraphs == 0
        assert aligned == ()

    def test_returns_all_candidates(self) -> None:
        paras = [
            f"párrafo número {i} con texto suficiente para generar"
            " ocho gramas distintos"
            for i in range(5)
        ]
        text = "\n\n".join(paras)
        summary, aligned = align_passages(text, text)
        assert len(aligned) == summary.matched_paragraphs

    def test_summary_counts_are_consistent(self) -> None:
        text = (
            "el rápido zorro marrón salta sobre el perro perezoso una y otra vez\n\n"
            "la lluvia cae sobre los campos verdes del valle durante el invierno"
        )
        summary, aligned = align_passages(text, text)
        assert summary.matched_paragraphs == (
            summary.count_identical + summary.count_high + summary.count_moderate
        )

    def test_returns_correct_types(self) -> None:
        text = "el rápido zorro marrón salta sobre el perro perezoso una y otra vez"
        summary, aligned = align_passages(text, text)
        assert isinstance(summary, AlignmentSummary)
        for ap in aligned:
            assert isinstance(ap, AlignedPassage)


# ---------------------------------------------------------------------------
# _dedup_aligned
# ---------------------------------------------------------------------------


class TestDedupAligned:
    def test_deduplicates_by_target_preview(self) -> None:
        ap1 = AlignedPassage(
            source_preview="source a",
            target_preview="same target text here",
            source_words=10,
            target_words=10,
            similarity_pct=90.0,
        )
        ap2 = AlignedPassage(
            source_preview="source b",
            target_preview="same target text here",
            source_words=10,
            target_words=10,
            similarity_pct=80.0,
        )
        result = _dedup_aligned((ap1, ap2), top_n=10)
        assert len(result) == 1

    def test_respects_top_n(self) -> None:
        passages = tuple(
            AlignedPassage(
                source_preview=f"source {i}",
                target_preview=f"target {i}",
                source_words=10,
                target_words=10,
                similarity_pct=90.0 - i,
            )
            for i in range(10)
        )
        result = _dedup_aligned(passages, top_n=3)
        assert len(result) == 3

    def test_empty_input(self) -> None:
        assert _dedup_aligned((), top_n=5) == ()


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    @pytest.mark.parametrize(
        ("label", "substring"),
        [
            ("Producto 3 vs MINEDU", "producto_3_vs_minedu"),
            ("Reducción (Brechas)", "reducción"),
        ],
    )
    def test_slugify(self, label: str, substring: str) -> None:
        result = _slugify(label)
        assert substring in result
        assert "(" not in result
        assert " " not in result


# ---------------------------------------------------------------------------
# _write_full_report
# ---------------------------------------------------------------------------


class TestWriteFullReport:
    def test_creates_file_with_content(self, tmp_path: Path) -> None:
        original_dir = compare.OUTPUT_DIR
        compare.OUTPUT_DIR = tmp_path / "output"
        try:
            report = _make_dummy_report(
                passages=(
                    Passage(
                        start_word=0,
                        end_word=10,
                        word_count=11,
                        text_preview="sample passage text",
                    ),
                ),
                aligned=(
                    AlignedPassage(
                        source_preview="source text",
                        target_preview="target text",
                        source_words=10,
                        target_words=10,
                        similarity_pct=85.0,
                    ),
                ),
            )
            path = _write_full_report(report)
            assert path.exists()
            content = path.read_text()
            assert "FASE 1" in content
            assert "FASE 2" in content
            assert "test_label" in content
            assert "sample passage text" in content
            assert "source text" in content
            assert "target text" in content
            assert "85.0%" in content
        finally:
            compare.OUTPUT_DIR = original_dir


def _make_dummy_report(
    passages: tuple[Passage, ...] = (),
    aligned: tuple[AlignedPassage, ...] = (),
) -> FullReport:
    cr = ComparisonResult(
        total_ngrams_source=10,
        total_ngrams_target=10,
        matching_ngrams=5,
        source_words=20,
        target_words=20,
        source_words_in_overlap=10,
        target_words_in_overlap=10,
        source_overlap_pct=50.0,
        target_overlap_pct=50.0,
        passages=passages,
    )
    summary = AlignmentSummary(
        source_paragraphs=5,
        matched_paragraphs=2,
        matched_pct=40.0,
        avg_similarity=60.0,
        count_identical=0,
        count_high=1,
        count_moderate=1,
    )
    return FullReport(
        label="test_label",
        source_name="source_file.txt",
        target_name="target_file.txt",
        ngram_result=cr,
        body_only_result=cr,
        alignment_summary=summary,
        aligned_passages=aligned,
    )
