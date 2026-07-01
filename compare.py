"""Plagiarism detection via n-gram fingerprinting.

Compares consulting deliverables against official publications
to find verbatim and near-verbatim text reuse.

Two-phase report per document pair:
  Phase 1 — Fast 8-gram scan with overlap percentages and top passages.
  Phase 2 — Side-by-side alignment of each source passage with its best
            match in the target, plus body-only metrics (bibliography
            stripped).
"""

import logging
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import attrs

logger = logging.getLogger(__name__)

def _pct(numerator: int, denominator: int) -> float:
    """Compute a percentage via Decimal to avoid floating-point drift."""
    if denominator == 0:
        return 0.0
    return float(round(Decimal(numerator) / Decimal(denominator) * 100, 1))


def _ratio(numerator: int, denominator: int) -> float:
    """Compute a ratio via Decimal to avoid floating-point drift."""
    if denominator == 0:
        return 0.0
    return float(Decimal(numerator) / Decimal(denominator))


NGRAM_SIZE = 8
MIN_PASSAGE_WORDS = 15
MAX_GAP = 3
SIDE_BY_SIDE_TOP_N = 20
PREVIEW_MAX_CHARS = 500

BIBLIO_HEADERS = frozenset((
    "referencias bibliograficas",
    "referencias",
    "bibliografía",
    "bibliografia",
    "references",
    "bibliography",
))


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@attrs.define(frozen=True, slots=True)
class Passage:
    """A contiguous span of matching text."""

    start_word: int
    end_word: int
    word_count: int
    text_preview: str


@attrs.define(frozen=True, slots=True)
class AlignedPassage:
    """A source passage paired with its best-matching target passage."""

    source_preview: str
    target_preview: str
    source_words: int
    target_words: int
    similarity_pct: float


@attrs.define(frozen=True, slots=True)
class AlignmentSummary:
    """Aggregate metrics for paragraph-level alignment."""

    source_paragraphs: int
    matched_paragraphs: int
    matched_pct: float
    avg_similarity: float
    count_identical: int
    count_high: int
    count_moderate: int


@attrs.define(frozen=True, slots=True)
class NormalizedText:
    """Normalized words with a mapping back to original character spans."""

    words: tuple[str, ...]
    original_spans: tuple[tuple[int, int], ...]
    original_text: str


@attrs.define(frozen=True, slots=True)
class ComparisonResult:
    """Overlap metrics and matched passages between two documents."""

    total_ngrams_source: int
    total_ngrams_target: int
    matching_ngrams: int
    source_words: int
    target_words: int
    source_words_in_overlap: int
    target_words_in_overlap: int
    source_overlap_pct: float
    target_overlap_pct: float
    passages: tuple[Passage, ...]


@attrs.define(frozen=True, slots=True)
class FullReport:
    """Complete report for one document pair."""

    label: str
    source_name: str
    target_name: str
    ngram_result: ComparisonResult
    body_only_result: ComparisonResult
    alignment_summary: AlignmentSummary
    aligned_passages: tuple[AlignedPassage, ...]


@attrs.define(frozen=True, slots=True)
class DocumentPair:
    """A pair of documents to compare."""

    source_path: Path
    target_path: Path
    label: str


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_with_spans(text: str) -> NormalizedText:
    """Normalize text and track each normalized word back to original chars."""
    decomposed = unicodedata.normalize("NFKD", text)
    char_map: list[int] = []
    cleaned_chars: list[str] = []
    orig_idx = 0
    for char in decomposed:
        if unicodedata.combining(char):
            continue
        lowered = char.lower()
        if re.match(r"[^\w\s]", lowered):
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(lowered)
        char_map.append(orig_idx)
        orig_idx += 1

    cleaned = "".join(cleaned_chars)

    words: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", cleaned):
        words.append(match.group())
        start_orig = char_map[match.start()]
        end_orig = char_map[match.end() - 1] + 1
        spans.append((start_orig, end_orig))

    return NormalizedText(
        words=tuple(words),
        original_spans=tuple(spans),
        original_text=text,
    )


# ---------------------------------------------------------------------------
# Bibliography stripping
# ---------------------------------------------------------------------------

def _strip_bibliography(text: str) -> str:
    """Remove everything from the last bibliography header to end of text.

    Uses the last occurrence to avoid hitting TOC entries near the top.
    Only matches headers that appear in the final 40% of the document.
    """
    lines = text.split("\n")
    cutoff = len(lines) * 6 // 10
    last_biblio_line = -1

    for i in range(len(lines) - 1, cutoff - 1, -1):
        normalized_line = unicodedata.normalize("NFKD", lines[i].strip())
        cleaned = "".join(
            c for c in normalized_line if not unicodedata.combining(c)
        ).lower().strip()
        cleaned = re.sub(r"[^\w\s]", "", cleaned).strip()
        if cleaned in BIBLIO_HEADERS:
            last_biblio_line = i
            break

    if last_biblio_line >= 0:
        return "\n".join(lines[:last_biblio_line])
    return text


# ---------------------------------------------------------------------------
# N-gram engine
# ---------------------------------------------------------------------------

def extract_ngrams(
    words: tuple[str, ...],
    n: int = NGRAM_SIZE,
) -> tuple[tuple[tuple[str, ...], int], ...]:
    """Return tuple of (ngram, start_word_index) from word sequence."""
    return tuple(
        (tuple(words[i : i + n]), i)
        for i in range(len(words) - n + 1)
    )


def _build_ngram_index(
    ngrams: tuple[tuple[tuple[str, ...], int], ...],
) -> dict[tuple[str, ...], tuple[int, ...]]:
    """Build a lookup from n-gram to all positions where it occurs."""
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for gram, pos in ngrams:
        index[gram].append(pos)
    return {gram: tuple(positions) for gram, positions in index.items()}


def _collect_matching_positions(
    source_ngrams: tuple[tuple[tuple[str, ...], int], ...],
    target_index: dict[tuple[str, ...], tuple[int, ...]],
    ngram_size: int,
) -> tuple[frozenset[int], frozenset[int], int]:
    """Find all word positions that participate in a matching n-gram.

    Returns:
        Tuple of (source_positions, target_positions, match_count).
    """
    source_positions: set[int] = set()
    target_positions: set[int] = set()
    match_count = 0

    for gram, s_pos in source_ngrams:
        if gram in target_index:
            match_count += 1
            source_positions.update(range(s_pos, s_pos + ngram_size))
            for t_pos in target_index[gram]:
                target_positions.update(range(t_pos, t_pos + ngram_size))

    return frozenset(source_positions), frozenset(target_positions), match_count


def _original_preview(
    normalized: NormalizedText,
    start_word: int,
    end_word: int,
    max_chars: int = PREVIEW_MAX_CHARS,
) -> str:
    """Extract a preview from the original text using word span mapping."""
    char_start = normalized.original_spans[start_word][0]
    char_end = normalized.original_spans[end_word][1]
    raw = normalized.original_text[char_start:char_end]
    collapsed = re.sub(r"\s+", " ", raw).strip()
    if len(collapsed) > max_chars:
        return collapsed[:max_chars] + "..."
    return collapsed


def _extract_passages(
    matching_positions: frozenset[int],
    normalized: NormalizedText,
    min_words: int = MIN_PASSAGE_WORDS,
    max_gap: int = MAX_GAP,
) -> tuple[Passage, ...]:
    """Group contiguous matching positions into passages."""
    if not matching_positions:
        return ()

    sorted_pos = sorted(matching_positions)
    passages: list[Passage] = []
    start = sorted_pos[0]
    end = sorted_pos[0]

    for pos in sorted_pos[1:]:
        if pos <= end + max_gap:
            end = pos
        else:
            length = end - start + 1
            if length >= min_words:
                passages.append(Passage(
                    start_word=start,
                    end_word=end,
                    word_count=length,
                    text_preview=_original_preview(normalized, start, end),
                ))
            start = pos
            end = pos

    length = end - start + 1
    if length >= min_words:
        passages.append(Passage(
            start_word=start,
            end_word=end,
            word_count=length,
            text_preview=_original_preview(normalized, start, end),
        ))

    return tuple(sorted(passages, key=lambda p: -p.word_count))


def compare_texts(
    source_text: str,
    target_text: str,
    ngram_size: int = NGRAM_SIZE,
) -> ComparisonResult:
    """Compare two texts and return overlap metrics with matched passages."""
    s_normalized = normalize_with_spans(source_text)
    t_normalized = normalize_with_spans(target_text)

    s_ngrams = extract_ngrams(s_normalized.words, ngram_size)
    t_ngrams = extract_ngrams(t_normalized.words, ngram_size)

    t_index = _build_ngram_index(t_ngrams)
    s_positions, t_positions, match_count = _collect_matching_positions(
        s_ngrams, t_index, ngram_size,
    )

    s_count = len(s_normalized.words)
    t_count = len(t_normalized.words)

    passages = _extract_passages(s_positions, s_normalized)

    return ComparisonResult(
        total_ngrams_source=len(s_ngrams),
        total_ngrams_target=len(t_ngrams),
        matching_ngrams=match_count,
        source_words=s_count,
        target_words=t_count,
        source_words_in_overlap=len(s_positions),
        target_words_in_overlap=len(t_positions),
        source_overlap_pct=_pct(len(s_positions), s_count),
        target_overlap_pct=_pct(len(t_positions), t_count),
        passages=passages,
    )


# ---------------------------------------------------------------------------
# Phase 2: side-by-side alignment
# ---------------------------------------------------------------------------

def _ngram_set(
    words: tuple[str, ...], n: int = NGRAM_SIZE,
) -> frozenset[tuple[str, ...]]:
    """Return the set of n-grams for fast Jaccard computation."""
    return frozenset(
        tuple(words[i : i + n]) for i in range(len(words) - n + 1)
    )


def _jaccard(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> float:
    """Jaccard similarity between two n-gram sets."""
    if not a and not b:
        return 0.0
    return _ratio(len(a & b), len(a | b))


def _split_paragraphs(text: str) -> tuple[str, ...]:
    """Split text into non-empty paragraphs."""
    raw = re.split(r"\n\s*\n|\n", text)
    return tuple(p.strip() for p in raw if p.strip())


def align_passages(
    source_text: str,
    target_text: str,
) -> tuple[AlignmentSummary, tuple[AlignedPassage, ...]]:
    """For each source paragraph, find its best-matching target paragraph.

    Returns all matched pairs ranked by similarity, excluding below 15%.
    """
    source_paras = _split_paragraphs(source_text)
    target_paras = _split_paragraphs(target_text)

    empty_summary = AlignmentSummary(
        source_paragraphs=0,
        matched_paragraphs=0,
        matched_pct=0.0,
        avg_similarity=0.0,
        count_identical=0,
        count_high=0,
        count_moderate=0,
    )

    if not source_paras or not target_paras:
        return empty_summary, ()

    target_normalized = tuple(
        normalize_with_spans(p) for p in target_paras
    )
    target_ngram_sets = tuple(
        _ngram_set(tn.words) for tn in target_normalized
    )

    candidates: list[AlignedPassage] = []
    eligible_count = 0

    for s_para in source_paras:
        s_norm = normalize_with_spans(s_para)
        if len(s_norm.words) < NGRAM_SIZE:
            continue
        eligible_count += 1
        s_ngrams = _ngram_set(s_norm.words)

        best_score = 0.0
        best_idx = 0
        for t_idx, t_ngrams in enumerate(target_ngram_sets):
            if len(target_normalized[t_idx].words) < NGRAM_SIZE:
                continue
            score = _jaccard(s_ngrams, t_ngrams)
            if score > best_score:
                best_score = score
                best_idx = t_idx

        if best_score < 0.15:
            continue

        s_preview = re.sub(r"\s+", " ", s_para).strip()
        t_preview = re.sub(r"\s+", " ", target_paras[best_idx]).strip()

        if len(s_preview) > PREVIEW_MAX_CHARS:
            s_preview = s_preview[:PREVIEW_MAX_CHARS] + "..."
        if len(t_preview) > PREVIEW_MAX_CHARS:
            t_preview = t_preview[:PREVIEW_MAX_CHARS] + "..."

        candidates.append(AlignedPassage(
            source_preview=s_preview,
            target_preview=t_preview,
            source_words=len(s_norm.words),
            target_words=len(target_normalized[best_idx].words),
            similarity_pct=float(round(Decimal(str(best_score)) * 100, 1)),
        ))

    matched_count = len(candidates)
    similarity_sum = sum(
        Decimal(str(c.similarity_pct)) for c in candidates
    )
    summary = AlignmentSummary(
        source_paragraphs=eligible_count,
        matched_paragraphs=matched_count,
        matched_pct=_pct(matched_count, eligible_count),
        avg_similarity=float(
            round(similarity_sum / Decimal(matched_count), 1)
            if matched_count
            else Decimal("0"),
        ),
        count_identical=sum(
            1 for c in candidates if c.similarity_pct >= 99.9
        ),
        count_high=sum(
            1 for c in candidates if 50.0 <= c.similarity_pct < 99.9
        ),
        count_moderate=sum(
            1 for c in candidates if 15.0 <= c.similarity_pct < 50.0
        ),
    )

    ranked = sorted(candidates, key=lambda a: -a.similarity_pct)
    return summary, tuple(ranked)


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------

def build_report(pair: DocumentPair) -> FullReport:
    """Build a complete two-phase report for a document pair."""
    source_text = pair.source_path.read_text()
    target_text = pair.target_path.read_text()

    ngram_result = compare_texts(source_text, target_text)

    source_body = _strip_bibliography(source_text)
    target_body = _strip_bibliography(target_text)
    body_only_result = compare_texts(source_body, target_body)

    summary, aligned = align_passages(source_body, target_body)

    return FullReport(
        label=pair.label,
        source_name=pair.source_path.name,
        target_name=pair.target_path.name,
        ngram_result=ngram_result,
        body_only_result=body_only_result,
        alignment_summary=summary,
        aligned_passages=aligned,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _log_ngram_phase(label: str, result: ComparisonResult) -> None:
    """Log Phase 1: n-gram scan results."""
    logger.info(
        "  Documento fuente (consultora): %s palabras",
        f"{result.source_words:,}",
    )
    logger.info(
        "  Documento destino (publicación): %s palabras",
        f"{result.target_words:,}",
    )
    logger.info(
        "  %d-gramas coincidentes: %s / %s del documento fuente",
        NGRAM_SIZE,
        f"{result.matching_ngrams:,}",
        f"{result.total_ngrams_source:,}",
    )
    logger.info(
        "  >> Texto de la consultora en la publicación: %s%% de las palabras"
        " de la consultora aparecen en la publicación",
        result.source_overlap_pct,
    )
    logger.info(
        "  >> Texto de la publicación copiado de la consultora: %s%% de las"
        " palabras de la publicación provienen del trabajo de la consultora",
        result.target_overlap_pct,
    )

    if result.passages:
        logger.info(
            "  Pasajes coincidentes encontrados: %d",
            len(result.passages),
        )


def _log_body_only(full: ComparisonResult, body: ComparisonResult) -> None:
    """Log body-only metrics (bibliography excluded)."""
    logger.info("")
    logger.info("  SOLO CUERPO (sin bibliografía):")
    logger.info(
        "    Fuente: %s -> %s palabras | Destino: %s -> %s palabras",
        f"{full.source_words:,}",
        f"{body.source_words:,}",
        f"{full.target_words:,}",
        f"{body.target_words:,}",
    )
    logger.info(
        "    >> Texto de la consultora en la publicación: %s%%"
        " (era %s%% con bibliografía)",
        body.source_overlap_pct,
        full.source_overlap_pct,
    )
    logger.info(
        "    >> Texto de la publicación copiado de la consultora: %s%%"
        " (era %s%% con bibliografía)",
        body.target_overlap_pct,
        full.target_overlap_pct,
    )


def _log_aligned_passages(
    summary: AlignmentSummary,
    aligned: tuple[AlignedPassage, ...],
) -> None:
    """Log Phase 2: summary metrics and side-by-side aligned passages."""
    logger.info("")
    logger.info(
        "  Párrafos elegibles en documento fuente: %d",
        summary.source_paragraphs,
    )
    logger.info(
        "  Párrafos con coincidencia (Jaccard >= 15%%): %d / %d (%s%%)",
        summary.matched_paragraphs,
        summary.source_paragraphs,
        summary.matched_pct,
    )
    logger.info(
        "  Similitud Jaccard promedio (entre los coincidentes): %s%%",
        summary.avg_similarity,
    )
    logger.info(
        "    Idénticos (>=99%%): %d | Altos (50-99%%): %d"
        " | Moderados (15-50%%): %d",
        summary.count_identical,
        summary.count_high,
        summary.count_moderate,
    )

    if not aligned:
        return


def _dedup_aligned(
    aligned: tuple[AlignedPassage, ...],
    top_n: int,
) -> tuple[AlignedPassage, ...]:
    """Deduplicate aligned passages by target preview, keep top_n."""
    seen: set[str] = set()
    result: list[AlignedPassage] = []
    for ap in aligned:
        key = ap.target_preview[:100]
        if key not in seen:
            seen.add(key)
            result.append(ap)
        if len(result) >= top_n:
            break
    return tuple(result)


OUTPUT_DIR = Path("output")


def _slugify(label: str) -> str:
    """Turn a comparison label into a safe filename slug."""
    slug = label.lower().replace(" ", "_")
    return re.sub(r"[^\w_]", "", slug)


def _write_full_report(report: FullReport) -> Path:
    """Write all passages and alignments to a .txt file in output/."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = f"{_slugify(report.label)}.txt"
    path = OUTPUT_DIR / filename

    full = report.ngram_result
    body = report.body_only_result
    summary = report.alignment_summary

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"COMPARACIÓN: {report.label}")
    lines.append(f"  Fuente (consultora): {report.source_name}")
    lines.append(f"  Destino (publicación): {report.target_name}")
    lines.append("=" * 80)

    # -- Phase 1: n-gram scan --
    lines.append("")
    lines.append("FASE 1: Escaneo de 8-gramas")
    lines.append("-" * 80)
    lines.append(f"  Documento fuente (consultora): {full.source_words:,}"
                 " palabras")
    lines.append(f"  Documento destino (publicación): {full.target_words:,}"
                 " palabras")
    lines.append(f"  8-gramas coincidentes: {full.matching_ngrams:,}"
                 f" / {full.total_ngrams_source:,} del documento fuente")
    lines.append("")
    lines.append(
        f"  Texto de la consultora en la publicación:"
        f" {full.source_overlap_pct}%"
    )
    lines.append(
        f"  Texto de la publicación copiado de la consultora:"
        f" {full.target_overlap_pct}%"
    )

    lines.append("")
    lines.append("  SOLO CUERPO (sin bibliografía):")
    lines.append(
        f"    Fuente: {full.source_words:,} -> {body.source_words:,}"
        f" palabras | Destino: {full.target_words:,} ->"
        f" {body.target_words:,} palabras"
    )
    lines.append(
        f"    Texto de la consultora en la publicación:"
        f" {body.source_overlap_pct}%"
        f" (era {full.source_overlap_pct}% con bibliografía)"
    )
    lines.append(
        f"    Texto de la publicación copiado de la consultora:"
        f" {body.target_overlap_pct}%"
        f" (era {full.target_overlap_pct}% con bibliografía)"
    )

    lines.append("")
    lines.append(
        f"  Total de pasajes coincidentes (solo cuerpo):"
        f" {len(body.passages)}"
    )
    lines.append("")
    for i, p in enumerate(body.passages):
        lines.append(
            f'  [{i + 1}] {p.word_count} palabras'
            f' (palabra {p.start_word}): "{p.text_preview}"'
        )
    lines.append("")

    # -- Phase 2: paragraph alignment --
    lines.append("")
    lines.append("FASE 2: Alineación lado a lado por párrafos"
                 " (sin bibliografía)")
    lines.append("-" * 80)
    lines.append(
        f"  Párrafos elegibles en documento fuente:"
        f" {summary.source_paragraphs}"
    )
    lines.append(
        f"  Párrafos con coincidencia (Jaccard >= 15%):"
        f" {summary.matched_paragraphs} / {summary.source_paragraphs}"
        f" ({summary.matched_pct}%)"
    )
    lines.append(
        f"  Similitud Jaccard promedio (entre los coincidentes):"
        f" {summary.avg_similarity}%"
    )
    lines.append(
        f"    Idénticos (>=99%): {summary.count_identical}"
        f" | Altos (50-99%): {summary.count_high}"
        f" | Moderados (15-50%): {summary.count_moderate}"
    )
    lines.append("")
    for i, ap in enumerate(report.aligned_passages):
        lines.append(
            f"  [{i + 1}] Similitud Jaccard: {ap.similarity_pct}%"
            f" | Fuente: {ap.source_words} palabras"
            f", Destino: {ap.target_words} palabras"
        )
        lines.append(f'    FUENTE:  "{ap.source_preview}"')
        lines.append(f'    DESTINO: "{ap.target_preview}"')
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_METHODOLOGY = """\

METODOLOGÍA
--------------------------------------------------------------------------------

  FASE 1: Escaneo de 8-gramas

    Ambos documentos se normalizan: se convierten a minúsculas, se eliminan
    tildes y signos de puntuación. Luego se extraen ventanas deslizantes de 8
    palabras consecutivas ('8-gramas'). Cuando la misma secuencia de 8 palabras
    aparece en ambos documentos, se cuenta como coincidencia textual. Una
    secuencia de 8 palabras idénticas es suficientemente larga como para
    descartar la coincidencia.

    Esta técnica (document fingerprinting mediante n-gramas) pertenece a la
    familia de métodos de string-matching usados en detección académica de
    plagio, junto con algoritmos como Rabin-Karp y Knuth-Morris-Pratt empleados
    por herramientas como Turnitin y CopyCatch (ver: Hamed et al., 'Plagiarism
    types and detection methods', Frontiers in Computer Science, 2025,
    https://doi.org/10.3389/fcomp.2025.1504725).

    'Texto de la consultora en la publicación' = porcentaje de las palabras
    del trabajo de la consultora que aparecen textualmente en la publicación
    oficial. Responde a: ¿cuánto del trabajo original fue copiado?

    'Texto de la publicación copiado de la consultora' = porcentaje de las
    palabras de la publicación oficial que provienen del trabajo de la
    consultora. Responde a: ¿cuánto de la publicación es texto copiado?

    'Solo cuerpo' repite el mismo análisis después de eliminar las secciones
    de bibliografía de ambos documentos, de modo que los porcentajes reflejan
    reuso de texto sustantivo, no citas académicas compartidas.

  FASE 2: Alineación lado a lado por párrafos

    Cada párrafo del documento fuente se compara con todos los párrafos del
    documento destino usando el índice de Jaccard: la cantidad de 8-gramas
    compartidos dividida por la cantidad total de 8-gramas distintos entre
    ambos párrafos (ver: https://es.wikipedia.org/wiki/%C3%8Dndice_de_Jaccard).
    Se muestra el párrafo destino más parecido junto al párrafo fuente.

    Se eligió el índice de Jaccard por tres razones:
      1. Está normalizado entre 0 y 1, lo que permite comparar párrafos de
         distinta longitud sin que el tamaño distorsione el resultado.
      2. Es simétrico: mide la coincidencia entre ambos textos por igual,
         sin privilegiar la dirección fuente-destino ni destino-fuente.
      3. Penaliza el relleno: si se copia un párrafo pero se agrega texto
         adicional alrededor, la unión crece pero la intersección no, y el
         puntaje baja. Esto distingue correctamente una copia textual de una
         coincidencia parcial.

    100% = texto idéntico. Sobre 50% = copiado sustancial con ediciones menores.
    Se excluyen coincidencias por debajo de 15%.

  NOTA SOBRE VERIFICACIÓN MANUAL

    Los pasajes mostrados provienen de los archivos .txt extraídos de los
    documentos originales (.docx y .pdf). Para verificar un pasaje en el PDF o
    DOCX original usando Ctrl+F, busque fragmentos cortos (5-10 palabras) del
    pasaje, no el pasaje completo. Los saltos de línea y de página en el
    documento original pueden impedir que Ctrl+F encuentre frases largas que
    cruzan esos límites.
"""

_CREDENTIALS = """\

RESPONSABLE DEL ANÁLISIS
--------------------------------------------------------------------------------

  Luis Bordo
  COO y cofundador, Dream Aim Deliver AI (Suiza)
  Candidato a doctor en Filosofía de la Ciencia, Université de Genève
  Sociólogo licenciado, Pontificia Universidad Católica del Perú (PUCP)
  https://www.linkedin.com/in/luisbordo/

  Experiencia relevante para este análisis:

    - Investigador social cuantitativo con 8 años de experiencia en diseño
      de encuestas, modelamiento estadístico y análisis de datos, incluyendo
      trabajo en el Ministerio de Educación del Perú (MINEDU) y la PUCP.
    - Formación doctoral en causalidad e inferencia causal aplicada a las
      ciencias sociales (Université de Genève, supervisor: Prof. Marcel Weber).
    - Maestría en Filosofía con especialización en Filosofía de la Ciencia
      (Université de Genève, tesis calificada 6/6, Premio Humbert 2021).
    - Competencias técnicas: Python, Rust, R, SQL, modelamiento
      estadístico, ingeniería de datos.
"""


def _log_header() -> None:
    """Log methodology and credentials at the top of the report."""
    logger.info("%s", _METHODOLOGY)
    logger.info("%s", _CREDENTIALS)


def run_comparison(pairs: tuple[DocumentPair, ...]) -> None:
    """Run full two-phase comparison for each document pair."""
    logger.info(
        "%s",
        "=" * 80,
    )
    logger.info(
        "ANÁLISIS DE PLAGIO: Productos de Consultoría vs Publicaciones"
        " Oficiales",
    )
    logger.info(
        "%s",
        "=" * 80,
    )

    _log_header()

    for pair in pairs:
        logger.info("")
        logger.info("%s", "=" * 80)
        logger.info("COMPARACIÓN: %s", pair.label)
        logger.info(
            "  Fuente (consultora): %s",
            pair.source_path.name,
        )
        logger.info(
            "  Destino (publicación): %s",
            pair.target_path.name,
        )
        logger.info("%s", "=" * 80)

        report = build_report(pair)

        logger.info("")
        logger.info("  FASE 1: Escaneo de 8-gramas")
        _log_ngram_phase(report.label, report.ngram_result)

        _log_body_only(report.ngram_result, report.body_only_result)

        logger.info("")
        logger.info("  FASE 2: Alineación por párrafos")
        _log_aligned_passages(
            report.alignment_summary, report.aligned_passages,
        )

        out_path = _write_full_report(report)
        logger.info("")
        logger.info("  Reporte completo: %s", out_path)


def main() -> None:
    """Entry point: configure logging and run comparisons."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = OUTPUT_DIR / "resumen.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    file_handler = logging.FileHandler(summary_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    base = Path(__file__).resolve().parent / "minedu_files"

    pairs = (
        DocumentPair(
            source_path=base / "Producto_3_v170524 modificado.txt",
            target_path=base / "Estudio para la identificación de desigualdades"
            " y barreras de acceso y permanencia en la educación"
            " superior universit.txt",
            label="Producto 3 vs MINEDU Estudio Desigualdades",
        ),
        DocumentPair(
            source_path=base / "Informe_Producto_5_08012025.txt",
            target_path=base / "Propuesta para reduccion de brechas.txt",
            label="Producto 5 vs MINEDU Propuesta Reducción Brechas",
        ),
        DocumentPair(
            source_path=base / "Producto_2_v06.03.2024 (2).txt",
            target_path=base / "Sistematización de experiencias y buenas"
            " prácticas para la igualdad de género en el ámbito"
            " universitario.txt",
            label="Producto 2 vs MINEDU Sistematización",
        ),
    )

    run_comparison(pairs)


if __name__ == "__main__":
    main()
