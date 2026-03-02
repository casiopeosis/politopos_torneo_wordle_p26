"""
find_best_openers.py — Equipo ISL
==================================
Calcula el mejor primer guess para cada una de las 6 configuraciones
del torneo usando el vocabulario real del corpus completo.

INCLUYE NON-WORDS: además del vocabulario real, evalúa combinaciones
construidas para maximizar cobertura de letras frecuentes del español.
Esto puede superar a cualquier palabra real como opener.

Cómo correrlo (desde la raíz del repo):
    python3 estudiantes/ISL/find_best_openers.py

Qué produce:
    Imprime los top-5 mejores primeros guesses para cada ronda,
    listos para hardcodear en strategy.py. Marca si el mejor es
    una palabra real o un non-word.

Criterio de evaluación:
    - Modo uniform  → maximiza entropía de Shannon pura
    - Modo frequency → maximiza Expected Value híbrido:
                       score = α * entropy + (1-α) * prob_acumulada_partition
      donde α=0.7 balancea información vs ir a las palabras probables primero.

Tiempo estimado: ~3-8 min dependiendo del tamaño del corpus.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict
from pathlib import Path

# ── Importaciones del framework ──────────────────────────────────────────────
# Asume que corres desde la raíz del repo
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from wordle_env import feedback
from lexicon import load_lexicon  # carga vocabulario + probabilidades reales

# ── Parámetros ────────────────────────────────────────────────────────────────
ALPHA = 0.7   # peso de entropía vs probabilidad en modo frequency
TOP_N = 5     # cuántos candidatos mostrar por configuración

# Letras más frecuentes del español — usadas para generar non-words candidatos
# Orden aproximado por frecuencia en corpus español escrito
LETRAS_FRECUENTES_ES = "aeorisntlcdup"

# Cuántos non-words generar por longitud (más = mejor búsqueda, más lento)
# Con 500 ya se cubren combinaciones muy diversas
MAX_NON_WORDS = 500

# ── Helpers ───────────────────────────────────────────────────────────────────

def generate_non_words(word_length: int, vocab: list[str]) -> list[str]:
    """
    Genera combinaciones de letras (non-words) para evaluar como opener.

    Estrategia:
    1. Toma las letras más frecuentes del corpus real (no asume nada)
    2. Genera permutaciones de subconjuntos de tamaño word_length
       priorizando letras distintas (sin repetir en el mismo non-word,
       porque repetir una letra da menos información)
    3. Limita a MAX_NON_WORDS para no tardar demasiado

    Ejemplo para length=5: "aeris", "oaern", "senla", ...
    """
    # Calcular frecuencia real de letras en el vocabulario actual
    freq: dict[str, int] = defaultdict(int)
    for word in vocab:
        for ch in word:
            freq[ch] += 1

    # Ordenar letras por frecuencia descendente
    letras_ordenadas = [ch for ch, _ in sorted(freq.items(), key=lambda x: -x[1])]

    # Tomar las top letras según el tamaño de palabra
    # Para length=4 → top 8 letras, para 5 → top 10, para 6 → top 12
    pool_size = word_length * 2
    top_letras = letras_ordenadas[:pool_size]

    non_words = []
    seen = set(vocab)  # no agregar palabras reales (ya están en el vocab)

    # Generar permutaciones de word_length letras distintas
    for perm in itertools.permutations(top_letras, word_length):
        candidate = "".join(perm)
        if candidate not in seen:
            non_words.append(candidate)
            seen.add(candidate)
        if len(non_words) >= MAX_NON_WORDS:
            break

    return non_words


def encode_pattern(pat: tuple[int, ...]) -> int:
    """Codifica un patrón de feedback como entero para hashing rápido."""
    val = 0
    for i, c in enumerate(pat):
        val += c * (3 ** i)
    return val


def compute_entropy(guess: str, candidates: list[str]) -> float:
    """Entropía de Shannon pura: H = -sum(p * log2(p))"""
    partition: dict[int, int] = defaultdict(int)
    for c in candidates:
        key = encode_pattern(feedback(c, guess))
        partition[key] += 1

    n = len(candidates)
    entropy = 0.0
    for count in partition.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def compute_hybrid_score(
    guess: str,
    candidates: list[str],
    probs: dict[str, float],
    alpha: float = ALPHA,
) -> float:
    """
    Score híbrido para modo frequency:
        score = alpha * entropy + (1 - alpha) * expected_prob_of_partition

    La segunda parte premia guesses que separan a las palabras más probables
    en particiones pequeñas (las resolvemos más rápido).
    """
    partition: dict[int, list[str]] = defaultdict(list)
    for c in candidates:
        key = encode_pattern(feedback(c, guess))
        partition[key].append(c)

    n = len(candidates)
    entropy = 0.0
    expected_prob = 0.0

    for group in partition.values():
        p_group = len(group) / n
        # Componente de entropía
        entropy -= p_group * math.log2(p_group)
        # Componente de probabilidad: max prob dentro del grupo
        # (qué tan "fácil" sería resolver este grupo en el siguiente intento)
        max_prob_in_group = max(probs.get(w, 0.0) for w in group)
        expected_prob += p_group * max_prob_in_group

    return alpha * entropy + (1 - alpha) * expected_prob * 10  # escala aprox


def find_best_opener(
    vocab: list[str],
    probs: dict[str, float],
    mode: str,
    word_length: int,
    top_n: int = TOP_N,
) -> list[tuple[str, float, bool]]:
    """
    Evalúa el vocabulario completo + non-words generados como posible primer guess.
    Devuelve los top_n mejores con su score y si son palabra real o no.
    """
    # Construir pool: palabras reales + non-words
    non_words = generate_non_words(word_length, vocab)
    full_pool = vocab + non_words
    vocab_set = set(vocab)

    print(f"  Pool de evaluación: {len(vocab):,} palabras reales + {len(non_words):,} non-words = {len(full_pool):,} total")
    results = []

    for i, guess in enumerate(full_pool):
        if i % 500 == 0 and i > 0:
            pct = i / len(full_pool) * 100
            print(f"    {i:,}/{len(full_pool):,} evaluados ({pct:.0f}%)...", end="\r")

        if mode == "uniform":
            score = compute_entropy(guess, vocab)
        else:
            score = compute_hybrid_score(guess, vocab, probs)

        is_real_word = guess in vocab_set
        results.append((guess, score, is_real_word))

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"    {len(full_pool):,}/{len(full_pool):,} evaluados (100%)... ✓     ")
    return results[:top_n]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CÁLCULO DE MEJORES OPENERS — Equipo ISL")
    print("=" * 60)

    configs = [
        (4, "uniform"),
        (4, "frequency"),
        (5, "uniform"),
        (5, "frequency"),
        (6, "uniform"),
        (6, "frequency"),
    ]

    results_summary = {}

    for word_length, mode in configs:
        ronda = f"{word_length} letras / {mode}"
        print(f"\n── Ronda: {ronda} ──")

        t0 = time.time()
        try:
            lex = load_lexicon(word_length=word_length, mode=mode)
            vocab = list(lex.words)
            probs = dict(lex.probs)
        except Exception as e:
            print(f"  ✗ No se pudo cargar el lexicón: {e}")
            print("  Asegúrate de correr 'python3 download_words.py --all-lengths' primero")
            continue

        print(f"  Vocabulario: {len(vocab):,} palabras")

        top = find_best_opener(vocab, probs, mode, word_length)

        elapsed = time.time() - t0
        print(f"  Tiempo: {elapsed:.1f}s")
        print(f"\n  Top {TOP_N} openers:")
        for rank, (word, score, is_real) in enumerate(top, 1):
            tipo = "palabra real" if is_real else "NON-WORD ★"
            marker = " ◄ MEJOR" if rank == 1 else ""
            print(f"    {rank}. '{word}'  score={score:.4f}  [{tipo}]{marker}")

        results_summary[(word_length, mode)] = (top[0][0], top[0][2])  # (palabra, is_real)

    # ── Resumen final: código listo para copiar ───────────────────────────────
    print("\n" + "=" * 60)
    print("  HARDCODE LISTO PARA strategy.py")
    print("  Copia el dict BEST_OPENERS en tu estrategia:")
    print("=" * 60)
    print("\nBEST_OPENERS = {")
    for (wl, mode), (word, is_real) in results_summary.items():
        tipo = "# palabra real" if is_real else "# non-word ★"
        print(f'    ({wl}, "{mode}"): "{word}",  {tipo}')
    print("}")
    print()


if __name__ == "__main__":
    main()
