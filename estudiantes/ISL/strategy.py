"""
strategy.py — Equipo ISL
=========================
Estrategia híbrida Entropía + Expected Value para el torneo de Wordle ITAM P26.

Diseño general:
  - Opener hardcodeado por configuración (pre-calculado offline con non-words)
  - begin_game() pre-computa vectores de letras por posición (numpy) para
    que el scoring de cada guess sea completamente vectorizado sin loops Python
  - Modo uniform  → maximiza entropía de Shannon pura
  - Modo frequency → maximiza score híbrido: α*entropía + (1-α)*prob_esperada
  - Casos base rápidos: ≤1 candidato → directo, 2 → el más probable
  - Threshold adaptivo: con pocos candidatos evalúa solo entre ellos

Pre-cómputo en begin_game (cabe en ~2-3s):
  1. Matriz de caracteres del vocab como array numpy int8  → O(n * wl)
  2. Vector de probabilidades ordenado igual que vocab     → O(n)
  3. Tabla de patrones SOLO si n ≤ TABLE_SIZE_LIMIT        → O(n²) vectorizado
  4. Para vocab grande: feedback vectorizado por columnas  → O(n) por guess
  5. Caché de segundos guesses por patrón del opener       → turnos 1 y 2 gratis
"""

from __future__ import annotations

import numpy as np

from strategy import Strategy, GameConfig
from wordle_env import filter_candidates


# ── Openers óptimos pre-calculados offline ────────────────────────────────────
BEST_OPENERS: dict[tuple[int, str], str] = {
    (4, "uniform"):   "aore",    # non-word ★
    (4, "frequency"): "aore",    # non-word ★
    (5, "uniform"):   "careo",   # palabra real
    (5, "frequency"): "careo",   # palabra real
    (6, "uniform"):   "careto",  # palabra real
    (6, "frequency"): "careto",  # palabra real
}

# Peso entropía vs probabilidad en modo frequency
ALPHA = 0.7

# Vocab más grande que esto → no construimos tabla n×n (evita timeout)
TABLE_SIZE_LIMIT = 1500

# Con ≤ este número de candidatos, evaluamos SOLO entre ellos
SMALL_CANDIDATE_THRESHOLD = 12


# ── Feedback vectorizado (sin loops Python) ───────────────────────────────────

def _feedback_vectorized(
    secret_chars: np.ndarray,  # shape (n, wl) dtype int16
    guess_chars: np.ndarray,   # shape (wl,)   dtype int16
    wl: int,
) -> np.ndarray:
    """
    Calcula encode(feedback(secret, guess)) para todos los secrets en paralelo.
    Retorna array shape (n,) con el patrón codificado como int32.
    """
    n = secret_chars.shape[0]
    pat = np.zeros((n, wl), dtype=np.int16)
    remaining = np.zeros((n, 256), dtype=np.int16)  # 256 para soportar ñ, á, é, etc.

    for pos in range(wl):
        remaining[np.arange(n), secret_chars[:, pos]] += 1

    # Paso 1: verdes
    for pos in range(wl):
        green = secret_chars[:, pos] == guess_chars[pos]
        pat[green, pos] = 2
        remaining[green, guess_chars[pos]] -= 1

    # Paso 2: amarillos
    for pos in range(wl):
        already_green = pat[:, pos] == 2
        has_letter = remaining[:, guess_chars[pos]] > 0
        yellow = (~already_green) & has_letter
        pat[yellow, pos] = 1
        remaining[yellow, guess_chars[pos]] -= 1

    powers = np.array([3 ** i for i in range(wl)], dtype=np.int32)
    return (pat.astype(np.int32) * powers).sum(axis=1)


class ISLStrategy(Strategy):

    @property
    def name(self) -> str:
        return "ISLStrategy_ISL"

    # ── begin_game ────────────────────────────────────────────────────────────

    def begin_game(self, config: GameConfig) -> None:
        self._vocab  = list(config.vocabulary)
        self._probs  = config.probabilities
        self._mode   = config.mode
        self._wl     = config.word_length
        self._opener = BEST_OPENERS.get((self._wl, self._mode), self._vocab[0])
        n            = len(self._vocab)

        # 1. Matriz de chars: shape (n, wl) — usa ord(c) directo para soportar ñ, á, etc.
        self._vocab_chars = np.array(
            [[ord(c) for c in w] for w in self._vocab],
            dtype=np.int16,
        )

        # 2. Vector de probabilidades: shape (n,)
        self._prob_vec = np.array(
            [self._probs.get(w, 0.0) for w in self._vocab],
            dtype=np.float32,
        )

        # 3. Índice palabra → posición en vocab
        self._word_to_idx = {w: i for i, w in enumerate(self._vocab)}

        # 4. Tabla n×n solo si vocab es pequeño/mediano
        if n <= TABLE_SIZE_LIMIT:
            self._pattern_table: np.ndarray | None = self._build_table(n)
            self._has_table = True
        else:
            self._pattern_table = None
            self._has_table = False

        # 5. Caché de segundos guesses: para cada patrón del opener → mejor guess
        self._second_cache: dict[tuple[int, ...], str] = {}
        self._precompute_second_guesses()

    def _build_table(self, n: int) -> np.ndarray:
        """
        Tabla[i, j] = encode(feedback(vocab[i], vocab[j]))
        Construida columna por columna con feedback vectorizado → rápida.
        """
        table = np.zeros((n, n), dtype=np.int16)
        for j in range(n):
            table[:, j] = _feedback_vectorized(
                self._vocab_chars, self._vocab_chars[j], self._wl
            )
        return table

    def _precompute_second_guesses(self) -> None:
        """
        Para cada patrón posible que puede devolver el opener, pre-calcula
        el mejor segundo guess. Turno 2 queda como lookup O(1).
        """
        # Calcular patrones del opener contra todo el vocab
        opener_chars = np.array(
            [ord(c) for c in self._opener], dtype=np.int16
        )
        if self._has_table and self._opener in self._word_to_idx:
            oi = self._word_to_idx[self._opener]
            all_patterns = self._pattern_table[:, oi]
        else:
            all_patterns = _feedback_vectorized(
                self._vocab_chars, opener_chars, self._wl
            )

        for pat_val in np.unique(all_patterns):
            idxs = np.where(all_patterns == pat_val)[0]
            candidates = [self._vocab[i] for i in idxs]
            pat_tuple = self._decode_pattern(int(pat_val))

            if len(candidates) == 1:
                self._second_cache[pat_tuple] = candidates[0]
            elif len(candidates) == 2:
                if self._mode == "frequency":
                    self._second_cache[pat_tuple] = max(
                        candidates, key=lambda w: self._probs.get(w, 0.0)
                    )
                else:
                    self._second_cache[pat_tuple] = candidates[0]
            else:
                self._second_cache[pat_tuple] = self._best_guess(candidates, candidates)

    def _decode_pattern(self, encoded: int) -> tuple[int, ...]:
        pat = []
        for _ in range(self._wl):
            pat.append(encoded % 3)
            encoded //= 3
        return tuple(pat)

    # ── guess ─────────────────────────────────────────────────────────────────

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:

        # Turno 1: opener gratis
        if not history:
            return self._opener

        # Turno 2: lookup en caché
        if len(history) == 1:
            g0, pat0 = history[0]
            if g0 == self._opener and pat0 in self._second_cache:
                return self._second_cache[pat0]

        # Turnos 3+: filtrar y calcular
        candidates = self._vocab
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)

        if not candidates:
            return self._vocab[0]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 2:
            if self._mode == "frequency":
                return max(candidates, key=lambda w: self._probs.get(w, 0.0))
            return candidates[0]

        guess_pool = candidates if len(candidates) <= SMALL_CANDIDATE_THRESHOLD else self._vocab
        return self._best_guess(candidates, guess_pool)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _best_guess(self, candidates: list[str], guess_pool: list[str]) -> str:
        candidate_idxs = np.array(
            [self._word_to_idx[w] for w in candidates if w in self._word_to_idx],
            dtype=np.int32,
        )
        if len(candidate_idxs) == 0:
            return candidates[0]

        cand_chars  = self._vocab_chars[candidate_idxs]
        candidate_set = set(candidates)
        n = len(candidate_idxs)
        best_guess = candidates[0]
        best_score = -1.0

        for gw in guess_pool:
            # Obtener patrones: tabla si está disponible, vectorizado si no
            if self._has_table and gw in self._word_to_idx:
                patterns = self._pattern_table[candidate_idxs, self._word_to_idx[gw]]
            else:
                gc = np.array([ord(c) for c in gw], dtype=np.int16)
                patterns = _feedback_vectorized(cand_chars, gc, self._wl)

            unique, counts = np.unique(patterns, return_counts=True)
            score = self._score(counts, unique, patterns, candidate_idxs, n)

            is_cand = gw in candidate_set
            if score > best_score or (
                score == best_score and is_cand and best_guess not in candidate_set
            ):
                best_score = score
                best_guess = gw

        return best_guess

    def _score(
        self,
        counts: np.ndarray,
        unique_patterns: np.ndarray,
        patterns: np.ndarray,
        candidate_idxs: np.ndarray,
        n: int,
    ) -> float:
        ps = counts / n
        entropy = float(-np.sum(ps * np.log2(ps + 1e-12)))

        if self._mode == "uniform":
            return entropy

        # Frequency: α*H + (1-α)*expected_max_prob_por_partición
        expected_prob = 0.0
        for pat_val, p_group in zip(unique_patterns, ps):
            group_idxs = candidate_idxs[patterns == pat_val]
            max_prob = float(self._prob_vec[group_idxs].max())
            expected_prob += float(p_group) * max_prob

        return ALPHA * entropy + (1.0 - ALPHA) * expected_prob * 10.0