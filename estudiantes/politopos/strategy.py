"""
Estrategia: Entropía Ponderada por Probabilidad (Weighted Entropy).

IDEA CENTRAL:
    El Entropy benchmark estándar maximiza H = -Σ (n_k/N) log2(n_k/N),
    tratando todas las palabras candidatas como igualmente posibles.

    Esta estrategia maximiza H_w = -Σ P_k log2(P_k), donde P_k es la
    MASA DE PROBABILIDAD del grupo k (suma de probs de las palabras en ese grupo).

    En modo 'uniform' ambas son idénticas.
    En modo 'frequency' esta versión es ESTRICTAMENTE MEJOR porque considera
    que algunas palabras secretas son mucho más probables que otras — da más
    valor a particiones que separan palabras de alta probabilidad.

ARQUITECTURA:
    - begin_game(): O(n·wl) — construye representación numpy del vocabulario.
    - Turno 1: guess inicial fijo verificado contra el vocabulario real.
    - Turno 2: si quedan muchos candidatos, segundo guess fijo complementario.
    - Turnos 3+: entropía ponderada sobre candidatos, con pool inteligente.
    - Caso 2 candidatos: elegir el más probable (MaxProb), no el primero.

ÚNICO CAMBIO vs versión ganadora original:
    Fix ñ en _feedback_batch: la versión original usaba offset ord("a")=97,
    lo que mapeaba ord("ñ")=241 al índice 144 — fuera del rango [0,25] de la
    matriz 'available'. El código lo filtraba con `(char_idx >= 0) & (char_idx < 26)`
    descartando silenciosamente las letras ñ. Resultado: palabras con ñ como
    "niño", "daño", "baño" (52 palabras en 4L) generaban patrones amarillos
    incorrectos, dejando candidatos que deberían eliminarse y viceversa.

    Fix: calcular el offset desde el carácter mínimo real del vocabulario
    (self._char_min) en lugar de ord("a"), y usar self._n_chars como tamaño
    de la dimensión del array 'available'. Todos los caracteres del vocab
    quedan en rango válido, incluyendo ñ.

    Todo lo demás — pools, caps, guesses hardcodeados, RNG, lógica de turno —
    es idéntico a la versión que ganó con 86 puntos vs Entropy 76 puntos.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

from strategy import Strategy, GameConfig
from wordle_env import feedback as _fb, filter_candidates


# ── Límites de rendimiento ────────────────────────────────────────────────────
_POOL_MAX_CANDIDATES = 150  # máx candidatos a incluir en pool cuando hay muchos
_POOL_MAX_EXTRA = 50        # máx palabras NO candidatas a explorar como guesses
_EVAL_MAX_CANDIDATES = 400  # máx candidatos contra los que calcular feedback

# Guesses iniciales verificados contra el vocab en begin_game()
_FIRST_GUESS_HINTS = {
    4: ["cora", "roia"],
    5: ["careo"],
    6: ["cerito", "careto"],
}
_SECOND_GUESS_HINTS = {
    4: ["lite", "cent"],
    5: ["sutil"],
    6: ["salman", "aislan"],
}


def _encode_pattern(pat: tuple) -> int:
    v = 0
    for i, p in enumerate(pat):
        v += p * (3 ** i)
    return v


class OptimalEG_politopos(Strategy):

    @property
    def name(self) -> str:
        return "OptimalEG_politopos"

    def begin_game(self, config: GameConfig) -> None:
        self._vocab = list(config.vocabulary)
        self._vocab_set = set(self._vocab)
        self._probs = config.probabilities
        self._mode = config.mode
        self._wl = config.word_length
        self._rng = random.Random(config.word_length * len(config.vocabulary))
        self._win_code = _encode_pattern((2,) * config.word_length)

        self._word_to_idx = {w: i for i, w in enumerate(self._vocab)}

        self._word_mat = np.array(
            [[ord(c) for c in w] for w in self._vocab], dtype=np.int16
        )
        self._powers3 = np.array([3 ** i for i in range(self._wl)], dtype=np.int32)

        # FIX ñ: offset desde el char mínimo real del vocab en lugar de ord("a").
        # ord("ñ")=241 con offset ord("a")=97 → índice 144, fuera de [0,25].
        # Con char_min real todos los caracteres quedan en rango válido.
        all_chars = sorted(set(c for w in self._vocab for c in w))
        self._char_min = ord(all_chars[0])
        self._n_chars = ord(all_chars[-1]) - self._char_min + 1

        self._first_guess = self._pick_verified_guess(_FIRST_GUESS_HINTS)
        self._second_guess = self._pick_verified_guess(_SECOND_GUESS_HINTS)

        self._fb_cache: dict[tuple[int, int], int] = {}

    def _pick_verified_guess(self, hints: dict[int, list[str]]) -> str | None:
        for g in hints.get(self._wl, []):
            if g in self._vocab_set:
                return g
        return None

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        candidates = self._vocab
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)

        if not candidates:
            return self._vocab[0]
        if len(candidates) == 1:
            return candidates[0]

        n_turn = len(history)

        if n_turn == 0 and self._first_guess:
            return self._first_guess

        if n_turn == 1 and len(candidates) > 80 and self._second_guess:
            gray_letters = {
                g_word[i]
                for g_word, pat in history
                for i, p in enumerate(pat)
                if p == 0
            }
            guess2 = self._second_guess
            overlap = sum(1 for c in set(guess2) if c in gray_letters)
            if overlap <= 1:
                return guess2

        if len(candidates) == 2:
            return max(candidates, key=lambda w: self._probs.get(w, 0.0))

        return self._best_guess_weighted_entropy(candidates)

    def _best_guess_weighted_entropy(self, candidates: list[str]) -> str:
        candidate_set = set(candidates)
        n_cands = len(candidates)

        z = sum(self._probs.get(w, 1e-9) for w in candidates)
        local_probs = {w: self._probs.get(w, 1e-9) / z for w in candidates}

        if n_cands <= _POOL_MAX_CANDIDATES:
            pool_from_candidates = candidates[:]
        else:
            by_prob = sorted(candidates, key=lambda w: -local_probs[w])
            top = by_prob[:_POOL_MAX_CANDIDATES // 2]
            rest = self._rng.sample(
                by_prob[_POOL_MAX_CANDIDATES // 2:],
                min(
                    _POOL_MAX_CANDIDATES // 2,
                    len(by_prob) - _POOL_MAX_CANDIDATES // 2,
                ),
            )
            pool_from_candidates = top + rest

        non_candidates = [w for w in self._vocab if w not in candidate_set]
        extra_n = min(_POOL_MAX_EXTRA, len(non_candidates))
        pool_extra = self._rng.sample(non_candidates, extra_n) if extra_n > 0 else []

        guess_pool = pool_from_candidates + pool_extra

        if n_cands <= _EVAL_MAX_CANDIDATES:
            eval_cands = candidates
            eval_probs = np.array([local_probs[w] for w in eval_cands])
        else:
            eval_cands = self._rng.sample(candidates, _EVAL_MAX_CANDIDATES)
            raw = np.array([local_probs[w] for w in eval_cands])
            eval_probs = raw / raw.sum()

        best_guess = candidates[0]
        best_entropy = -1.0
        best_is_candidate = candidates[0] in candidate_set

        for g in guess_pool:
            pat_codes = self._feedback_batch(eval_cands, g)
            H = self._weighted_entropy(pat_codes, eval_probs)

            is_cand = g in candidate_set
            if H > best_entropy or (
                abs(H - best_entropy) < 1e-9 and is_cand and not best_is_candidate
            ):
                best_entropy = H
                best_guess = g
                best_is_candidate = is_cand

        return best_guess

    def _weighted_entropy(self, pat_codes: np.ndarray, prob_arr: np.ndarray) -> float:
        partition: dict[int, float] = defaultdict(float)
        for code, p in zip(pat_codes.tolist(), prob_arr.tolist()):
            partition[code] += p
        H = 0.0
        for p_k in partition.values():
            if p_k > 1e-15:
                H -= p_k * math.log2(p_k)
        return H

    def _feedback_batch(self, candidates: list[str], guess: str) -> np.ndarray:
        """
        Calcula feedback(c, guess) para todos los candidatos en batch numpy.

        FIX ñ: usa self._char_min como offset (calculado desde el char mínimo
        real del vocab) en lugar de ord("a"). Esto garantiza que ñ y cualquier
        otro caracter no-ascii del español queden en un índice válido dentro
        de la matriz 'available', en lugar de ser descartados silenciosamente.
        """
        if guess not in self._word_to_idx:
            return np.array(
                [_encode_pattern(_fb(c, guess)) for c in candidates], dtype=np.int32
            )

        n = len(candidates)
        wl = self._wl
        char_min = self._char_min
        n_chars = self._n_chars

        cand_indices = np.array(
            [self._word_to_idx[c] for c in candidates], dtype=np.int32
        )
        secrets_mat = self._word_mat[cand_indices].astype(np.int32)
        g_idx = self._word_to_idx[guess]
        guess_vec = self._word_mat[g_idx].astype(np.int32)

        pattern = np.zeros((n, wl), dtype=np.int8)

        greens = secrets_mat == guess_vec[np.newaxis, :]
        pattern[greens] = 2

        # available[i, c] = veces que el char c está en secret_i sin ser verde
        available = np.zeros((n, n_chars), dtype=np.int8)
        for pos in range(wl):
            char_idx = secrets_mat[:, pos] - char_min
            not_green = ~greens[:, pos]
            valid = not_green & (char_idx >= 0) & (char_idx < n_chars)
            rows = np.where(valid)[0]
            if len(rows):
                np.add.at(available, (rows, char_idx[rows]), 1)

        for pos in range(wl):
            g_char_idx = int(guess_vec[pos]) - char_min
            if g_char_idx < 0 or g_char_idx >= n_chars:
                continue
            not_green = ~greens[:, pos]
            can_be_yellow = not_green & (available[:, g_char_idx] > 0)
            rows = np.where(can_be_yellow)[0]
            if len(rows):
                pattern[rows, pos] = 1
                available[rows, g_char_idx] -= 1

        codes = (pattern.astype(np.int32) * self._powers3[np.newaxis, :]).sum(axis=1)
        return codes