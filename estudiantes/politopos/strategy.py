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
    - begin_game(): O(n) — solo preparación. Nada costoso.
    - Turno 1: guess inicial fijo verificado contra el vocabulario real.
    - Turno 2: si quedan muchos candidatos, segundo guess fijo complementario.
    - Turnos 3+: entropía ponderada sobre candidatos, con pool inteligente.
    - Caso 2 candidatos: elegir el más probable (MaxProb), no el primero.

VENTAJAS vs versión anterior:
    - Sin heurística arbitraria (H/log2(3)+1 no tiene base matemática sólida)
    - Sin lookahead costoso que raramente cabe en el timeout
    - Feedback vectorizado correctamente con numpy
    - Pool de guesses balanceado: todos los candidatos + muestra de exploratorios
    - Primer guess verificado contra el corpus real (no hardcodeado ciegamente)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

from strategy import Strategy, GameConfig
from wordle_env import feedback as _fb, filter_candidates


# ── Límites de rendimiento ────────────────────────────────────────────────────
# Turno 1 tiene ~4500 candidatos — no podemos evaluar todo el vocab como pool.
# A partir del turno 2 los candidatos caen rápido (<300 típicamente).
_POOL_MAX_CANDIDATES = 150  # máx candidatos a incluir en pool cuando hay muchos
_POOL_MAX_EXTRA = 50  # máx palabras NO candidatas a explorar como guesses
_EVAL_MAX_CANDIDATES = 400  # máx candidatos contra los que calcular feedback

# Guesses iniciales precomputados offline: maximizan cobertura de letras frecuentes
# en español. Se verifican contra el vocab en begin_game() — si no están, se calculan.
_FIRST_GUESS_HINTS = {
    4: ["cora", "roia"],  # frequency primero, luego uniform
    5: ["careo"],  # uniforme y frequency
    6: ["cerito", "careto"],  # frequency primero, luego uniform
}
_SECOND_GUESS_HINTS = {
    4: ["lite", "cent"],
    5: ["sutil"],
    6: ["salman", "aislan"],
}


def _encode_pattern(pat: tuple) -> int:
    """Codifica un patrón de feedback como entero base-3."""
    v = 0
    for i, p in enumerate(pat):
        v += p * (3**i)
    return v


class OptimalEG_politopos(Strategy):

    @property
    def name(self) -> str:
        return "OptimalEG_politopos"

    # ── Inicialización ────────────────────────────────────────────────────────

    def begin_game(self, config: GameConfig) -> None:
        """O(n·wl) — construye representación numpy del vocabulario."""
        self._vocab = list(config.vocabulary)
        self._vocab_set = set(self._vocab)
        self._probs = config.probabilities
        self._mode = config.mode
        self._wl = config.word_length
        self._rng = random.Random(config.word_length * len(config.vocabulary))
        self._win_code = _encode_pattern((2,) * config.word_length)

        # Índice palabra → posición para acceso O(1)
        self._word_to_idx = {w: i for i, w in enumerate(self._vocab)}

        # Matriz numérica (n_words × wl) para vectorización
        self._word_mat = np.array(
            [[ord(c) for c in w] for w in self._vocab], dtype=np.int16
        )
        self._powers3 = np.array([3**i for i in range(self._wl)], dtype=np.int32)

        # Seleccionar primer y segundo guess verificados contra este vocabulario
        self._first_guess = self._pick_verified_guess(_FIRST_GUESS_HINTS)
        self._second_guess = self._pick_verified_guess(_SECOND_GUESS_HINTS)

        # Caché por turno — se limpia en begin_game (crítico para correctitud)
        self._fb_cache: dict[tuple[int, int], int] = {}

    def _pick_verified_guess(self, hints: dict[int, list[str]]) -> str | None:
        """Devuelve el primer guess de la lista que existe en el vocabulario."""
        for g in hints.get(self._wl, []):
            if g in self._vocab_set:
                return g
        return None

    # ── Interfaz principal ────────────────────────────────────────────────────

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        # Filtrar candidatos con el historial acumulado
        candidates = self._vocab
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)

        if not candidates:
            return self._vocab[0]
        if len(candidates) == 1:
            return candidates[0]

        n_turn = len(history)

        # Turno 1: usar guess inicial fijo si disponible
        if n_turn == 0 and self._first_guess:
            return self._first_guess

        # Turno 2: si quedan muchos candidatos, usar segundo guess complementario
        # Solo si no comparte letras grises del turno 1
        if n_turn == 1 and len(candidates) > 80 and self._second_guess:
            gray_letters = {
                g_word[i]
                for g_word, pat in history
                for i, p in enumerate(pat)
                if p == 0
            }
            guess2 = self._second_guess
            # Usar segundo guess si aporta información nueva (pocas letras grises)
            overlap = sum(1 for c in set(guess2) if c in gray_letters)
            if overlap <= 1:
                return guess2

        # Caso 2 candidatos: elegir el más probable (evita turno extra)
        if len(candidates) == 2:
            return max(candidates, key=lambda w: self._probs.get(w, 0.0))

        # Caso general: entropía ponderada
        return self._best_guess_weighted_entropy(candidates)

    # ── Núcleo: entropía ponderada ────────────────────────────────────────────

    def _best_guess_weighted_entropy(self, candidates: list[str]) -> str:
        """
        Elige el guess que maximiza la entropía ponderada por probabilidad.

        H_w(g) = -Σ_k P_k · log2(P_k)

        donde P_k = Σ_{w en grupo_k} prob(w) / Z  (masa de probabilidad normalizada)
        y Z = Σ_{w en candidatos} prob(w).

        En modo 'uniform': equivalente a entropía estándar (todas las probs iguales).
        En modo 'frequency': mejor que entropía estándar porque pondera por prob real.
        """
        candidate_set = set(candidates)
        n_cands = len(candidates)

        # Normalizar probabilidades locales (los candidatos son el universo actual)
        z = sum(self._probs.get(w, 1e-9) for w in candidates)
        local_probs = {w: self._probs.get(w, 1e-9) / z for w in candidates}

        # ── Construir pool de guesses ──────────────────────────────────────────
        # Siempre incluir todos los candidatos (son los mejores guesses potenciales)
        # + una muestra de no-candidatos para exploración informativa
        if n_cands <= _POOL_MAX_CANDIDATES:
            pool_from_candidates = candidates[:]
        else:
            # Muestra estratificada: incluir los más probables + aleatorios
            by_prob = sorted(candidates, key=lambda w: -local_probs[w])
            top = by_prob[: _POOL_MAX_CANDIDATES // 2]
            rest = self._rng.sample(
                by_prob[_POOL_MAX_CANDIDATES // 2 :],
                min(
                    _POOL_MAX_CANDIDATES // 2, len(by_prob) - _POOL_MAX_CANDIDATES // 2
                ),
            )
            pool_from_candidates = top + rest

        non_candidates = [w for w in self._vocab if w not in candidate_set]
        extra_n = min(_POOL_MAX_EXTRA, len(non_candidates))
        pool_extra = self._rng.sample(non_candidates, extra_n) if extra_n > 0 else []

        guess_pool = pool_from_candidates + pool_extra

        # ── Submuestra para evaluación si hay demasiados candidatos ───────────
        if n_cands <= _EVAL_MAX_CANDIDATES:
            eval_cands = candidates
            eval_probs = np.array([local_probs[w] for w in eval_cands])
        else:
            eval_cands = self._rng.sample(candidates, _EVAL_MAX_CANDIDATES)
            raw = np.array([local_probs[w] for w in eval_cands])
            eval_probs = raw / raw.sum()

        # ── Evaluar cada guess ────────────────────────────────────────────────
        best_guess = candidates[0]
        best_entropy = -1.0
        best_is_candidate = candidates[0] in candidate_set

        for g in guess_pool:
            pat_codes = self._feedback_batch(eval_cands, g)
            H = self._weighted_entropy(pat_codes, eval_probs)

            is_cand = g in candidate_set
            # Desempate: preferir candidatos reales (si aciertas, terminas el juego)
            if H > best_entropy or (
                abs(H - best_entropy) < 1e-9 and is_cand and not best_is_candidate
            ):
                best_entropy = H
                best_guess = g
                best_is_candidate = is_cand

        return best_guess

    def _weighted_entropy(self, pat_codes: np.ndarray, prob_arr: np.ndarray) -> float:
        """
        Calcula H_w = -Σ_k P_k log2(P_k) para la partición inducida por pat_codes.

        pat_codes[i] = código del feedback de guess contra eval_cands[i]
        prob_arr[i]  = probabilidad normalizada de eval_cands[i]
        """
        # Sumar masa de probabilidad por grupo
        partition: dict[int, float] = defaultdict(float)
        for code, p in zip(pat_codes.tolist(), prob_arr.tolist()):
            partition[code] += p

        H = 0.0
        for p_k in partition.values():
            if p_k > 1e-15:
                H -= p_k * math.log2(p_k)
        return H

    # ── Feedback vectorizado con numpy ────────────────────────────────────────

    def _feedback_batch(self, candidates: list[str], guess: str) -> np.ndarray:
        """
        Calcula feedback(c, guess) para todos los candidatos en batch.

        Si guess está en el vocabulario usa numpy vectorizado.
        Si es no-palabra usa la función oficial de wordle_env.

        Retorna array int32 de códigos base-3.
        """
        if guess not in self._word_to_idx:
            # No-palabra: fallback a función oficial
            return np.array(
                [_encode_pattern(_fb(c, guess)) for c in candidates], dtype=np.int32
            )

        n = len(candidates)
        wl = self._wl

        cand_indices = np.array(
            [self._word_to_idx[c] for c in candidates], dtype=np.int32
        )
        secrets_mat = self._word_mat[cand_indices]  # (n, wl)
        g_idx = self._word_to_idx[guess]
        guess_vec = self._word_mat[g_idx]  # (wl,)

        pattern = np.zeros((n, wl), dtype=np.int8)

        # ── Paso 1: Verdes ────────────────────────────────────────────────────
        greens = secrets_mat == guess_vec[np.newaxis, :]  # (n, wl) bool
        pattern[greens] = 2

        # ── Paso 2: Amarillos (correctamente vectorizado) ─────────────────────
        # Para cada letra del guess (en orden), verificar si está disponible
        # en el secret (no consumida por verdes anteriores ni amarillos previos).
        #
        # Usamos una matriz 'available' (n, 26) que cuenta cuántas veces
        # cada letra del secret está disponible (no verde).
        available = np.zeros((n, 26), dtype=np.int8)
        for pos in range(wl):
            char_idx = secrets_mat[:, pos].astype(np.int32) - ord("a")
            # Solo contar letras no-verdes en el secret
            not_green = ~greens[:, pos]
            # Indexing seguro: solo letras a-z (índices 0-25)
            valid = not_green & (char_idx >= 0) & (char_idx < 26)
            np.add.at(available, (np.where(valid)[0], char_idx[valid]), 1)

        for pos in range(wl):
            g_char_idx = int(guess_vec[pos]) - ord("a")
            if g_char_idx < 0 or g_char_idx >= 26:
                continue
            not_green = ~greens[:, pos]  # (n,)

            # Candidatos donde esta posición no es verde Y hay disponibilidad
            can_be_yellow = not_green & (available[:, g_char_idx] > 0)
            rows = np.where(can_be_yellow)[0]
            if len(rows) > 0:
                pattern[rows, pos] = 1
                available[rows, g_char_idx] -= 1  # consumir disponibilidad

        # ── Codificar base-3 ──────────────────────────────────────────────────
        codes = (pattern.astype(np.int32) * self._powers3[np.newaxis, :]).sum(axis=1)
        return codes
