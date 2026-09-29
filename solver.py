import sys
from heapq import heappop, heappush
from threading import Event
from utils import SATSolverResult, load_formula, lit_to_dimacs


class Solver:
    def __init__(self, filename: str, sigkill: Event):
        self.sigkill = sigkill
        self.formula = load_formula(filename)
        self.num_vars = self.formula.num_vars
        num_lits = self.formula.num_lits

        # Присваивание: values[ℓ] = 1 (истинен), -1 (ложен), 0 (не означен).
        # Хранится и для ℓ, и для ¬ℓ: values[ℓ] == -values[ℓ ^ 1].
        self.values = [0] * num_lits

        # Трейл — означенные литералы в порядке присваивания.
        # trail[:propagated] уже распространены, trail[propagated:] — ещё нет.
        self.trail = []
        self.propagated = 0

        # control[i] — позиция в trail решения уровня i + 1;
        # текущий уровень решения = len(control).
        self.control = []

        self.model = None

        self.preprocess()

    def preprocess(self):
        """
        Разбор дизъюнктов формулы:
          clauses          — дизъюнкты длины ≥ 2, без повторов литералов и тавтологий (a ∨ ¬a ∨ ...)
          units            — литералы единичных дизъюнктов
          has_empty_clause — во входе есть пустой дизъюнкт (формула невыполнима)
        """
        self.clauses = []
        self.units = []
        self.has_empty_clause = False
        for clause in self.formula.clauses:
            lits = set(clause)
            if not lits:
                self.has_empty_clause = True
            elif any(lit ^ 1 in lits for lit in lits):
                continue
            elif len(lits) == 1:
                self.units.append(lits.pop())
            else:
                self.clauses.append(list(lits))

    def level(self) -> int:
        return len(self.control)

    def assign(self, lit: int):
        """Сделать ℓ истинным на текущем уровне."""
        self.values[lit] = 1
        self.values[lit ^ 1] = -1
        self.trail.append(lit)

    def decide(self, lit: int):
        """Открыть новый уровень решения и сделать ℓ истинным."""
        self.control.append(len(self.trail))
        self.assign(lit)

    def decision(self, level: int) -> int:
        """Литерал-решение уровня level (1 ≤ level ≤ self.level())."""
        return self.trail[self.control[level - 1]]

    def backtrack(self, level: int):
        """Отменить все присваивания уровней > level."""
        if level >= len(self.control):
            return
        values, trail = self.values, self.trail
        start = self.control[level]
        for i in range(start, len(trail)):
            lit = trail[i]
            values[lit] = 0
            values[lit ^ 1] = 0
        del trail[start:]
        del self.control[level:]
        self.propagated = start

    def save_model(self):
        values = self.values
        self.model = [lit_to_dimacs(2 * v if values[2 * v] > 0 else 2 * v + 1)
                      for v in range(1, self.num_vars + 1)]

    def build_occurrences(self):
        self.occurrences = [[] for _ in range(self.formula.num_lits)]
        for c in self.clauses:
            for lit in c:
                self.occurrences[lit].append(c)

    def build_watches(self):
        num_lits = self.formula.num_lits
        self.binary = [[] for _ in range(num_lits)]
        self.watches = [[] for _ in range(num_lits)]
        for c in self.clauses:
            if len(c) == 2:
                self.binary[c[0]].append(c[1])
                self.binary[c[1]].append(c[0])
            else:
                self.watches[c[0]].append([c[1], c])
                self.watches[c[1]].append([c[0], c])

    def init_heuristics(self):
        self.act = [0] * (self.num_vars + 1)
        self.phase = [0] * (self.num_vars + 1)
        pos = [0] * (self.num_vars + 1)
        for c in self.clauses:
            for lit in c:
                v = lit >> 1
                self.act[v] += 1
                if lit & 1:
                    pos[v] -= 1
                else:
                    pos[v] += 1
        for u in self.units:
            v = u >> 1
            self.act[v] += 1
            if u & 1:
                pos[v] -= 1
            else:
                pos[v] += 1
        for v in range(1, self.num_vars + 1):
            if pos[v] < 0:
                self.phase[v] = 1

        self.heap = []
        self.in_heap = [False] * (self.num_vars + 1)
        for v in range(1, self.num_vars + 1):
            if self.act[v]:
                heappush(self.heap, (-self.act[v], v))
                self.in_heap[v] = True

    def propagate(self) -> bool:
        """
        UnitPropagate: распространить литералы trail[propagated:].
        Возвращает True, если найден конфликт (все литералы дизъюнкта ложны).
        """
        values, trail = self.values, self.trail
        binary, watches = self.binary, self.watches
        while self.propagated < len(trail):
            p = trail[self.propagated]
            self.propagated += 1
            nf = p ^ 1

            for q in binary[nf]:
                v = values[q]
                if v == 0:
                    values[q] = 1
                    values[q ^ 1] = -1
                    trail.append(q)
                elif v < 0:
                    return True

            ws = watches[nf]
            n = len(ws)
            if not n:
                continue
            i = j = 0
            while i < n:
                entry = ws[i]
                i += 1
                blocker = entry[0]
                if values[blocker] == 1:
                    ws[j] = entry
                    j += 1
                    continue
                c = entry[1]
                pos = 0 if c[0] == nf else 1
                new = -1
                for t in range(2, len(c)):
                    if values[c[t]] != -1:
                        new = t
                        break
                if new != -1:
                    c[pos], c[new] = c[new], c[pos]
                    watches[c[pos]].append([c[pos ^ 1], c])
                    continue
                rest = c[pos ^ 1]
                vr = values[rest]
                if vr == 0:
                    values[rest] = 1
                    values[rest ^ 1] = -1
                    trail.append(rest)
                    entry[0] = rest
                    ws[j] = entry
                    j += 1
                elif vr == 1:
                    ws[j] = entry
                    j += 1
                else:
                    k = n - i + 1
                    ws[j:j + k] = ws[i - 1:]
                    del ws[j + k:]
                    return True
            del ws[j:]

    def choose_literal(self):
        """
        ChooseLiteral: литерал для следующего решения или None, если все
        переменные означены.
        """
        values, heap = self.values, self.heap
        in_heap, phase = self.in_heap, self.phase
        while heap:
            v = heap[0][1]
            if values[2 * v] == 0:
                return 2 * v + 1 if phase[v] else 2 * v
            heappop(heap)
            in_heap[v] = False
        return None

    def solve(self) -> SATSolverResult:
        if self.sigkill.is_set():
            return SATSolverResult.UNKNOWN
        if self.has_empty_clause:
            return SATSolverResult.UNSAT
        self.build_watches()
        self.init_heuristics()

        values, trail = self.values, self.trail
        for u in self.units:
            if values[u] == 0:
                values[u] = 1
                values[u ^ 1] = -1
                trail.append(u)
            elif values[u] < 0:
                return SATSolverResult.UNSAT

        if self.propagate():
            return SATSolverResult.UNSAT
        if self.sigkill.is_set():
            return SATSolverResult.UNKNOWN

        control, in_heap, phase, act = self.control, self.in_heap, self.phase, self.act
        checks = 0
        while True:
            if not self.propagate():
                lit = self.choose_literal()
                if lit is None:
                    self.save_model()
                    return SATSolverResult.SAT
                control.append(len(trail))
                values[lit] = 1
                values[lit ^ 1] = -1
                trail.append(lit)
            else:
                level = len(control)
                if level == 0:
                    return SATSolverResult.UNSAT
                d = trail[control[level - 1]]
                start = control[level - 1]
                for i in range(start, len(trail)):
                    t = trail[i]
                    tv = t >> 1
                    phase[tv] = t & 1
                    values[t] = 0
                    values[t ^ 1] = 0
                    if not in_heap[tv]:
                        heappush(self.heap, (-act[tv], tv))
                        in_heap[tv] = True
                del trail[start:]
                del control[level - 1:]
                self.propagated = start
                nl = d ^ 1
                values[nl] = 1
                values[nl ^ 1] = -1
                trail.append(nl)

            checks += 1
            if not (checks & 0x3FF) and self.sigkill.is_set():
                return SATSolverResult.UNKNOWN


if __name__ == "__main__":
    result = Solver(sys.argv[1], Event()).solve()
    if result == SATSolverResult.SAT:
        print("sat")
    elif result == SATSolverResult.UNSAT:
        print("unsat")
    else:
        print("unknown")
