"""
Q-learning tabular pe un grid world cu numpy.

Harta:  S = start, G = tinta (+1), H = groapa (-1, episod terminat), . = liber
Fiecare pas costa -0.01, ca agentul sa prefere drumurile scurte.
"""
import numpy as np

HARTA = [
    "S...H",
    ".H...",
    "...H.",
    "H....",
    "..H.G",
]
ACTIUNI = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # sus, jos, stanga, dreapta
SAGETI = ["↑", "↓", "←", "→"]


class GridWorld:
    def __init__(self, harta):
        self.harta = [list(r) for r in harta]
        self.n_rand, self.n_col = len(harta), len(harta[0])
        self.n_stari = self.n_rand * self.n_col
        self.start = self._gaseste("S")

    def _gaseste(self, c):
        for i, r in enumerate(self.harta):
            if c in r:
                return (i, r.index(c))

    def stare(self, poz):
        return poz[0] * self.n_col + poz[1]  # (rand, col) -> index

    def reset(self):
        self.poz = self.start
        return self.stare(self.poz)

    def step(self, a):
        dr, dc = ACTIUNI[a]
        r = min(max(self.poz[0] + dr, 0), self.n_rand - 1)  # peretii blocheaza
        c = min(max(self.poz[1] + dc, 0), self.n_col - 1)
        self.poz = (r, c)
        celula = self.harta[r][c]
        if celula == "G":
            return self.stare(self.poz), 1.0, True
        if celula == "H":
            return self.stare(self.poz), -1.0, True
        return self.stare(self.poz), -0.01, False


def antreneaza(env, episoade=3000, alfa=0.1, gamma=0.99,
               eps_start=1.0, eps_min=0.05, eps_decay=0.998, max_pasi=100, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_stari, len(ACTIUNI)))
    eps, recompense = eps_start, []

    for _ in range(episoade):
        s, total = env.reset(), 0.0
        for _ in range(max_pasi):
            # epsilon-greedy: exploram cu probabilitatea eps
            if rng.random() < eps:
                a = rng.integers(len(ACTIUNI))
            else:
                a = int(np.argmax(Q[s]))
            s2, r, gata = env.step(a)

            # actualizarea Bellman; la stari terminale nu exista viitor
            tinta = r if gata else r + gamma * np.max(Q[s2])
            Q[s, a] += alfa * (tinta - Q[s, a])

            s, total = s2, total + r
            if gata:
                break
        eps = max(eps_min, eps * eps_decay)
        recompense.append(total)
    return Q, np.array(recompense)


def afiseaza_politica(env, Q):
    for i, rand in enumerate(env.harta):
        linie = []
        for j, c in enumerate(rand):
            linie.append(c if c in "GH" else SAGETI[int(np.argmax(Q[env.stare((i, j))]))])
        print(" ".join(linie))


if __name__ == "__main__":
    env = GridWorld(HARTA)
    Q, rec = antreneaza(env)

    for k in range(0, len(rec), 500):
        print(f"episoadele {k:4d}-{k + 499:4d}: recompensa medie {rec[k:k + 500].mean():+.3f}")
    print("\nPolitica invatata:")
    afiseaza_politica(env, Q)

    try:  # grafic optional, daca ai matplotlib
        import matplotlib.pyplot as plt
        medie = np.convolve(rec, np.ones(100) / 100, mode="valid")
        plt.plot(medie)
        plt.xlabel("episod"); plt.ylabel("recompensa (medie mobila 100)")
        plt.title("Q-learning pe grid world"); plt.show()
    except ImportError:
        pass