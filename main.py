## Project de recherche operationnelle - Methode des deux phases
## Auteurs: BAHLAOUI AHMED, LOGRAINE WIAM, RAMADAN ILIAS, ENNACHKLAOUI AYA
## Date: 27 Decembre 2025
## Filiere: INDIA/SD
## Encadre par: Pr. ES-SADEK


import numpy as np


def simplex_solver(tableau):
    """
    Fonction 1: Le Moteur Simplexe.
    Résout un tableau donné (problème de minimisation).
    """
    m, n = tableau.shape

    while True:
        # 1. Vérification de l'optimalité (coûts réduits >= 0)
        if np.all(tableau[-1, :-1] >= -1e-9):
            return tableau, "OPTIMAL"

        # 2. Variable entrante (la plus négative)
        pivot_col = np.argmin(tableau[-1, :-1])

        # 3. Vérification non-borné
        if np.all(tableau[:-1, pivot_col] <= 1e-9):
            return tableau, "UNBOUNDED"

        # 4. Variable sortante (Ratio Test)
        ratios = []
        for i in range(m - 1):
            coeff = tableau[i, pivot_col]
            rhs = tableau[i, -1]
            if coeff > 1e-9:
                ratios.append(rhs / coeff)
            else:
                ratios.append(np.inf)

        pivot_row = np.argmin(ratios)
        if ratios[pivot_row] == np.inf:
            return tableau, "UNBOUNDED"

        # 5. Pivot
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_val

        for i in range(m):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i, :] -= factor * tableau[pivot_row, :]

    return tableau, "ERROR"


def phase1_solve(A, b):
    """
    Fonction 2: Phase 1
    Ajoute les variables artificielles et minimise leur somme.
    """
    m, n = A.shape
    identity = np.eye(m)
    tableau = np.hstack([A, identity, b.reshape(-1, 1)])

    # Fonction objectif W = somme des artificielles (minimisation)
    # W = -sum(contraintes)
    w_row = -np.sum(tableau, axis=0)
    tableau_phase1 = np.vstack([tableau, w_row])

    # Ajustement canonique: les coûts réduits des variables artificielles doivent être 0
    for i in range(m):
        col_idx = n + i
        factor = tableau_phase1[-1, col_idx]
        tableau_phase1[-1, :] -= factor * tableau_phase1[i, :]

    print("\n--- Exécution Phase 1 ---")
    optimal_tableau, status = simplex_solver(tableau_phase1)
    return optimal_tableau, status


def phase2_solve(A, b, c):
    """
    Fonction 3: Phase 2
    Résout le problème original en partant de la base trouvée en Phase 1.
    """
    m, n_orig = A.shape

    # Exécution Phase 1
    tableau_p1, status = phase1_solve(A, b)

    # Vérification faisabilité
    if abs(tableau_p1[-1, -1]) > 1e-6:
        print("PROBLÈME IMPOSSIBLE : Pas de solution réalisable.")
        return None, "INFEASABLE"

    print("Solution réalisable trouvée. Passage à la Phase 2.")

    # Construction tableau Phase 2
    # On garde les variables de décision + variables d'écart (n_orig colonnes) + RHS
    tableau_p2 = tableau_p1[:-1, :n_orig]
    rhs_col = tableau_p1[:-1, -1].reshape(-1, 1)
    tableau_p2 = np.hstack([tableau_p2, rhs_col])

    # Ajout de la fonction objectif originale
    c_row = np.append(c, 0).astype(float)
    tableau_p2 = np.vstack([tableau_p2, c_row])

    # Mise à l'échelle des coûts réduits pour les variables de base
    m_p2, n_p2 = tableau_p2.shape
    for j in range(n_orig):
        col = tableau_p2[:-1, j]
        # Si c'est une colonne de base (un 1 et des 0)
        if np.isclose(np.sum(np.abs(col)), 1) and np.isclose(np.max(col), 1):
            row_idx = np.argmax(col)
            coeff_obj = tableau_p2[-1, j]
            if abs(coeff_obj) > 1e-9:
                tableau_p2[-1, :] -= coeff_obj * tableau_p2[row_idx, :]

    print("\n--- Exécution Phase 2 ---")
    final_tableau, final_status = simplex_solver(tableau_p2)
    return final_tableau, final_status


def saisir_probleme():
    """
    Fonction utilitaire pour guider l'utilisateur et formater les données.
    Convertit automatiquement les inégalités en égalités (ajout variables d'écart).
    """
    print("=== SAISIE DU PROBLÈME ===")

    # 1. Type d'optimisation
    type_opt = input("Type (min/max) : ").strip().lower()
    is_max = type_opt == "max"

    # 2. Variables de décision
    try:
        n_vars = int(input("Nombre de variables de décision : "))
    except ValueError:
        print("Erreur: Entrez un nombre entier.")
        return None

    # 3. Fonction objectif
    print("Entrez les coefficients de la fonction objectif (séparés par espace) :")
    print("Exemple pour 3x1 + 5x2 : 3 5")
    c_input = list(map(float, input("Coefficients C : ").strip().split()))

    if len(c_input) != n_vars:
        print(f"Erreur : attendu {n_vars} coefficients.")
        return None

    c = np.array(c_input)
    if is_max:
        c = -c  # Conversion Max -> Min

    # 4. Contraintes
    try:
        n_contraintes = int(input("Nombre de contraintes : "))
    except ValueError:
        return None

    A_rows = []
    b_rows = []

    # Listes pour gérer les colonnes supplémentaires (slack/surplus)
    slack_columns = []

    print("\nSaisie des contraintes (Format: Coeffs...  Opérateur  RHS)")
    print("Opérateurs acceptés : <=, >=, =")
    print("Exemple: 2 1 <= 10")

    for i in range(n_contraintes):
        print(f"Contrainte {i + 1} :")
        raw_coeffs = list(
            map(float, input(f"  Coefficients (x1..x{n_vars}) : ").strip().split())
        )
        operator = input("  Opérateur (<=, >=, =) : ").strip()
        rhs = float(input("  Valeur droite (RHS) : "))

        if len(raw_coeffs) != n_vars:
            print("Erreur nombre de coefficients.")
            return None

        # Gestion de la forme standard
        # On construit la colonne de slack pour cette ligne
        col_slack = [0.0] * n_contraintes

        if operator == "<=" or operator == "<":
            col_slack[i] = 1.0  # + s
            slack_columns.append(col_slack)
        elif operator == ">=" or operator == ">":
            col_slack[i] = -1.0  # - s (surplus)
            slack_columns.append(col_slack)
        elif operator == "=":
            # Pas de variable d'écart, mais on garde la structure
            pass
        else:
            print("Opérateur inconnu.")
            return None

        A_rows.append(raw_coeffs)
        b_rows.append(rhs)

    # Construction des matrices finales
    A_base = np.array(A_rows)
    b = np.array(b_rows)

    # Ajout des colonnes d'écart (Slack/Surplus) à A et à c
    if slack_columns:
        # Transposer car on a construit liste de colonnes
        S_matrix = np.array(slack_columns).T
        A = np.hstack([A_base, S_matrix])
        # Les variables d'écart ont un coût de 0 dans la fonction objectif
        c = np.concatenate([c, np.zeros(len(slack_columns))])
    else:
        A = A_base

    return A, b, c, is_max


# --- Main ---
if __name__ == "__main__":
    donnees = saisir_probleme()

    if donnees:
        A, b, c, was_max = donnees

        print("\n--- Données Formatées pour le Simplexe ---")
        print(f"Matrice A :\n{A}")
        print(f"Vecteur b : {b}")
        print(f"Vecteur c : {c}")

        result, status = phase2_solve(A, b, c)
        if status == "INFEASIBLE":
            print("No solution")

        if result is not None:
            z_optimal = result[-1, -1]
            if was_max:
                z_optimal = (
                    -z_optimal
                )  # On inverse le signe si c'était une maximisation

            print(f"\nRÉSULTAT FINAL : {status}")
            print(f"Valeur optimale de la fonction objectif : {z_optimal}")

            # Extraction propre des variables (ignorer les slacks)
            # Les variables originales sont les premières colonnes de c (avant extension)
            # Note: c a été étendu avec des 0 pour les slacks, donc on peut afficher tout le vecteur x
            # ou juste les variables de décision initiales.

            print("Variables (y compris variables d'écart/surplus) :")
            vars_vals = np.zeros(len(c))
            for j in range(len(c)):
                col = result[:-1, j]
                if np.isclose(np.sum(np.abs(col)), 1) and np.max(col) > 0.9:
                    row_idx = np.argmax(col)
                    vars_vals[j] = result[row_idx, -1]

            print(vars_vals)
