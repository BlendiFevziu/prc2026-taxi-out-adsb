"""check_dataset.py - meldet, wenn die Veranstalter die Daten austauschen.

the organiser hat am 01.09.2026 angekuendigt, die fehlenden Julibewegungen nachzuliefern.
Bis dahin lohnt keine ernsthafte Abgabe. Dieses Werkzeug haelt den Stand des Eimers fest
und meldet jede Abweichung: neue Dateien, verschwundene, geaenderte Groesse oder Datum.

Der Hinweis stammt von another participant im Wettbewerbskanal, der vorschlug, Ergebnisse gegen den
Stand der Dateien zu binden.

Aufruf:
    python3 check_dataset.py            vergleicht gegen den letzten Stand
    python3 check_dataset.py --setzen   schreibt den aktuellen Stand als Bezug
"""
import json
import os
import re
import subprocess
import sys

EIMER = "prc26/prc-2026-datasets/"
STAND = "dataset_state.json"


def hole():
    r = subprocess.run(["mc", "ls", EIMER], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        sys.exit(f"mc ls fehlgeschlagen:\n{r.stderr.strip()}")
    aus = {}
    for z in r.stdout.splitlines():
        m = re.match(r"\[(.+?)\]\s+(\S+)\s+\S+\s+(.+)$", z.strip())
        if m:
            aus[m.group(3).strip()] = {"datum": m.group(1), "groesse": m.group(2)}
    return aus


def main(setzen=False):
    jetzt = hole()
    print(f"{len(jetzt)} Objekte in {EIMER}")
    if setzen or not os.path.exists(STAND):
        json.dump(jetzt, open(STAND, "w"), indent=1, sort_keys=True)
        print(f"Stand festgehalten in {STAND}")
        for k in sorted(jetzt):
            print(f"  {jetzt[k]['datum']}  {jetzt[k]['groesse']:>7}  {k}")
        return

    alt = json.load(open(STAND))
    neu = sorted(set(jetzt) - set(alt))
    weg = sorted(set(alt) - set(jetzt))
    aend = sorted(k for k in set(alt) & set(jetzt) if alt[k] != jetzt[k])

    if not (neu or weg or aend):
        print("Unveraendert. Kein Grund, etwas neu zu rechnen.")
        return

    print("\n*** DER DATENSATZ HAT SICH GEAENDERT ***\n")
    for k in neu:
        print(f"  NEU       {jetzt[k]['datum']}  {jetzt[k]['groesse']:>7}  {k}")
    for k in weg:
        print(f"  ENTFERNT  {k}")
    for k in aend:
        print(f"  GEAENDERT {k}")
        print(f"            vorher {alt[k]['datum']}  {alt[k]['groesse']}")
        print(f"            jetzt  {jetzt[k]['datum']}  {jetzt[k]['groesse']}")
    print("\nNaechste Schritte:")
    print("  1. mc cp --recursive prc26/prc-2026-datasets/ daten/")
    print("  2. python3 main_model.py      Modelle neu rechnen")
    print("  3. python3 submission.py           Abgabedatei neu erzeugen")
    print("  4. python3 check_dataset.py --setzen")


if __name__ == "__main__":
    main("--setzen" in sys.argv)
