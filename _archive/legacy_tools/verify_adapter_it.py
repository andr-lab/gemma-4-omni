# -*- coding: utf-8 -*-
import os, sys, json

def verify(report_path="eval_adapter_it_report.json", stage2_report_path="stage2adapter_it_report.json"):
    print("=== VERIFICA ADAPTER VOCE ITALIANA (MDC STAGE 2) ===", flush=True)
    all_ok = True

    if not os.path.exists(report_path):
        print(f"[FAIL] Report di valutazione non trovato: {report_path}")
        return False
    
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    status = rep.get("status", "FAIL")
    cos_sim = rep.get("cos_sim", 0.0)
    ce = rep.get("coupling_ce", 99.0)
    rk = rep.get("rank_svd", 0)
    lid = rep.get("language_identification", {})
    verdict_lang = rep.get("verdict_language", rep.get("language_summary", {}).get("verdict", "UNKNOWN"))

    print(f"  - Status Generale: {status}")
    print(f"  - Similarita Coseno (Val): {cos_sim:.4f}")
    print(f"  - Coupling Cross-Entropy: {ce:.4f}")
    print(f"  - Rango SVD: {rk}")
    print(f"  - Verdetto Lingua: {verdict_lang}")

    print("\n--- Risultati Language Identification (LID) ---")
    for tag, scores in lid.items():
        print(f"  [{tag}] Scores: {scores}")
        ita_score = 0.0
        cmn_score = 0.0
        for k, v in scores.items():
            if "ita" in k.lower() or k.lower() == "it":
                ita_score = max(ita_score, float(v))
            if "cmn" in k.lower() or "zh" in k.lower() or "mandarin" in k.lower():
                cmn_score = max(cmn_score, float(v))
        if ita_score < 0.80 or cmn_score > 0.10:
            print(f"    [WARN] Classificazione lingua incerta per {tag}: ita={ita_score}, cmn={cmn_score}")
        else:
            print(f"    [OK] Conferma lingua ITALIANA ({ita_score*100:.1f}%), Bias Mandarino ({cmn_score*100:.2f}%)")

    gates = rep.get("gates", {})
    print("\n--- Gate di Validazione ---")
    for gname, gval in gates.items():
        print(f"  - {gname:25s}: {'PASS' if gval else 'FAIL'}")
        if not gval:
            all_ok = False

    if all_ok and status == "PASS":
        print("\n>>> TUTTI I GATE ADAPTER VOCE ITALIANA SUPERATI CON SUCCESSO! <<<")
        return True
    else:
        print("\n>>> ALMENO UN GATE HA FALLITO LA VERIFICA <<<")
        return False

if __name__ == "__main__":
    rep = sys.argv[1] if len(sys.argv) > 1 else "eval_adapter_it_report.json"
    ok = verify(rep)
    sys.exit(0 if ok else 1)
