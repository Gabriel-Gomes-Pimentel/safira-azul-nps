"""Gera pipeline e manifesto privados; demonstracao usa dados artificiais."""

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artefato_modelo import ErroArtefato, sha256, treinar_modelo, salvar_artefato
from dados_sinteticos_operacionais import criar_base_sintetica


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fonte = parser.add_mutually_exclusive_group(required=True)
    fonte.add_argument("--base", type=Path, help="Base analitica Parquet no ambiente autorizado")
    fonte.add_argument("--sintetico", action="store_true", help="Demonstracao artificial; nao utilizavel em passageiros")
    parser.add_argument("--modelo", type=Path, required=True)
    parser.add_argument("--manifesto", type=Path, required=True)
    parser.add_argument("--versao-modelo", required=True)
    parser.add_argument("--versao-fontes", required=True)
    args = parser.parse_args()
    try:
        if args.sintetico:
            base = criar_base_sintetica()
            fingerprint = hashlib.sha256(base.to_json(date_format="iso").encode()).hexdigest()
        else:
            import pandas as pd
            fingerprint = sha256(args.base)
            base = pd.read_parquet(args.base)
        pipeline, proveniencia = treinar_modelo(base)
        registro = salvar_artefato(pipeline, proveniencia, modelo=args.modelo, manifesto=args.manifesto,
                                  versao_modelo=args.versao_modelo, versao_fontes=args.versao_fontes,
                                  fingerprint_base=fingerprint, sintetico=args.sintetico)
        print("Artefato gerado; sha256=" + registro["sha256"] + "; aprovado_producao=false")
        return 0
    except ErroArtefato as erro:
        print("Geracao interrompida: " + str(erro), file=sys.stderr)
        return 3
    except (ValueError, TypeError, KeyError, OSError):
        print("Geracao interrompida: conferir schema, configuracao, fontes e destino privado.", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
