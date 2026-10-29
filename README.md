# Case: Classificação de Tópicos

## Como Executar

Na raiz deste diretório:

```powershell
python -m pip install -r requirements.txt
python run_final.py
```

O comando usa o corpus externo da OAB somente para ajustar o vocabulário/IDF
do TF-IDF, refaz a validação OOF e gera a entrega final em
`publicacao.csv` e `previsao.txt`.

Em uma máquina comum, a execução leva alguns minutos e depende de acesso à
internet na primeira leitura do parquet da OAB. Os embeddings já estão
versionados em `artefatos/`, portanto não são recalculados.

## Pipeline

1. Carrega e valida os arquivos brutos.
2. Limpa HTML, espaços, quebras e caracteres de controle.
3. Constrói a árvore ativa de temas e valida pai/filho.
4. Gera documentos enriquecidos de temas com caminho, descrição e termos.
5. Extrai entidades jurídicas simples das questões (leis, artigos e referências).
6. Gera os artefatos de preparação e modelagem.

Todo o enriquecimento usado nesta entrega é determinístico e reproduzível.
O `answerKey` e qualquer rótulo externo da OAB são ignorados.

Artefatos:

- `data/`: anexos originais do case.
- `artefatos/preparacao/`: textos limpos, árvore e relatórios de qualidade.
- `artefatos/emb_*.npy`: embeddings BGE-M3 pré-computados.
- `artefatos/termos_temas.txt`: dicionário lexical usado nos perfis.
- `artefatos/modelagem/`: OOF, métricas e classificação completa.

A única decisão manual de escopo é enviar questões cuja disciplina de origem
contém “Militar” para `fora`, pois esse ramo não pertence à taxonomia do
recorte. As demais decisões são geradas pelo pipeline.

## Entrega

- `publicacao.csv`: 659 decisões no formato exigido.
- `previsao.txt`: previsão de precisão e cobertura fora da folha-verdade.
- `relatorio_final.md`: método, validação, diagnóstico e produção.
- `memo_professor.md`: recomendações para revisão da árvore e dos rótulos.
