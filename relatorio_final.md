# Relatório final — classificação de tópicos

## Sistema

O pipeline limpa as questões, valida a árvore de temas e cria três sinais:
TF-IDF de palavras, TF-IDF de caracteres e similaridade dos embeddings BGE-M3.
Os sinais são combinados por pesos aprendidos com os rótulos da folha-verdade.
A matriz hierárquica considera pai e filho como equivalentes para a métrica de
acerto. O threshold é escolhido por validação OOF, buscando o maior conjunto
publicável com precisão estimada de pelo menos 96%.

O corpus `eduagarcia/oab_exams` é um enriquecimento externo lexical: somente
enunciados e alternativas não anuladas ajustam o vocabulário/IDF. `answerKey`,
respostas e rótulos externos não são usados.

## Rótulos e validação

Os 100 registros da folha-verdade são usados para ajustar pesos e estimar o
threshold por OOF repetido. Eles não representam prova independente da precisão
nas 559 questões restantes. A métrica operacional atual é uma estimativa, não
uma garantia populacional.

Com OAB, o experimento obteve aproximadamente 79,8% de top-1, 96,02% de
precisão OOF nas publicações e 40,2% de cobertura OOF. O resultado deve ser
recalculado pelo comando `python run_final.py`.

Foram testadas e descartadas regressão logística, SVM linear e Ridge como
classificadores principais: com apenas 100 rótulos para centenas de temas,
ficaram abaixo do desempenho do ranqueador híbrido. Também foi testada a
redução dos perfis aos 10, 20 e 30 termos mais discriminativos; a precisão
subiu marginalmente, mas a cobertura caiu, então a versão final mantém os
perfis completos.

## Diagnóstico

Os principais riscos são temas irmãos semanticamente próximos, temas genéricos
e assuntos de origem incorretos. Por isso, casos abaixo do threshold vão para
revisão. A análise não força tema quando a disciplina é militar, que fica fora
da taxonomia deste recorte.

Os 10 casos `fora` são definidos pelo gate de disciplina militar; não são
publicações erradas nem entram no conjunto publicado.

## Produção

Os embeddings ficam em cache e a inferência não chama LLM. Em produção, as
correções dos professores devem ser armazenadas como novos exemplos, revisadas
por amostragem e incorporadas em retreinamentos periódicos. O IDF e os
artefatos devem ser versionados; uma nova versão só deve ser promovida após
validação OOF e auditoria de amostras publicadas.

Para 50 mil questões por semana, a vetorização e a multiplicação esparsa são
paralelizáveis e podem ser processadas em lotes. O custo de inferência é
principalmente CPU e armazenamento dos vetores; os embeddings podem ser
gerados em um worker com GPU e reutilizados em cache. Recomenda-se retreino
semanal com as correções aprovadas e uma revisão mensal de threshold, árvore e
qualidade, antecipando-a quando a precisão amostral cair abaixo de 95%.

## Entrega e limitações

`publicacao.csv` contém uma decisão para cada uma das 659 questões.
`previsao.txt` registra a previsão para as questões fora da folha-verdade.
A confiança é um score calibrado operacionalmente pelo threshold, não uma
probabilidade comprovada. O uso externo da OAB deve ser documentado e validado
quanto à licença antes de publicação pública.

O único comando oficial de regeneração é `python run_final.py`. Os artefatos
gerados em `artefatos/` fazem parte desta solução final.
